import itertools
import math
import networkx as nx
import pymimir as mm
import random

from abc import ABC, abstractmethod
from collections import defaultdict

from .trajectories import Trajectory


class TrajectoryRefiner(ABC):
    """
    Abstract base class for refining trajectories.
    """

    @abstractmethod
    def refine(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """
        Refine the sampled trajectories.

        Args:
            trajectories (list[Trajectory]): A list of sampled trajectories.
            reward_function (RewardFunction): A reward function to compute the rewards along the refined trajectories.

        Returns:
            list[Trajectory]: A refined list of trajectories.
        """
        pass


class IdentityTrajectoryRefiner(TrajectoryRefiner):
    """
    Does not refine the trajectories.
    """

    def refine(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        return trajectories


class StateHindsightTrajectoryRefiner(TrajectoryRefiner):
    def __init__(self, max_generated: int):
        self.max_generated = max_generated

    def refine(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        refined_trajectories: list[Trajectory] = []
        for trajectory in trajectories:
            if len(trajectory) == 0:
                refined_trajectories.append(trajectory)
            else:
                end_index = len(trajectory) - 1
                while end_index >= 0:
                    goal_atoms = trajectory[end_index].successor_state.get_atoms()
                    goal_literals = [mm.GroundLiteral.new(atom, True, trajectory.problem) for atom in goal_atoms]
                    goal_condition = mm.GroundConjunctiveCondition.new(goal_literals, trajectory.problem)
                    closed_set: set[mm.State] = set()
                    start_index = end_index
                    while start_index >= 0:
                        transition = trajectory[start_index]
                        closed_set.add(transition.successor_state)
                        if (transition.current_state in closed_set) or goal_condition.holds(transition.current_state):
                            break
                        start_index -= 1
                    if (start_index + 1) < end_index:
                        refined_trajectories.append(trajectory.clone_with_goal(start_index + 1, end_index, goal_condition))
                    if len(refined_trajectories) >= self.max_generated:
                        return refined_trajectories
                    end_index = start_index - 1
        return refined_trajectories


class PropositionalHindsightTrajectoryRefiner(TrajectoryRefiner):
    def __init__(self, problems: list[mm.Problem], max_generated: int):
        self.problems = problems
        self.max_generated = max_generated
        self.problem_subgoals: dict[mm.Problem, list[mm.GroundConjunctiveCondition]] = {}
        for problem in problems:
            goal_condition = problem.get_goal_condition()
            goal_literals: list[mm.GroundLiteral] = list(goal_condition)  # type: ignore
            # Enumerate all possible combinations of goal_atoms
            subgoals: list[mm.GroundConjunctiveCondition] = []
            for size in range(1, len(goal_literals) + 1):
                list_of_subgoal_literals = itertools.islice(itertools.combinations(goal_literals, size), 100)
                for subgoal_literals in list_of_subgoal_literals:
                    subgoal = mm.GroundConjunctiveCondition.new(list(subgoal_literals), problem)
                    subgoals.append(subgoal)
            subgoals.sort(key=len, reverse=True)
            self.problem_subgoals[problem] = subgoals

    def refine(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        refined_trajectories: list[Trajectory] = []
        for trajectory in trajectories:
            if len(trajectory) == 0:
                refined_trajectories.append(trajectory)
            else:
                assert trajectory.problem in self.problems
                subgoals = self.problem_subgoals[trajectory.problem]
                end_index = len(trajectory) - 1
                while end_index >= 0:
                    for subgoal in subgoals:
                        if subgoal.holds(trajectory[end_index].successor_state):
                            closed_set: set[mm.State] = set()
                            start_index = end_index
                            while start_index >= 0:
                                transition = trajectory[start_index]
                                closed_set.add(transition.successor_state)
                                if (transition.current_state in closed_set) or subgoal.holds(transition.current_state):
                                    break  # Stop when we have found the first "invalid" transition
                                start_index -= 1
                            if (start_index + 1) < end_index:
                                refined_trajectories.append(trajectory.clone_with_goal(start_index + 1, end_index, subgoal))
                            end_index = start_index
                            break  # Only extract the largest subgoal per index
                    if len(refined_trajectories) >= self.max_generated:
                        return refined_trajectories
                    end_index -= 1
        return refined_trajectories


class LiftedHindsightTrajectoryRefiner(TrajectoryRefiner):
    def __init__(self, problems: list[mm.Problem], max_generated: int, max_subgoals_per_size: int = 2):
        self.problems = problems
        self.max_generated = max_generated
        self.problem_subgoals: dict[mm.Problem, list[mm.ConjunctiveCondition]] = {}
        self.problem_blacklist: dict[mm.Problem, list[mm.Predicate]] = {}
        for problem in problems:
            domain = problem.get_domain()
            subgoals: list[mm.ConjunctiveCondition] = []
            blacklist = [domain.get_predicate('=')] if domain.has_predicate('=') else []
            grounded_subgoals = self._generate_grounded_subgoals(problem)
            for subgoal_size in sorted(grounded_subgoals.keys(), reverse=True):
                selected_subgoals = grounded_subgoals[subgoal_size][:max_subgoals_per_size]
                subgoals.extend([subgoal.lift(True) for subgoal in selected_subgoals])
            # Warmup, ground() initializes the generator.
            initial_state = problem.get_initial_state()
            for subgoal in subgoals:
                subgoal.ground(initial_state, 1)
            self.problem_subgoals[problem] = subgoals
            self.problem_blacklist[problem] = blacklist

    def _generate_grounded_subgoals(self, problem: mm.Problem) -> dict[int, list[mm.GroundConjunctiveCondition]]:
        goal_literals: list[mm.GroundLiteral] = list(problem.get_goal_condition())  # type: ignore
        goal_graph = nx.Graph()
        # Add nodes.
        for idx_i in list(range(len(goal_literals))):
            goal_graph.add_node(idx_i)  # type: ignore
        # Add edges.
        for idx_i in range(len(goal_literals)):
            for idx_j in range(idx_i + 1, len(goal_literals)):
                objs_i = goal_literals[idx_i].get_atom().get_terms()
                objs_j = goal_literals[idx_j].get_atom().get_terms()
                if any(o in objs_j for o in objs_i):
                    goal_graph.add_edge(idx_i, idx_j)  # type: ignore
        # Enumerate all connected subcomponents.
        # The variable `all_subcomponents` contains a list of lists of connected subgraphs, one for each original connected component.
        MAX_SUBCOMPONENTS_PER_SIZE = 100
        all_subcomponents = []
        for component in nx.connected_components(goal_graph):  # type: ignore
            subcomponents = []
            for size in range(1, len(component) + 1):  # type: ignore
                size_count = 0
                for subset in itertools.combinations(component, size):  # type: ignore
                    if nx.is_connected(goal_graph.subgraph(subset)):  # type: ignore
                        subcomponents.append(subset)  # type: ignore
                        size_count += 1
                        if size_count >= MAX_SUBCOMPONENTS_PER_SIZE:
                            break
            all_subcomponents.append(subcomponents)  # type: ignore
        # Sample a number of combinations of all subcomponents.
        MAX_SUBGOAL_SIZE = 10
        MAX_SUBCOMPONENT_SAMPLES = 5
        MAX_COMBINATION_SAMPLES = 1000
        sampled_subgraphs = []
        total_subcomponents = len(all_subcomponents)  # type: ignore
        for num_sampled_subcomponents in range(1, min(MAX_SUBGOAL_SIZE, total_subcomponents) + 1):
            num_subcomponent_samples = min(MAX_SUBCOMPONENT_SAMPLES, math.comb(total_subcomponents, num_sampled_subcomponents))
            for sampled_subcomponent_indices in {tuple(sorted(random.sample(range(total_subcomponents), k=num_sampled_subcomponents))) for _ in range(num_subcomponent_samples)}:
                sampled_subcomponents = [all_subcomponents[index] for index in sampled_subcomponent_indices]  # type: ignore
                sampled_subgraphs.extend([sum(combination, ()) for combination in itertools.islice(itertools.product(*sampled_subcomponents), MAX_COMBINATION_SAMPLES)])  # type: ignore
        # Convert the combinations to actual grounded subgoals.
        all_subgoals = [
            mm.GroundConjunctiveCondition.new([goal_literals[i] for i in subgraph], problem)  # type: ignore
            for subgraph in sampled_subgraphs if len(subgraph) <= MAX_SUBGOAL_SIZE  # type: ignore
        ]
        subgoals_by_size = defaultdict(list)  # type: ignore
        for subgoal in all_subgoals:
            subgoals_by_size[len(subgoal)].append(subgoal)  # type: ignore
        return subgoals_by_size  # type: ignore

    def refine(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        refined_trajectories: list[Trajectory] = []
        for trajectory in trajectories:
            assert trajectory.problem in self.problems
            subgoals = self.problem_subgoals[trajectory.problem]
            blacklist = self.problem_blacklist[trajectory.problem]
            end_index = len(trajectory) - 1
            while end_index >= 0:
                for lifted_subgoal in subgoals:
                    grounded_subgoals = lifted_subgoal.ground(trajectory[end_index].successor_state, 1, blacklist)
                    if len(grounded_subgoals) > 0:
                        grounded_subgoal =  grounded_subgoals[0]  # We only ask for one binding.
                        closed_set: set[mm.State] = set()
                        start_index = end_index
                        while start_index >= 0:
                            transition = trajectory[start_index]
                            closed_set.add(transition.successor_state)
                            if (transition.current_state in closed_set) or grounded_subgoal.holds(transition.current_state):
                                break  # Stop when we have found the first "invalid" transition
                            start_index -= 1
                        if (start_index + 1) < end_index:
                            refined_trajectories.append(trajectory.clone_with_goal(start_index + 1, end_index, grounded_subgoal))
                        end_index = start_index
                        break  # Only extract the largest subgoal per index
                if len(refined_trajectories) >= self.max_generated:
                    return refined_trajectories
                end_index -= 1
        return refined_trajectories
