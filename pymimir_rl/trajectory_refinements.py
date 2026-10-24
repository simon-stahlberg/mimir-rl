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


class PartialStateHindsightTrajectoryRefiner(TrajectoryRefiner):
    def __init__(self, max_generated: int, remove_non_goal_predicates: bool = True) -> None:
        super().__init__()
        self.max_generated = max_generated
        self.remove_non_goal_predicates = remove_non_goal_predicates

    def refine(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        refined_trajectories: list[Trajectory] = []
        for trajectory in trajectories:
            if len(trajectory) == 0:
                refined_trajectories.append(trajectory)
            else:
                problem = trajectory.problem
                if self.remove_non_goal_predicates:
                    whitelist_predicates = set(literal.atom.predicate for literal in problem.goal)
                else:
                    whitelist_predicates = set(problem.domain.predicates)
                end_index = len(trajectory) - 1
                while end_index >= 0:
                    goal_atoms = [atom for atom in trajectory[end_index].successor_state.atoms if atom.predicate in whitelist_predicates]
                    if len(goal_atoms) > 0:
                        goal_size = random.randint(1, len(goal_atoms))
                        goal_atoms = random.sample(goal_atoms, goal_size)
                        goal_literals = [problem.ground_literal(atom) for atom in goal_atoms]
                        goal_condition = problem.ground_condition(*goal_literals)
                        closed_set: set[mm.State] = set()
                        start_index = end_index
                        while start_index >= 0:
                            transition = trajectory[start_index]
                            closed_set.add(transition.successor_state)
                            if (transition.current_state in closed_set) or goal_condition.holds(transition.current_state):
                                break
                            start_index -= 1
                        if (start_index + 1) <= end_index:
                            refined_trajectories.append(trajectory.clone_with_goal(start_index + 1, end_index, goal_condition))
                        if len(refined_trajectories) >= self.max_generated:
                            return refined_trajectories
                        end_index = start_index - 1
                    else:
                        end_index -= 1
        return refined_trajectories


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
                    goal_atoms = trajectory[end_index].successor_state.atoms
                    goal_literals = [trajectory.problem.ground_literal(atom) for atom in goal_atoms]
                    goal_condition = trajectory.problem.ground_condition(*goal_literals)
                    closed_set: set[mm.State] = set()
                    start_index = end_index
                    while start_index >= 0:
                        transition = trajectory[start_index]
                        closed_set.add(transition.successor_state)
                        if (transition.current_state in closed_set) or goal_condition.holds(transition.current_state):
                            break
                        start_index -= 1
                    if (start_index + 1) <= end_index:
                        refined_trajectories.append(trajectory.clone_with_goal(start_index + 1, end_index, goal_condition))
                    if len(refined_trajectories) >= self.max_generated:
                        return refined_trajectories
                    end_index = start_index - 1
        return refined_trajectories


def _largest_holding_subgoal(subgoals: list[tuple[int, list[tuple[mm.GroundConjunctiveCondition, int]]]],
                             holding_mask: int) -> tuple[mm.GroundConjunctiveCondition, int] | None:
    """The first subgoal of the largest size whose goal literals all hold, given the bitmask of the goal literals that
    hold, and its own bitmask; the subgoals are by size, largest first."""
    for size, sized_subgoals in subgoals:
        # A subgoal cannot hold if it has more literals than hold.
        if size > holding_mask.bit_count():
            continue
        for subgoal, subgoal_mask in sized_subgoals:
            if subgoal_mask & ~holding_mask == 0:
                return subgoal, subgoal_mask
    return None


class PropositionalHindsightTrajectoryRefiner(TrajectoryRefiner):
    def __init__(self, problems: list[mm.Problem], max_generated: int):
        self.problems = problems
        self.max_generated = max_generated
        # Per problem, a condition per goal literal, and the subgoals by size, largest first, each with the bitmask of
        # its goal literals: a subgoal holds in a state if its literals are among the goal literals that hold there.
        self.problem_literal_conditions: dict[mm.Problem, list[mm.GroundConjunctiveCondition]] = {}
        self.problem_subgoals: dict[mm.Problem, list[tuple[int, list[tuple[mm.GroundConjunctiveCondition, int]]]]] = {}
        for problem in problems:
            goal_literals = list(problem.goal)
            # Enumerate combinations of the goal literals, at most 100 per size.
            subgoals: list[tuple[int, list[tuple[mm.GroundConjunctiveCondition, int]]]] = []
            for size in range(len(goal_literals), 0, -1):
                sized_subgoals: list[tuple[mm.GroundConjunctiveCondition, int]] = []
                for subgoal_indices in itertools.islice(itertools.combinations(range(len(goal_literals)), size), 100):
                    subgoal = problem.ground_condition(*(goal_literals[index] for index in subgoal_indices))
                    assert len(subgoal) == size, "Goal literals must be distinct."
                    sized_subgoals.append((subgoal, sum(1 << index for index in subgoal_indices)))
                subgoals.append((size, sized_subgoals))
            self.problem_literal_conditions[problem] = [problem.ground_condition(literal) for literal in goal_literals]
            self.problem_subgoals[problem] = subgoals

    def refine(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        refined_trajectories: list[Trajectory] = []
        for trajectory in trajectories:
            if len(trajectory) == 0:
                refined_trajectories.append(trajectory)
            else:
                assert trajectory.problem in self.problems
                literal_conditions = self.problem_literal_conditions[trajectory.problem]
                subgoals = self.problem_subgoals[trajectory.problem]
                # The goal literals that hold in each state, computed once per state; states are keyed by identity,
                # as the trajectory keeps them alive and hashing states calls native code.
                holding_masks: dict[int, int] = {}

                def holding_mask(state: mm.State) -> int:
                    mask = holding_masks.get(id(state))
                    if mask is None:
                        mask = sum(1 << index for index, condition in enumerate(literal_conditions) if condition.holds(state))
                        holding_masks[id(state)] = mask
                    return mask

                end_index = len(trajectory) - 1
                while end_index >= 0:
                    # Only extract the largest subgoal per index.
                    match = _largest_holding_subgoal(subgoals, holding_mask(trajectory[end_index].successor_state))
                    if match is not None:
                        subgoal, subgoal_mask = match
                        closed_set: set[mm.State] = set()
                        start_index = end_index
                        while start_index >= 0:
                            transition = trajectory[start_index]
                            closed_set.add(transition.successor_state)
                            if (transition.current_state in closed_set) or subgoal_mask & ~holding_mask(transition.current_state) == 0:
                                break  # Stop when we have found the first "invalid" transition
                            start_index -= 1
                        if (start_index + 1) <= end_index:
                            refined_trajectories.append(trajectory.clone_with_goal(start_index + 1, end_index, subgoal))
                        end_index = start_index
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
            domain = problem.domain
            subgoals: list[mm.ConjunctiveCondition] = []
            blacklist = [domain.predicate('=')] if domain.has_predicate('=') else []
            grounded_subgoals = self._generate_grounded_subgoals(problem)
            for subgoal_size in sorted(grounded_subgoals.keys(), reverse=True):
                selected_subgoals = grounded_subgoals[subgoal_size][:max_subgoals_per_size]
                subgoals.extend([subgoal.lift(add_inequalities=True) for subgoal in selected_subgoals])
            self.problem_subgoals[problem] = subgoals
            self.problem_blacklist[problem] = blacklist

    def _generate_grounded_subgoals(self, problem: mm.Problem) -> dict[int, list[mm.GroundConjunctiveCondition]]:
        goal_literals = list(problem.goal)
        goal_graph: nx.Graph[int] = nx.Graph()
        # Add nodes.
        for idx_i in list(range(len(goal_literals))):
            goal_graph.add_node(idx_i)  # type: ignore
        # Add edges.
        for idx_i in range(len(goal_literals)):
            for idx_j in range(idx_i + 1, len(goal_literals)):
                objs_i = goal_literals[idx_i].atom.arguments
                objs_j = goal_literals[idx_j].atom.arguments
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
            problem.ground_condition(*(goal_literals[i] for i in subgraph))  # type: ignore
            for subgraph in sampled_subgraphs if len(subgraph) <= MAX_SUBGOAL_SIZE  # type: ignore
        ]
        subgoals_by_size = defaultdict(list)  # type: ignore
        for subgoal in all_subgoals:
            subgoals_by_size[len(subgoal)].append(subgoal)  # type: ignore
        return subgoals_by_size  # type: ignore

    @staticmethod
    def _ground_condition(
        condition: mm.ConjunctiveCondition,
        state: mm.State,
        blacklist: list[mm.Predicate],
    ) -> mm.GroundConjunctiveCondition | None:
        groundings = condition.groundings(
            state,
            limit=1,
            omit_predicates=blacklist,
        )
        return groundings[0] if groundings else None

    def refine(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        refined_trajectories: list[Trajectory] = []
        for trajectory in trajectories:
            assert trajectory.problem in self.problems
            subgoals = self.problem_subgoals[trajectory.problem]
            blacklist = self.problem_blacklist[trajectory.problem]
            end_index = len(trajectory) - 1
            while end_index >= 0:
                for lifted_subgoal in subgoals:
                    grounded_subgoal = self._ground_condition(
                        lifted_subgoal,
                        trajectory[end_index].successor_state,
                        blacklist,
                    )
                    if grounded_subgoal is not None:
                        closed_set: set[mm.State] = set()
                        start_index = end_index
                        while start_index >= 0:
                            transition = trajectory[start_index]
                            closed_set.add(transition.successor_state)
                            if (transition.current_state in closed_set) or grounded_subgoal.holds(transition.current_state):
                                break  # Stop when we have found the first "invalid" transition
                            start_index -= 1
                        if (start_index + 1) <= end_index:
                            refined_trajectories.append(trajectory.clone_with_goal(start_index + 1, end_index, grounded_subgoal))
                        end_index = start_index
                        break  # Only extract the largest subgoal per index
                if len(refined_trajectories) >= self.max_generated:
                    return refined_trajectories
                end_index -= 1
        return refined_trajectories
