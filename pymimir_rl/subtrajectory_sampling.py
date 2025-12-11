import pymimir as mm
import random

from abc import ABC, abstractmethod

from .reward_functions import RewardFunction
from .trajectories import Trajectory


class SubtrajectorySampler(ABC):
    """
    Abstract base class for sampling subtrajectories.
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def sample(self, state: mm.State, goal_condition: mm.GroundConjunctiveCondition) -> Trajectory | None:
        """Generate a subtrajectory from the state."""
        pass


class IWSubtrajectorySampler(SubtrajectorySampler):
    def __init__(self, reward_function: RewardFunction, width: int) -> None:
        super().__init__()
        self.reward_function = reward_function
        self.width = width

    def _get_goal_improvement(self,
                              state: mm.State,
                              achieved_goal_condition: list[mm.GroundLiteral],
                              unachieved_goal_condition: list[mm.GroundLiteral]) -> int:
        for literal in achieved_goal_condition:
            if not state.literal_holds(literal):
                return 0
        improvement = 0
        for literal in unachieved_goal_condition:
            if state.literal_holds(literal):
                improvement += 1
        return improvement

    def sample(self, state: mm.State, goal_condition: mm.GroundConjunctiveCondition) -> Trajectory | None:
        achieved_goal_condition: list[mm.GroundLiteral] = []
        unachieved_goal_condition: list[mm.GroundLiteral] = []
        for literal in goal_condition:
            if isinstance(literal, mm.GroundLiteral):
                if (state.literal_holds(literal)): achieved_goal_condition.append(literal)
                else: unachieved_goal_condition.append(literal)
        # Construct the search space graph.
        predecessors: dict[mm.State, tuple[mm.GroundAction, mm.State]] = {}
        distances: dict[mm.State, int] = {state: 0}
        def add_transition(current_state: mm.State, action: mm.GroundAction, _: float, successor_state: mm.State) -> None:
            nonlocal predecessors, distances
            current_distance = distances[current_state]
            successor_distance = current_distance + 1
            if (successor_state not in distances) or (successor_distance < distances[successor_state]):
                distances[successor_state] = successor_distance
                predecessors[successor_state] = (action, current_state)
        mm.iw(state.get_problem(), state, self.width, on_generate_new_state=add_transition)
        # Identify destination candidates.
        # We prioritize large overlaps and then short distances.
        best_improvement: int = 0
        best_distance: int = 0
        best_candidates: list[mm.State] = []
        for state, distance in distances.items():
            improvement = self._get_goal_improvement(state, achieved_goal_condition, unachieved_goal_condition)
            if improvement > best_improvement:
                best_improvement = improvement
                best_distance = distance
                best_candidates.clear()
                best_candidates.append(state)
            elif (improvement == best_improvement) and (distance < best_distance):
                best_candidates.clear()
                best_candidates.append(state)
            elif (improvement == best_improvement) and (distance == best_distance):
                best_candidates.append(state)
        # Uniformly sample a single candidate as the destination.
        # Then backtrack to the initial state to produce a subtrajectory.
        if (len(best_candidates) > 0) and (best_distance > 0):
            destination = random.choice(best_candidates)
            state_sequence = [destination]
            action_sequence = []
            value_sequence = []
            q_value_sequence = []
            reward_sequence = []
            state = destination
            while state in predecessors:
                action, predecessor_state = predecessors[state]
                reward = self.reward_function(predecessor_state, action, state, goal_condition)
                state_sequence.append(predecessor_state)
                action_sequence.append(action)
                value_sequence.append(float('nan'))  # TODO: Get a valid value using a model?
                q_value_sequence.append(float('nan'))  # TODO: Get a valid value using a model?
                reward_sequence.append(reward)
                state = predecessor_state
            state_sequence.reverse()
            action_sequence.reverse()
            value_sequence.reverse()
            q_value_sequence.reverse()
            reward_sequence.reverse()
            return Trajectory(state_sequence, action_sequence, value_sequence, q_value_sequence, reward_sequence, self.reward_function, goal_condition)
        # We were unable to find a single state that overlapped with the goal.
        return None
