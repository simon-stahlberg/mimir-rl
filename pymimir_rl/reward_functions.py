import math
import pymimir as mm

from abc import ABC, abstractmethod
from collections import OrderedDict



class RewardFunction(ABC):
    @abstractmethod
    def __call__(
        self,
        current_state: mm.State,
        action: mm.GroundAction,
        successor_state: mm.State,
        goal_condition: mm.GroundConjunctiveCondition,
    ) -> float:
        pass

    @staticmethod
    def get_dead_end_reward() -> float:
        return -10_000

    def get_value_bounds(self, immediate_reward: float, future_rewards: float, part_of_solution: bool) -> tuple[float, float]:
        return float('-inf'), float('inf')


class SumRewardFunction(RewardFunction):
    def __init__(self, reward_functions: list[RewardFunction]) -> None:
        self.reward_functions = reward_functions

    def __call__(
        self,
        current_state: mm.State,
        action: mm.GroundAction,
        successor_state: mm.State,
        goal_condition: mm.GroundConjunctiveCondition,
    ) -> float:
        total_reward = 0.0
        for reward_function in self.reward_functions:
            total_reward += reward_function(current_state, action, successor_state, goal_condition)
        return total_reward

    def get_value_bounds(self, immediate_reward: float, future_rewards: float, part_of_solution: bool) -> tuple[float, float]:
        min_bound = float('inf')
        max_bound = float('-inf')
        for reward_function in self.reward_functions:
            local_min_bound, local_max_bound = reward_function.get_value_bounds(immediate_reward, future_rewards, part_of_solution)
            min_bound = min(min_bound, local_min_bound)
            max_bound = max(max_bound, local_max_bound)
        return min_bound, max_bound


class GoalTransitionRewardFunction(RewardFunction):
    def __init__(self, constant: float = 1.0) -> None:
        assert constant > 0.0, "Constant must be positive."
        self.constant = constant

    def __call__(
        self,
        current_state: mm.State,
        action: mm.GroundAction,
        successor_state: mm.State,
        goal_condition: mm.GroundConjunctiveCondition,
    ) -> float:
        return self.constant if goal_condition.holds(successor_state) else 0

    def get_value_bounds(self, immediate_reward: float, future_rewards: float, part_of_solution: bool) -> tuple[float, float]:
        return (0.0, self.constant) if part_of_solution else (float('-inf'), float('inf'))


class ConstantRewardFunction(RewardFunction):
    def __init__(self, constant: float = -1.0) -> None:
        assert constant < 0.0, "Constant must be negative."
        self.constant = constant

    def __call__(
        self,
        current_state: mm.State,
        action: mm.GroundAction,
        successor_state: mm.State,
        goal_condition: mm.GroundConjunctiveCondition,
    ) -> float:
        return self.constant

    def get_value_bounds(self, immediate_reward: float, future_rewards: float, part_of_solution: bool) -> tuple[float, float]:
        return (immediate_reward + future_rewards, self.constant) if part_of_solution else (float('-inf'), float('inf'))


class FFRewardFunction(RewardFunction):
    """
    Reward function based on the Fast-Forward (FF) heuristic.
    The reward is computed as the difference in FF heuristic values
    between the current state and the successor state.
    """

    def __init__(self) -> None:
        self.heuristics: dict[mm.Problem, mm.FFHeuristic] = {}
        self.cache: OrderedDict[tuple[mm.State, mm.GroundConjunctiveCondition], float] = OrderedDict()

    def __call__(
        self,
        current_state: mm.State,
        action: mm.GroundAction,
        successor_state: mm.State,
        goal_condition: mm.GroundConjunctiveCondition,
    ) -> float:
        # Obtain or create the FF heuristic for the problem.
        problem = current_state.get_problem()
        if problem not in self.heuristics:
            self.heuristics[problem] = mm.FFHeuristic(problem)
        heuristic = self.heuristics[problem]
        # Obtain the FF value for the current state.
        current_state_goal = (current_state, goal_condition)
        if current_state_goal in self.cache:
            ff_current = self.cache[current_state_goal]
        else:
            ff_current = heuristic.compute_value(current_state, goal_condition)
            self.cache[current_state_goal] = ff_current
        # Obtain the FF value for the successor state.
        successor_state_goal = (successor_state, goal_condition)
        if successor_state_goal in self.cache:
            ff_successor = self.cache[successor_state_goal]
        else:
            ff_successor = heuristic.compute_value(successor_state, goal_condition)
            self.cache[successor_state_goal] = ff_successor
        # Trim the cache to avoid excessive memory usage.
        while len(self.cache) > 10000:
            self.cache.popitem(last=False)
        # Return the difference in FF values as the reward, unless one of them is a dead-end state.
        reward = ff_current - ff_successor
        return reward if math.isfinite(reward) else self.get_dead_end_reward()

    def get_value_bounds(self, immediate_reward: float, future_rewards: float, part_of_solution: bool) -> tuple[float, float]:
        return (immediate_reward + future_rewards, float('inf')) if part_of_solution else (float('-inf'), float('inf'))
