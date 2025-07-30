import pymimir as mm

from abc import ABC, abstractmethod


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

    def get_value_bounds(self, immediate_reward: float, future_rewards: float, part_of_solution: bool) -> tuple[float, float]:
        return float('-inf'), float('inf')


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
