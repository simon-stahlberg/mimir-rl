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


class GoalTransitionRewardFunction(RewardFunction):
    def __init__(self, reward_constant: int = 1) -> None:
        self.constant = reward_constant

    def __call__(
        self,
        current_state: mm.State,
        action: mm.GroundAction,
        successor_state: mm.State,
        goal_condition: mm.GroundConjunctiveCondition,
    ) -> float:
        return self.constant if goal_condition.holds(successor_state) else 0


class ConstantRewardFunction(RewardFunction):
    def __init__(self, penalty_constant: int = -1) -> None:
        self.constant = penalty_constant

    def __call__(
        self,
        current_state: mm.State,
        action: mm.GroundAction,
        successor_state: mm.State,
        goal_condition: mm.GroundConjunctiveCondition,
    ) -> float:
        return self.constant