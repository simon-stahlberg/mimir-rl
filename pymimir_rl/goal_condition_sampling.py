import pymimir as mm

from abc import ABC, abstractmethod


class GoalConditionSampler(ABC):
    """
    Abstract base class for sampling goal conditions.
    """

    @abstractmethod
    def sample(self, problems: list[mm.Problem]) -> list[mm.GroundConjunctiveCondition]:
        """
        Sample goal conditions from the given problems.

        Args:
            problems (list[mm.Problem]): A list of problem instances.

        Returns:
            list[mm.GroundConjunctiveCondition]: A list of sampled goal conditions.
        """
        pass


class OriginalGoalConditionSampler(GoalConditionSampler):
    """
    Samples the goal condition that is specified in the PDDL file.
    """

    def sample(self, problems: list[mm.Problem]) -> list[mm.GroundConjunctiveCondition]:
        return [problem.get_goal_condition() for problem in problems]