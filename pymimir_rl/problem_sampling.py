import pymimir as mm
import random

from abc import ABC, abstractmethod


class ProblemSampler(ABC):
    """
    Abstract base class for sampling problems.
    """

    @abstractmethod
    def sample(self, problems: list[mm.Problem], n: int) -> list[mm.Problem]:
        """
        Sample problems for the strategy.

        Args:
            problems (list[mm.Problem]): A list of problem instances.
            n (int): The number of problems to sample.

        Returns:
            list[mm.Problem]: A list of sampled problems.
        """
        pass


class UniformProblemSampler(ProblemSampler):
    """
    Samples the problems that are specified in the PDDL file.
    """

    def sample(self, problems: list[mm.Problem], n: int) -> list[mm.Problem]:
        return random.choices(problems, k=n)