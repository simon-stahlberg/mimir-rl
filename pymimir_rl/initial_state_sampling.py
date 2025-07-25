import pymimir as mm

from abc import ABC, abstractmethod


class InitialStateSampler(ABC):
    """
    Abstract base class for sampling initial states.
    """

    @abstractmethod
    def sample(self, problems: list[mm.Problem]) -> list[mm.State]:
        """
        Sample initial states from the given problems.

        Args:
            problems (list[mm.Problem]): A list of problem instances.

        Returns:
            list[mm.State]: A list of sampled initial states.
        """
        pass


class OriginalInitialStateSampler(InitialStateSampler):
    """
    Samples the initial state that is specified in the PDDL file.
    """

    def sample(self, problems: list[mm.Problem]) -> list[mm.State]:
        return [problem.get_initial_state() for problem in problems]