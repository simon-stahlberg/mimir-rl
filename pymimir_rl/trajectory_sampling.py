import pymimir as mm

from abc import ABC, abstractmethod
from typing import Any

from .reward_functions import RewardFunction
from .trajectories import Trajectory


class TrajectoryState:
    """
    Represents the state of a trajectory during sampling.
    """
    def __init__(self, start_state: mm.State, goal_condition: mm.GroundConjunctiveCondition) -> None:
        self.start_state: mm.State = start_state
        self.goal_condition: mm.GroundConjunctiveCondition = goal_condition
        self.state_sequence: list[mm.State] = []
        self.action_sequence: list[mm.GroundAction] = []
        self.value_sequence: list[float] = []
        self.q_value_sequence: list[float] = []
        self.reward_sequence: list[float] = []
        self.solved: bool = goal_condition.holds(start_state)
        self.done: bool = self.solved or (len(start_state.generate_applicable_actions()) == 0)


class TrajectorySampler(ABC):
    """
    Abstract base class for sampling trajectories.
    """
    def _to_trajectory(self, trajectory_state: TrajectoryState, reward_function: RewardFunction) -> Trajectory:
        return Trajectory(
            trajectory_state.state_sequence,
            trajectory_state.action_sequence,
            trajectory_state.value_sequence,
            trajectory_state.q_value_sequence,
            trajectory_state.reward_sequence,
            reward_function,
            trajectory_state.goal_condition,
        )

    def _to_trajectories(self, trajectory_states: list[TrajectoryState], reward_function: RewardFunction) -> list[Trajectory]:
        return [self._to_trajectory(ts, reward_function) for ts in trajectory_states]

    @abstractmethod
    def _initialize(self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]) -> tuple[list[TrajectoryState], list[Any]]:
        """
        Create trajectory and internal states for the given states and goal conditions.

        Args:
            state_goals (list[tuple[mm.State, mm.GroundConjunctiveCondition]]): A list of states and goal conditions.

        Returns:
            tuple[list[TrajectoryState], list[Any]]: The created trajectory states and internal states.
        """
        pass

    @abstractmethod
    def _internal_sample(self, trajectory_states: list[TrajectoryState], internal_states: list[Any], max_steps: list[int]) -> None:
        """
        Perform sampling for the given trajectories using the model.

        Args:
            trajectory_states (list[TrajectoryState]): The trajectory states of the trajectories to extend.
            internal_states (list[Any]): The internal states of the sampler for each trajectory.
            max_steps (list[int]): The maximum to extend the trajectories to.
        """

    @abstractmethod
    def _finalize(self, trajectory_states: list[TrajectoryState], internal_states: list[Any]) -> None:
        """
        Finalize the sampling for the given trajectories.

        Args:
            trajectory_states (list[TrajectoryState]): The trajectory states of the trajectories to finalize.
            internal_states (list[Any]): The internal states of the sampler for each trajectory.
        """
        pass

    @abstractmethod
    def sample(self, initial_state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]], horizon: int) -> list[Trajectory]:
        """
        Generate trajectories for the given instances.

        Args:
            initial_state_goals (list[tuple[mm.State, mm.GroundConjunctiveCondition]]): A list of initial states and goal conditions.
            horizon (int): The maximum horizon to sample the trajectories to.
        """
