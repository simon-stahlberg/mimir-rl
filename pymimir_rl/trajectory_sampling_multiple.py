import pymimir as mm
import random

from typing import Any

from .reward_functions import RewardFunction
from .trajectories import Trajectory
from .trajectory_sampling import TrajectorySampler, TrajectoryState


class MultipleTrajectorySampler(TrajectorySampler):
    """
    Samples trajectories using multiple trajectory samplers with specified probabilities.
    """
    def __init__(self,
                 reward_function: RewardFunction,
                 trajectory_samplers: list[TrajectorySampler],
                 unnormalized_probabilities: list[float]) -> None:
        super().__init__()
        assert isinstance(reward_function, RewardFunction), "reward_function must be an instance of RewardFunction."
        assert isinstance(trajectory_samplers, list), "trajectory_samplers must be a list."
        assert isinstance(unnormalized_probabilities, list), "unnormalized_probabilities must be a list."
        assert len(trajectory_samplers) == len(unnormalized_probabilities), "Number of samplers must match number of probabilities."
        total_unnormalized = sum(unnormalized_probabilities)
        self.reward_function = reward_function
        self.trajectory_samplers = trajectory_samplers
        self.probabilities = [p / total_unnormalized for p in unnormalized_probabilities]
        self.last_sampler_index = -1
        self.last_sampler_states: tuple[list[TrajectoryState], list[Any]] = ([], [])

    def _append_trajectory_states(self, trajectory_states: list[TrajectoryState], subtrajectory_states: list[TrajectoryState]) -> None:
        for trajectory_state, subtrajectory_state in zip(trajectory_states, subtrajectory_states):
            trajectory_state.state_sequence.extend(subtrajectory_state.state_sequence[1:])  # Skip the first state to avoid duplication.
            trajectory_state.action_sequence.extend(subtrajectory_state.action_sequence)
            trajectory_state.value_sequence.extend(subtrajectory_state.value_sequence)
            trajectory_state.q_value_sequence.extend(subtrajectory_state.q_value_sequence)
            trajectory_state.reward_sequence.extend(subtrajectory_state.reward_sequence)
            trajectory_state.solved = subtrajectory_state.solved
            trajectory_state.done = subtrajectory_state.done

    def _initialize(self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]) -> tuple[list[TrajectoryState], list[Any]]:
        trajectory_states, internal_states = [TrajectoryState(state, goal) for state, goal in state_goals], [None for _ in state_goals]
        for trajectory_state in trajectory_states:
            trajectory_state.state_sequence.append(trajectory_state.start_state)
        return trajectory_states, internal_states

    def _internal_sample(self, trajectory_states: list[TrajectoryState], internal_states: list[Any], max_steps: list[int]) -> None:
        # Select a sampler based on the defined probabilities.
        sampler_index = random.choices(range(len(self.trajectory_samplers)), weights=self.probabilities, k=1)[0]
        if sampler_index != self.last_sampler_index:
            # Finalize the subtrajectory of the last sampler and append it to the main trajectory.
            if self.last_sampler_index != -1:
                last_trajectory_states, last_internal_states = self.last_sampler_states
                self.trajectory_samplers[self.last_sampler_index]._finalize(last_trajectory_states, last_internal_states)
                self._append_trajectory_states(trajectory_states, last_trajectory_states)
            # Initialize the new sampler's subtrajectory states.
            subtrajectory_state_goals = [(ts.state_sequence[-1], ts.goal_condition) for ts in trajectory_states]
            self.last_sampler_states = self.trajectory_samplers[sampler_index]._initialize(subtrajectory_state_goals)
            self.last_sampler_index = sampler_index
        # Perform sampling step with the selected sampler.
        subtrajectory_states, subinternal_states = self.last_sampler_states
        remaining_max_steps = [max_step - len(ts.action_sequence) for max_step, ts in zip(max_steps, trajectory_states)]
        self.trajectory_samplers[sampler_index]._internal_sample(subtrajectory_states, subinternal_states, remaining_max_steps)
        # If all subtrajectories are done, finalize and append them.
        if all(subtrajectory_state.done for subtrajectory_state in self.last_sampler_states[0]):
            self.trajectory_samplers[self.last_sampler_index]._finalize(*self.last_sampler_states)
            self._append_trajectory_states(trajectory_states, self.last_sampler_states[0])
            self.last_sampler_index = -1
            self.last_sampler_states = ([], [])

    def _finalize(self, trajectory_states: list[TrajectoryState], internal_states: list[Any]) -> None:
        if self.last_sampler_index != -1:
            last_trajectory_states, last_internal_states = self.last_sampler_states
            self.trajectory_samplers[self.last_sampler_index]._finalize(last_trajectory_states, last_internal_states)
            self._append_trajectory_states(trajectory_states, last_trajectory_states)
            self.last_sampler_index = -1
            self.last_sampler_states = ([], [])

    def sample(self, initial_state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]], horizon: int) -> list[Trajectory]:
        trajectory_states, internal_states = self._initialize(initial_state_goals)
        max_steps = [horizon for _ in trajectory_states]
        while any(not trajectory_state.done for trajectory_state in trajectory_states):
            self._internal_sample(trajectory_states, internal_states, max_steps)
        self._finalize(trajectory_states, internal_states)
        return self._to_trajectories(trajectory_states, self.reward_function)
