import pymimir as mm
import torch

from dataclasses import dataclass

from .models import ActionScalarModel
from .reward_functions import RewardFunction
from .trajectories import Trajectory
from .trajectory_sampling import TrajectorySampler, TrajectoryState


@dataclass(frozen=True)
class _BeamCandidate:
    q_value: float
    current_state: mm.State
    action: mm.GroundAction
    reward: float
    successor_state: mm.State
    is_goal: bool
    is_dead_end: bool


class BeamSearchTrajectorySampler(TrajectorySampler):
    """
    Samples trajectories based on the Beam Search strategy.
    """
    DEAD_END_VALUE = RewardFunction.get_dead_end_reward()

    def __init__(self,
                 model: ActionScalarModel,
                 reward_function: RewardFunction,
                 max_beam_size: int) -> None:
        assert isinstance(max_beam_size, int), "Maximum beam size must be an integer."
        assert max_beam_size > 0, "Maximum beam size must be positive."
        super().__init__()
        self.model = model
        self.reward_function = reward_function
        self.max_beam_size = max_beam_size

    class SearchState:
        def __init__(self,
                     initial_state: mm.State,
                     goal_condition: mm.GroundConjunctiveCondition,
                     reward_function: RewardFunction | None = None) -> None:
            self.transition_map: dict[mm.State, tuple[mm.State, mm.GroundAction, float, float]] = {}
            self.value_map: dict[mm.State, float] = {}
            self.beam_list: list[mm.State] = [initial_state]
            self.open_list: list[mm.State] = []
            self.closed_set: set[mm.State] = set([initial_state])
            self.depth: int = 0
            if goal_condition.holds(initial_state):
                self.value_map[initial_state] = 0.0
            elif (len(initial_state.applicable_actions()) == 0 or
                  (reward_function is not None and reward_function.is_dead_end(initial_state, goal_condition))):
                self.value_map[initial_state] = BeamSearchTrajectorySampler.DEAD_END_VALUE
            else:
                self.open_list = [initial_state]

    def _compute_current_state_value(self, q_values: torch.Tensor) -> float:
        return q_values.max().item()

    def _build_candidate_map(self,
                             trajectory_state: TrajectoryState,
                             search_state: SearchState,
                             beam_successor_values: list[tuple[torch.Tensor, list[mm.GroundAction]]]) -> dict[mm.State, _BeamCandidate]:
        # A correctly implemented model returns exactly one aligned result per requested state.
        assert len(beam_successor_values) == len(search_state.open_list), "Model output count must match the number of open states."
        candidate_map: dict[mm.State, _BeamCandidate] = {}
        for open_idx, (successor_values, applicable_actions) in enumerate(beam_successor_values):
            current_state = search_state.open_list[open_idx]
            assert successor_values.ndim == 1, "Beam search requires one scalar Q-value per action."
            assert successor_values.numel() == len(applicable_actions), "Q-values and applicable actions must have equal lengths."
            if len(applicable_actions) == 0:
                search_state.value_map[current_state] = self.DEAD_END_VALUE
                continue

            successor_values = successor_values.cpu()
            successor_states: list[mm.State] = []
            rewards: list[float] = []
            successor_goal_flags: list[bool] = []
            successor_dead_end_flags: list[bool] = []
            for action in applicable_actions:
                successor_state = action.apply(current_state)
                successor_is_goal = trajectory_state.goal_condition.holds(successor_state)
                successor_is_dead_end = (not successor_is_goal) and (
                    len(successor_state.applicable_actions()) == 0 or
                    self.reward_function.is_dead_end(successor_state, trajectory_state.goal_condition)
                )
                successor_states.append(successor_state)
                successor_goal_flags.append(successor_is_goal)
                successor_dead_end_flags.append(successor_is_dead_end)
                rewards.append(self.DEAD_END_VALUE if successor_is_dead_end else self.reward_function(current_state, action, successor_state, trajectory_state.goal_condition))

            search_state.value_map[current_state] = self._compute_current_state_value(successor_values)

            iterator = zip(successor_values,
                           applicable_actions,
                           rewards,
                           successor_states,
                           successor_goal_flags,
                           successor_dead_end_flags)
            for successor_value, action, reward, successor_state, successor_is_goal, successor_is_dead_end in iterator:
                if successor_state in search_state.closed_set:
                    continue
                q_value = successor_value.item()
                candidate = _BeamCandidate(q_value,
                                           current_state,
                                           action,
                                           reward,
                                           successor_state,
                                           successor_is_goal,
                                           successor_is_dead_end)
                existing = candidate_map.get(successor_state)
                if (existing is None) or (candidate.q_value > existing.q_value):
                    candidate_map[successor_state] = candidate
        return candidate_map

    def _beam_step(self, trajectory_state: TrajectoryState, search_state: SearchState, max_depth: int, beam_successor_values: list[tuple[torch.Tensor, list[mm.GroundAction]]]) -> None:
        if search_state.depth >= max_depth:
            trajectory_state.done = True
            return

        candidate_map = self._build_candidate_map(trajectory_state, search_state, beam_successor_values)
        if len(candidate_map) == 0:
            # No more successors to explore.
            trajectory_state.done = True
        else:
            # Select the most promising successors.
            best_candidates = sorted(candidate_map.values(), key=lambda candidate: candidate.q_value, reverse=True)[:self.max_beam_size]
            search_state.beam_list = [candidate.successor_state for candidate in best_candidates]
            search_state.open_list = []
            for candidate in best_candidates:
                search_state.transition_map[candidate.successor_state] = (candidate.current_state, candidate.action, candidate.reward, candidate.q_value)
                if candidate.is_goal:
                    search_state.value_map[candidate.successor_state] = 0.0
                elif candidate.is_dead_end:
                    search_state.value_map[candidate.successor_state] = self.DEAD_END_VALUE
                else:
                    search_state.value_map[candidate.successor_state] = candidate.q_value
                    search_state.open_list.append(candidate.successor_state)
            search_state.closed_set.update(search_state.beam_list)
            search_state.depth += 1
            trajectory_state.solved = any(trajectory_state.goal_condition.holds(state) for state in search_state.beam_list)
            trajectory_state.done = trajectory_state.solved or (search_state.depth >= max_depth) or (len(search_state.open_list) == 0)

    def _initialize(self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]) -> tuple[list[TrajectoryState], list[SearchState]]:
        trajectory_states = [TrajectoryState(state, goal_condition) for state, goal_condition in state_goals]
        for trajectory_state in trajectory_states:
            if (not trajectory_state.solved and
                    self.reward_function.is_dead_end(trajectory_state.start_state, trajectory_state.goal_condition)):
                trajectory_state.done = True
        search_states = [self.SearchState(state, goal_condition, self.reward_function) for state, goal_condition in state_goals]
        return trajectory_states, search_states

    def _internal_sample(self, trajectory_states: list[TrajectoryState], internal_states: list[SearchState], max_steps: list[int]) -> None:
        with torch.no_grad():
            self.model.eval()
            for trajectory_state, search_state, max_step in zip(trajectory_states, internal_states, max_steps):
                if not trajectory_state.done and search_state.depth >= max_step:
                    trajectory_state.done = True
            state_goals_input: list[tuple[mm.State, mm.GroundConjunctiveCondition]] = []
            for trajectory_state, search_state in zip(trajectory_states, internal_states):
                if not trajectory_state.done:
                    state_goals_input.extend([(state, trajectory_state.goal_condition) for state in search_state.open_list])
            if len(state_goals_input) > 0:
                # Evaluate all states in the beams.
                batch_successor_values = self.model.forward(state_goals_input)
                assert len(batch_successor_values) == len(state_goals_input), "Model forward must return one result per input state."
                # Avoid visiting already visited states.
                offset = 0
                for trajectory_state, search_state, max_step in zip(trajectory_states, internal_states, max_steps):
                    if not trajectory_state.done:
                        beam_size = len(search_state.open_list)
                        beam_successor_values = batch_successor_values[offset:offset + beam_size]
                        self._beam_step(trajectory_state, search_state, max_step, beam_successor_values)
                        offset += beam_size

    def _finalize_state(self, trajectory_state: TrajectoryState, search_state: SearchState) -> None:
        assert len(search_state.beam_list) > 0, "Beam search must retain at least one final candidate."
        goal_states = [state for state in search_state.beam_list if trajectory_state.goal_condition.holds(state)]
        if len(goal_states) > 0:
            state = goal_states[0]
        elif len(search_state.open_list) > 0:
            # Prefer an unfinished live branch; return a dead end only when no live branch remains.
            state = search_state.open_list[0]
        else:
            state = search_state.beam_list[0]

        reversed_states = [state]
        reversed_actions: list[mm.GroundAction] = []
        reversed_rewards: list[float] = []
        reversed_q_values: list[float] = []
        while state in search_state.transition_map:
            predecessor_state, action, reward, q_value = search_state.transition_map[state]
            reversed_states.append(predecessor_state)
            reversed_actions.append(action)
            reversed_rewards.append(reward)
            reversed_q_values.append(q_value)
            state = predecessor_state

        trajectory_state.state_sequence.extend(reversed(reversed_states))
        trajectory_state.action_sequence.extend(reversed(reversed_actions))
        trajectory_state.reward_sequence.extend(reversed(reversed_rewards))
        trajectory_state.q_value_sequence.extend(reversed(reversed_q_values))
        trajectory_state.value_sequence.extend(search_state.value_map[current_state] for current_state in trajectory_state.state_sequence[:-1])

    def _finalize(self, trajectory_states: list[TrajectoryState], internal_states: list[SearchState]) -> None:
        for trajectory_state, search_state in zip(trajectory_states, internal_states):
            self._finalize_state(trajectory_state, search_state)

    def sample(self, initial_state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]], horizon: int) -> list[Trajectory]:
        """Generate trajectories for the given instances using the model."""
        assert isinstance(horizon, int), "Horizon must be an integer."
        assert horizon > 0, "Horizon must be positive."
        trajectory_states, search_states = self._initialize(initial_state_goals)
        max_steps = [horizon for _ in trajectory_states]
        while any(not trajectory_state.done for trajectory_state in trajectory_states):
            self._internal_sample(trajectory_states, search_states, max_steps)
        # Create trajectories from contexts.
        self._finalize(trajectory_states, search_states)
        return self._to_trajectories(trajectory_states, self.reward_function)
