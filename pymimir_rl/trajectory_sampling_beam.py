import pymimir as mm
import torch

from dataclasses import dataclass

from .models import ActionScalarModel
from .reward_functions import RewardFunction
from .trajectories import Trajectory
from .trajectory_sampling import TrajectorySampler, TrajectoryState


@dataclass(frozen=True)
class _BeamCandidate:
    priority: float
    q_value: float
    current_state: mm.State
    action: mm.GroundAction
    reward: float
    successor_state: mm.State
    expandable: bool


class BeamSearchTrajectorySampler(TrajectorySampler):
    """
    Samples trajectories based on the Beam Search strategy.
    """
    DEAD_END_VALUE = -10_000.0

    def __init__(self,
                 model: ActionScalarModel,
                 reward_function: RewardFunction,
                 max_beam_size: int) -> None:
        super().__init__()
        self.model = model
        self.reward_function = reward_function
        self.max_beam_size = max_beam_size

    class SearchState:
        def __init__(self, initial_state: mm.State, goal_condition: mm.GroundConjunctiveCondition) -> None:
            self.transition_map: dict[mm.State, tuple[mm.State, mm.GroundAction, float, float]] = {}
            self.value_map: dict[mm.State, float] = {}
            self.beam_list: list[mm.State] = [initial_state]
            self.open_list: list[mm.State] = []
            self.closed_set: set[mm.State] = set([initial_state])
            self.depth: int = 0
            if goal_condition.holds(initial_state):
                self.value_map[initial_state] = 0.0
            elif len(initial_state.generate_applicable_actions()) == 0:
                self.value_map[initial_state] = BeamSearchTrajectorySampler.DEAD_END_VALUE
            else:
                self.open_list = [initial_state]

    def _compute_current_state_value(self, successor_values: torch.Tensor, rewards: list[float]) -> float:
        reward_tensor = torch.tensor(rewards, dtype=torch.float, device=successor_values.device)
        return (successor_values + reward_tensor).max().item()

    def _build_candidate_map(self,
                             trajectory_state: TrajectoryState,
                             search_state: SearchState,
                             beam_successor_values: list[tuple[torch.Tensor, list[mm.GroundAction]]]) -> dict[mm.State, _BeamCandidate]:
        candidate_map: dict[mm.State, _BeamCandidate] = {}
        for open_idx, (successor_values, applicable_actions) in enumerate(beam_successor_values):
            current_state = search_state.open_list[open_idx]
            if len(applicable_actions) == 0:
                search_state.value_map[current_state] = self.DEAD_END_VALUE
                continue

            successor_values = successor_values.cpu()
            successor_states: list[mm.State] = []
            rewards: list[float] = []
            for action in applicable_actions:
                successor_state = action.apply(current_state)
                successor_states.append(successor_state)
                rewards.append(self.reward_function(current_state, action, successor_state, trajectory_state.goal_condition))

            search_state.value_map[current_state] = self._compute_current_state_value(successor_values, rewards)

            for successor_value, action, reward, successor_state in zip(successor_values, applicable_actions, rewards, successor_states):
                if successor_state in search_state.closed_set:
                    continue
                successor_is_goal = trajectory_state.goal_condition.holds(successor_state)
                successor_is_expandable = (not successor_is_goal) and (len(successor_state.generate_applicable_actions()) > 0)
                candidate = _BeamCandidate(successor_value.item(),
                                           successor_value.item(),
                                           current_state,
                                           action,
                                           reward,
                                           successor_state,
                                           successor_is_expandable)
                existing = candidate_map.get(successor_state)
                if (existing is None) or (candidate.priority > existing.priority):
                    candidate_map[successor_state] = candidate
        return candidate_map

    def _beam_step(self, trajectory_state: TrajectoryState, search_state: SearchState, max_depth: int, beam_successor_values: list[tuple[torch.Tensor, list[mm.GroundAction]]]) -> None:
        candidate_map = self._build_candidate_map(trajectory_state, search_state, beam_successor_values)
        if len(candidate_map) == 0:
            # No more successors to explore.
            trajectory_state.done = True
        else:
            # Select the most promising successors.
            best_candidates = sorted(candidate_map.values(), key=lambda candidate: candidate.priority, reverse=True)[:self.max_beam_size]
            search_state.beam_list = [candidate.successor_state for candidate in best_candidates]
            search_state.open_list = []
            for candidate in best_candidates:
                search_state.transition_map[candidate.successor_state] = (candidate.current_state, candidate.action, candidate.reward, candidate.q_value)
                if trajectory_state.goal_condition.holds(candidate.successor_state):
                    search_state.value_map[candidate.successor_state] = 0.0
                elif candidate.expandable:
                    search_state.value_map[candidate.successor_state] = candidate.q_value
                    search_state.open_list.append(candidate.successor_state)
                else:
                    search_state.value_map[candidate.successor_state] = self.DEAD_END_VALUE
            search_state.closed_set.update(search_state.beam_list)
            search_state.depth += 1
            trajectory_state.solved = any(trajectory_state.goal_condition.holds(state) for state in search_state.beam_list)
            trajectory_state.done = trajectory_state.solved or (search_state.depth >= max_depth) or (len(search_state.open_list) == 0)

    def _initialize(self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]) -> tuple[list[TrajectoryState], list[SearchState]]:
        trajectory_states = [TrajectoryState(state, goal_condition) for state, goal_condition in state_goals]
        search_states = [self.SearchState(state, goal_condition) for state, goal_condition in state_goals]
        return trajectory_states, search_states

    def _internal_sample(self, trajectory_states: list[TrajectoryState], internal_states: list[SearchState], max_steps: list[int]) -> None:
        with torch.no_grad():
            self.model.eval()
            state_goals_input: list[tuple[mm.State, mm.GroundConjunctiveCondition]] = []
            for trajectory_state, search_state in zip(trajectory_states, internal_states):
                if not trajectory_state.done:
                    state_goals_input.extend([(state, trajectory_state.goal_condition) for state in search_state.open_list])
            if len(state_goals_input) > 0:
                # Evaluate all states in the beams.
                batch_successor_values = self.model.forward(state_goals_input)
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
        for state in search_state.beam_list:
            assert state in search_state.value_map, "Beam states must have associated values during finalization."

        final_candidates = [(int(trajectory_state.goal_condition.holds(state)), search_state.value_map[state], state) for state in search_state.beam_list]
        final_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, _, state = final_candidates[0]

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
        trajectory_states, search_states = self._initialize(initial_state_goals)
        max_steps = [horizon for _ in trajectory_states]
        while any(not trajectory_state.done for trajectory_state in trajectory_states):
            self._internal_sample(trajectory_states, search_states, max_steps)
        # Create trajectories from contexts.
        self._finalize(trajectory_states, search_states)
        return self._to_trajectories(trajectory_states, self.reward_function)
