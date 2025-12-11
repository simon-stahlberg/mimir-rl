import pymimir as mm
import torch

from .models import ActionScalarModel
from .reward_functions import RewardFunction
from .trajectories import Trajectory
from .trajectory_sampling import TrajectorySampler, TrajectoryState


class BeamSearchTrajectorySampler(TrajectorySampler):
    """
    Samples trajectories based on the Beam Search strategy.
    """
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
            self.open_list: list[mm.State] = [initial_state]
            self.closed_set: set[mm.State] = set([initial_state])
            self.depth: int = 0
            if goal_condition.holds(initial_state):
                self.value_map[initial_state] = 0.0

    def _beam_step(self, trajectory_state: TrajectoryState, search_state: SearchState, max_depth: int, beam_successor_values: list[tuple[torch.Tensor, list[mm.GroundAction]]]) -> None:
        # Add current open list states to closed set.
        for state in search_state.open_list:
            search_state.closed_set.add(state)
        # Expand each beam.
        unique_successors = set()
        all_candidates: list[tuple[float, float, mm.State, mm.GroundAction, float, mm.State]] = []
        for open_idx, (successor_values, applicable_actions) in enumerate(beam_successor_values):
            successor_values = successor_values.cpu()
            priorities = successor_values.clone()
            current_state = search_state.open_list[open_idx]
            successor_states: list[mm.State] = []
            rewards: list[float] = []
            # Avoid actions leading to already visited states.
            for action_idx, action in enumerate(applicable_actions):
                successor_state = action.apply(current_state)
                successor_states.append(successor_state)
                rewards.append(self.reward_function(current_state, action, successor_state, trajectory_state.goal_condition))
                if successor_state in search_state.closed_set:
                    priorities[action_idx] = -1000000.0
            # Add the value of the current state if not already present.
            if current_state not in search_state.value_map:
                current_value = (successor_values + torch.tensor(rewards, dtype=torch.float, device=successor_values.device)).max().item()
                search_state.value_map[current_state] = current_value
            # Sample an action to apply.
            for priority, successor_value, action, reward, successor_state in zip(priorities, successor_values, applicable_actions, rewards, successor_states):
                if (successor_state not in unique_successors) and (len(successor_state.generate_applicable_actions()) > 0):
                    unique_successors.add(successor_state)
                    all_candidates.append((priority.item(), successor_value.item(), current_state, action, reward, successor_state))
        # Select the most promising successors.
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        best_candidates = all_candidates[:self.max_beam_size]
        # Update the beam states.
        for _, successor_value, current_state, action, reward, successor_state in best_candidates:
            if successor_state not in search_state.closed_set:
                search_state.transition_map[successor_state] = (current_state, action, reward, successor_value)
        search_state.open_list = [candidate[5] for candidate in best_candidates]
        search_state.depth += 1
        trajectory_state.solved = any(trajectory_state.goal_condition.holds(state) for state in search_state.open_list)
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
        # Approximate the missing values of the final states by using the q-values.
        for state in search_state.open_list:
            if state not in search_state.value_map:
                assert state in search_state.transition_map
                _, _, _, successor_value = search_state.transition_map[state]
                search_state.value_map[state] = successor_value
        # Create trajectory from the best state in the final open list.
        final_candidates = [(int(trajectory_state.goal_condition.holds(state)), search_state.value_map[state], state) for state in search_state.open_list]
        final_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, value, state = final_candidates[0]
        trajectory_state.state_sequence.append(state)
        trajectory_state.value_sequence.append(value)
        while state in search_state.transition_map:
            predecessor_state, action, reward, q_value = search_state.transition_map[state]
            trajectory_state.state_sequence.append(predecessor_state)
            trajectory_state.action_sequence.append(action)
            trajectory_state.reward_sequence.append(reward)
            trajectory_state.q_value_sequence.append(q_value)
            trajectory_state.value_sequence.append(value)
            state = predecessor_state
        trajectory_state.state_sequence.reverse()
        trajectory_state.action_sequence.reverse()
        trajectory_state.reward_sequence.reverse()
        trajectory_state.q_value_sequence.reverse()
        trajectory_state.value_sequence.reverse()

    def _finalize(self, trajectory_states: list[TrajectoryState], internal_states: list[SearchState]) -> None:
        for trajectory_state, search_state in zip(trajectory_states, internal_states):
            self._finalize_state(trajectory_state, search_state)

    def sample(self, initial_state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]], horizon: int) -> list[Trajectory]:
        """Generate trajectories for the given instances using the model."""
        trajectory_states, search_states = self._initialize(initial_state_goals)
        while any(not trajectory_state.done for trajectory_state in trajectory_states):
            max_steps = [horizon for _ in trajectory_states]
            self._internal_sample(trajectory_states, search_states, max_steps)
        # Create trajectories from contexts.
        self._finalize(trajectory_states, search_states)
        return self._to_trajectories(trajectory_states, self.reward_function)
