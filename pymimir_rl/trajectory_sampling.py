import pymimir as mm
import random
import torch

from abc import ABC, abstractmethod
from collections import defaultdict

from .models import ActionScalarModel
from .reward_functions import RewardFunction
from .subtrajectory_sampling import SubtrajectorySampler
from .trajectories import Trajectory


class TrajectorySampler(ABC):
    """
    Abstract base class for sampling trajectories.
    """

    @abstractmethod
    def sample(self, initial_state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]], horizon: int) -> list[Trajectory]:
        """Generate trajectories for the given instances using the model."""


class PolicyRolloutSampler(TrajectorySampler):
    """
    Abstract base class for sampling trajectories.
    """
    def __init__(self,
                 model: ActionScalarModel,
                 reward_function: RewardFunction,
                 subtrajectory_sampler: SubtrajectorySampler | None,
                 subtrajectory_sampling_probability: float) -> None:
        assert isinstance(model, ActionScalarModel), "Model must be an instance of ActionScalarModel."
        assert isinstance(reward_function, RewardFunction), "Reward function must be an instance of RewardFunction."
        assert isinstance(subtrajectory_sampler, SubtrajectorySampler) or (subtrajectory_sampler is None), "Subtrajectory sampler must be an instance of SubtrajectorySampler or None."
        assert isinstance(subtrajectory_sampling_probability, float), "Subtrajectory sampling probability must be a float."
        assert subtrajectory_sampling_probability >= 0.0, "Subtrajectory sampling probability must be at least 0.0."
        assert subtrajectory_sampling_probability <= 1.0, "Subtrajectory sampling probability must be at most 1.0."
        super().__init__()
        self.model = model
        self.reward_function = reward_function
        self.subtrajectory_sampler = subtrajectory_sampler
        self.subtrajectory_sampling_probability = subtrajectory_sampling_probability

    class RolloutContext:
        def __init__(self, initial_state: mm.State, goal_condition: mm.GroundConjunctiveCondition) -> None:
            self.state_sequence: list[mm.State] = [initial_state]
            self.action_sequence: list[mm.GroundAction] = []
            self.value_sequence: list[float] = []
            self.q_value_sequence: list[float] = []
            self.reward_sequence: list[float] = []
            self.closed_set: set[mm.State] = set([initial_state])
            self.goal_condition: mm.GroundConjunctiveCondition = goal_condition
            self.solved: bool = goal_condition.holds(initial_state)
            self.done = self.solved or (len(initial_state.generate_applicable_actions()) == 0)

    @abstractmethod
    def sample_action_index(self, state: mm.State, actions: list[mm.GroundAction], values: torch.Tensor) -> int:
        """
        Sample an action index.

        Args:
            values (Tensor): A tensor of values, e.g., Q-values or logits.

        Returns:
            int: An index of an action.
        """
        pass

    def sample(self, initial_state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]], horizon: int) -> list[Trajectory]:
        """Generate trajectories for the given instances using the model."""
        rollout_contexts = [self.RolloutContext(initial_state, goal_condition) for initial_state, goal_condition in initial_state_goals]
        with torch.no_grad():
            self.model.eval()
            for _ in range(horizon):
                state_goals = [(context.state_sequence[-1], context.goal_condition) for context in rollout_contexts if not context.done]
                context_indices = [idx for idx, context in enumerate(rollout_contexts) if not context.done]
                if len(state_goals) == 0:
                    break  # All trajectories are done.
                q_values_batch = self.model.forward(state_goals)
                for rollout_idx, (q_values, applicable_actions) in zip(context_indices, q_values_batch):
                    context = rollout_contexts[rollout_idx]
                    rewards: list[float] = []
                    q_values = q_values.cpu()  # Move the result to CPU, the remaining operations are very cheap.
                    q_values_copy = q_values.clone()
                    current_state = context.state_sequence[-1]
                    # Reduce the value actions leading to already visited states.
                    for action_idx, action in enumerate(applicable_actions):
                        successor_state = action.apply(current_state)
                        rewards.append(self.reward_function(current_state, action, successor_state, context.goal_condition))
                        if successor_state in context.closed_set:
                            q_values_copy[action_idx] = -1000000.0
                    # Sample an action to apply.
                    action_idx = self.sample_action_index(current_state, applicable_actions, q_values_copy)
                    action = applicable_actions[action_idx]
                    successor_state = action.apply(current_state)
                    q_value = q_values[action_idx].item()
                    reward = rewards[action_idx]
                    value = (q_values + torch.tensor(rewards, dtype=torch.float, device=q_values.device)).max().item()
                    is_solved = context.goal_condition.holds(successor_state)
                    is_dead_end = len(successor_state.generate_applicable_actions()) == 0
                    exceeds_horizon = len(context.state_sequence) >= horizon
                    is_done = is_solved or is_dead_end or exceeds_horizon
                    # Update context.
                    context.state_sequence.append(successor_state)
                    context.action_sequence.append(action)
                    context.value_sequence.append(value)
                    context.q_value_sequence.append(q_value)
                    context.reward_sequence.append(reward)
                    context.closed_set.add(successor_state)
                    context.solved = is_solved
                    context.done = is_done
                # If enabled, sample a subtrajectory
                if (self.subtrajectory_sampler is not None) and (random.uniform(0.0, 1.0 - 1E-16) < self.subtrajectory_sampling_probability):
                    for rollout_idx in context_indices:
                        context = rollout_contexts[rollout_idx]
                        if not context.done:
                            start_state = context.state_sequence[-1]
                            goal_condition = context.goal_condition
                            subtrajectory = self.subtrajectory_sampler.sample(start_state, goal_condition)
                            if subtrajectory is not None:
                                for transition in subtrajectory:
                                    context.state_sequence.append(transition.successor_state)
                                    context.action_sequence.append(transition.selected_action)
                                    context.value_sequence.append(transition.predicted_value)
                                    context.q_value_sequence.append(transition.predicted_q_value)
                                    context.reward_sequence.append(transition.immediate_reward)
                                    context.closed_set.add(transition.successor_state)
                                final_state = context.state_sequence[-1]
                                is_solved = goal_condition.holds(final_state)
                                is_dead_end = len(final_state.generate_applicable_actions()) == 0
                                exceeds_horizon = len(context.state_sequence) >= horizon
                                is_done = is_solved or is_dead_end or exceeds_horizon
                                context.solved = is_solved
                                context.done = is_done
        # Create trajectories from contexts.
        trajectories: list[Trajectory] = []
        for context in rollout_contexts:
            trajectories.append(
                Trajectory(
                    context.state_sequence,
                    context.action_sequence,
                    context.value_sequence,
                    context.q_value_sequence,
                    context.reward_sequence,
                    self.reward_function,
                    context.goal_condition,
                )
            )
        return trajectories


class PolicyTrajectorySampler(PolicyRolloutSampler):
    """
    Values are treated as logits, and an action is sampled according to the resulting probability distribution.
    """

    def __init__(self,
                 model: ActionScalarModel,
                 reward_function: RewardFunction,
                 subtrajectory_sampler: SubtrajectorySampler | None = None,
                 subtrajectory_sampling_probability: float = 0.1) -> None:
        super().__init__(model, reward_function, subtrajectory_sampler, subtrajectory_sampling_probability)

    def sample_action_index(self, state: mm.State, actions: list[mm.GroundAction], values: torch.Tensor) -> int:
        probabilities = values.softmax(0)
        action_index = probabilities.multinomial(num_samples=1)
        return action_index.item()  # type: ignore


class EpsilonGreedyTrajectorySampler(PolicyRolloutSampler):
    """
    Samples trajectories according to epsilon-greedy.
    """

    def __init__(self,
                 model: ActionScalarModel,
                 reward_function: RewardFunction,
                 epsilon: float,
                 subtrajectory_sampler: SubtrajectorySampler | None = None,
                 subtrajectory_sampling_probability: float = 0.1) -> None:
        assert isinstance(epsilon, float), "Epsilon must be a float."
        assert epsilon >= 0.0, "Epsilon must be a probability."
        assert epsilon <= 1.0, "Epsilon must be a probability."
        super().__init__(model, reward_function, subtrajectory_sampler, subtrajectory_sampling_probability)
        self.epsilon = epsilon

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def get_epsilon(self) -> float:
        return self.epsilon

    def sample_action_index(self, state: mm.State, actions: list[mm.GroundAction], values: torch.Tensor) -> int:
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, values.numel(), (1,)).item()  # type: ignore
        else:
            return torch.argmax(values).item()  # type: ignore


class BoltzmannTrajectorySampler(PolicyRolloutSampler):
    """
    Samples trajectories based on a Boltzmann distribution.
    """

    def __init__(self,
                 model: ActionScalarModel,
                 reward_function: RewardFunction,
                 temperature: float,
                 subtrajectory_sampler: SubtrajectorySampler | None = None,
                 subtrajectory_sampling_probability: float = 0.1) -> None:
        assert isinstance(temperature, float), "Temperature must be a float."
        assert temperature > 0.0, "Temperature must be positive."
        super().__init__(model, reward_function, subtrajectory_sampler, subtrajectory_sampling_probability)
        self.temperature = temperature

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    def get_temperature(self) -> float:
        return self.temperature

    def sample_action_index(self, state: mm.State, actions: list[mm.GroundAction], values: torch.Tensor) -> int:
        probabilities = (values / self.temperature).softmax(dim=0)
        action_index = probabilities.multinomial(num_samples=1)
        return action_index.item()  # type: ignore


class StateBoltzmannTrajectorySampler(PolicyRolloutSampler):
    """
    Samples trajectories based on a Boltzmann distribution, that takes into account state counts.
    """

    def __init__(self,
                 model: ActionScalarModel,
                 reward_function: RewardFunction,
                 initial_temperature: float,
                 final_temperature: float,
                 temperature_steps: int,
                 subtrajectory_sampler: SubtrajectorySampler | None = None,
                 subtrajectory_sampling_probability: float = 0.1) -> None:
        super().__init__(model, reward_function, subtrajectory_sampler, subtrajectory_sampling_probability)
        assert isinstance(initial_temperature, float), "Initial temperature must be a float."
        assert isinstance(final_temperature, float), "Final temperature must be a float."
        assert isinstance(temperature_steps, int), "Temperature steps must be an integer."
        assert initial_temperature > 0.0, "Initial temperature must be positive."
        assert final_temperature > 0.0, "Final temperature must be positive."
        assert temperature_steps > 0, "Temperature steps must be positive."
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature_steps = temperature_steps
        self.counts: defaultdict[mm.State, int] = defaultdict(int)

    def update_counts(self, trajectories: list[Trajectory]) -> None:
        # We do not count the final state in the state sequence, as no decision was taken in it.
        for trajectory in trajectories:
            for transition in trajectory:
                self.counts[transition.current_state] += 1

    def sample_action_index(self, state: mm.State, actions: list[mm.GroundAction], values: torch.Tensor) -> int:
        state_counts = self.counts[state]
        ratio = min(1.0, state_counts / self.temperature_steps)
        temperature = self.initial_temperature * (1.0 - ratio) + self.final_temperature * ratio
        probabilities = (values / temperature).softmax(dim=0)
        action_idx = probabilities.multinomial(num_samples=1)
        return action_idx.item()  # type: ignore


class GreedyPolicyTrajectorySampler(PolicyRolloutSampler):
    """
    Samples trajectories based on greedily and deterministically following the policy.
    """

    def __init__(self,
                 model: ActionScalarModel,
                 reward_function: RewardFunction,
                 subtrajectory_sampler: SubtrajectorySampler | None = None,
                 subtrajectory_sampling_probability: float = 0.1) -> None:
        super().__init__(model, reward_function, subtrajectory_sampler, subtrajectory_sampling_probability)

    def sample_action_index(self, state: mm.State, actions: list[mm.GroundAction], values: torch.Tensor) -> int:
        action_index = values.argmax().item()
        return action_index  # type: ignore


class PolicySearchSampler(TrajectorySampler):
    """
    Abstract base class for sampling trajectories using search strategies.
    """

    @abstractmethod
    def sample(self, initial_state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]], horizon: int) -> list[Trajectory]:
        """Generate trajectories for the given instances using the model."""


# class GBFSTrajectorySampler(PolicySearchSampler):
#     """
#     Samples trajectories based on the Greedy Best-First Search (GBFS) strategy.
#     """

#     def __init__(self,
#                  model: ActionScalarModel,
#                  reward_function: RewardFunction) -> None:
#         super().__init__()
#         self.model = model
#         self.reward_function = reward_function

#     def sample(self, initial_state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]], horizon: int) -> list[Trajectory]:
#         """Generate trajectories for the given instances using the model."""
#         raise NotImplementedError("GBFSTrajectorySampler is not implemented yet.")


class BeamSearchTrajectorySampler(PolicySearchSampler):
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

    class SearchContext:
        def __init__(self, initial_state: mm.State, goal_condition: mm.GroundConjunctiveCondition) -> None:
            self.transition_map: dict[mm.State, tuple[mm.State, mm.GroundAction, float, float]] = {}
            self.value_map: dict[mm.State, float] = {}
            self.open_list: list[mm.State] = [initial_state]
            self.closed_set: set[mm.State] = set([initial_state])
            self.goal_condition: mm.GroundConjunctiveCondition = goal_condition
            self.solved: bool = goal_condition.holds(initial_state)
            self.done: bool = self.solved or (len(initial_state.generate_applicable_actions()) == 0)
            self.depth: int = 0
            if self.solved:
                self.value_map[initial_state] = 0.0

    def _beam_step(self, context: SearchContext, horizon: int, beam_successor_values: list[tuple[torch.Tensor, list[mm.GroundAction]]]) -> None:
        # Add current open list states to closed set.
        for state in context.open_list:
            context.closed_set.add(state)
        # Expand each beam.
        unique_successors = set()
        all_candidates: list[tuple[float, float, mm.State, mm.GroundAction, float, mm.State]] = []
        for open_idx, (successor_values, applicable_actions) in enumerate(beam_successor_values):
            successor_values = successor_values.cpu()
            priorities = successor_values.clone()
            current_state = context.open_list[open_idx]
            successor_states: list[mm.State] = []
            rewards: list[float] = []
            # Avoid actions leading to already visited states.
            for action_idx, action in enumerate(applicable_actions):
                successor_state = action.apply(current_state)
                successor_states.append(successor_state)
                rewards.append(self.reward_function(current_state, action, successor_state, context.goal_condition))
                if successor_state in context.closed_set:
                    priorities[action_idx] = -1000000.0
            # Add the value of the current state if not already present.
            if current_state not in context.value_map:
                current_value = (successor_values + torch.tensor(rewards, dtype=torch.float, device=successor_values.device)).max().item()
                context.value_map[current_state] = current_value
            # Sample an action to apply.
            for priority, successor_value, action, reward, successor_state in zip(priorities, successor_values, applicable_actions, rewards, successor_states):
                if (successor_state not in unique_successors) and (len(successor_state.generate_applicable_actions()) > 0):
                    unique_successors.add(successor_state)
                    all_candidates.append((priority.item(), successor_value.item(), current_state, action, reward, successor_state))
        # Select the most promising successors.
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        best_candidates = all_candidates[:self.max_beam_size]
        # Update the beam context.
        for _, successor_value, current_state, action, reward, successor_state in best_candidates:
            if successor_state not in context.closed_set:
                context.transition_map[successor_state] = (current_state, action, reward, successor_value)
        context.open_list = [candidate[5] for candidate in best_candidates]
        context.solved = any(context.goal_condition.holds(state) for state in context.open_list)
        context.done = context.solved or (context.depth >= horizon) or (len(context.open_list) == 0)
        context.depth += 1

    def sample(self, initial_state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]], horizon: int) -> list[Trajectory]:
        """Generate trajectories for the given instances using the model."""
        contexts = [self.SearchContext(initial_state, goal_condition) for initial_state, goal_condition in initial_state_goals]
        with torch.no_grad():
            self.model.eval()
            for _ in range(horizon):
                state_goals_input: list[tuple[mm.State, mm.GroundConjunctiveCondition]] = []
                for context in contexts:
                    if not context.done:
                        state_goals_input.extend([(state, context.goal_condition) for state in context.open_list])
                if len(state_goals_input) == 0:
                    break  # All trajectories are done.
                # Evaluate all states in the beams.
                batch_successor_values = self.model.forward(state_goals_input)
                # Avoid visiting already visited states.
                offset = 0
                for context in contexts:
                    if not context.done:
                        beam_size = len(context.open_list)
                        beam_successor_values = batch_successor_values[offset:offset + beam_size]
                        self._beam_step(context, horizon, beam_successor_values)
                        offset += beam_size
        # Create trajectories from contexts.
        trajectories: list[Trajectory] = []
        for context in contexts:
            # Convert data to the trajectory format.
            state_sequence: list[mm.State] = []
            action_sequence: list[mm.GroundAction] = []
            reward_sequence: list[float] = []
            q_value_sequence: list[float] = []
            value_sequence: list[float] = []
            # Approximate the missing values of the final states by using the q-values.
            for state in context.open_list:
                if state not in context.value_map:
                    assert state in context.transition_map
                    _, _, _, successor_value = context.transition_map[state]
                    context.value_map[state] = successor_value
            # For each beam, produce a trajectory from the best state in the final open list.
            final_candidates = [(int(context.goal_condition.holds(state)), context.value_map[state], state) for state in context.open_list]
            final_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            _, value, state = final_candidates[0]
            state_sequence.append(state)
            value_sequence.append(value)
            while state in context.transition_map:
                predecessor_state, action, reward, q_value = context.transition_map[state]
                state_sequence.append(predecessor_state)
                action_sequence.append(action)
                reward_sequence.append(reward)
                q_value_sequence.append(q_value)
                value_sequence.append(value)
                state = predecessor_state
            state_sequence.reverse()
            action_sequence.reverse()
            reward_sequence.reverse()
            q_value_sequence.reverse()
            value_sequence.reverse()
            # Create trajectory.
            trajectories.append(
                Trajectory(
                    state_sequence,
                    action_sequence,
                    value_sequence,
                    q_value_sequence,
                    reward_sequence,
                    self.reward_function,
                    context.goal_condition,
                )
            )
        return trajectories
