import pymimir as mm
import torch

from abc import abstractmethod
from collections import defaultdict
from typing import Any

from .models import ActionScalarModel
from .reward_functions import RewardFunction
from .trajectories import Trajectory
from .trajectory_sampling import TrajectorySampler, TrajectoryState


class PolicyRolloutSampler(TrajectorySampler):
    """
    Abstract base class for sampling trajectories.
    """
    def __init__(self,
                 model: ActionScalarModel,
                 reward_function: RewardFunction) -> None:
        assert isinstance(model, ActionScalarModel), "Model must be an instance of ActionScalarModel."
        assert isinstance(reward_function, RewardFunction), "Reward function must be an instance of RewardFunction."
        super().__init__()
        self.model = model
        self.reward_function = reward_function

    class RolloutContext:
        def __init__(self, initial_state: mm.State) -> None:
            self.closed_set: set[mm.State] = set([initial_state])

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

    def _initialize(self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]) -> tuple[list[TrajectoryState], list[RolloutContext]]:
        trajectory_states: list[TrajectoryState] = [TrajectoryState(initial_state, goal_condition) for initial_state, goal_condition in state_goals]
        for trajectory_state in trajectory_states:
            trajectory_state.state_sequence.append(trajectory_state.start_state)
        rollout_states = [self.RolloutContext(initial_state) for initial_state, _ in state_goals]
        return trajectory_states, rollout_states

    def _internal_sample(self, trajectory_states: list[TrajectoryState], internal_states: list[RolloutContext], max_steps: list[int]) -> None:
        with torch.no_grad():
            self.model.eval()
            state_goals = [(context.state_sequence[-1], context.goal_condition) for context in trajectory_states if not context.done]
            context_indices = [idx for idx, context in enumerate(trajectory_states) if not context.done]
            if len(state_goals) > 0:
                q_values_batch = self.model.forward(state_goals)
                for rollout_idx, (q_values, applicable_actions) in zip(context_indices, q_values_batch):
                    trajectory_state = trajectory_states[rollout_idx]
                    internal_state = internal_states[rollout_idx]
                    rewards: list[float] = []
                    q_values = q_values.cpu()  # Move the result to CPU, the remaining operations are very cheap.
                    q_values_copy = q_values.clone()
                    current_state = trajectory_state.state_sequence[-1]
                    # Reduce the value actions leading to already visited states.
                    for action_idx, action in enumerate(applicable_actions):
                        successor_state = action.apply(current_state)
                        rewards.append(self.reward_function(current_state, action, successor_state, trajectory_state.goal_condition))
                        if successor_state in internal_state.closed_set:
                            q_values_copy[action_idx] = -1000000.0
                    # Sample an action to apply.
                    action_idx = self.sample_action_index(current_state, applicable_actions, q_values_copy)
                    action = applicable_actions[action_idx]
                    successor_state = action.apply(current_state)
                    q_value = q_values[action_idx].item()
                    reward = rewards[action_idx]
                    value = (q_values + torch.tensor(rewards, dtype=torch.float, device=q_values.device)).max().item()
                    is_solved = trajectory_state.goal_condition.holds(successor_state)
                    is_dead_end = len(successor_state.generate_applicable_actions()) == 0
                    exceeds_horizon = len(trajectory_state.state_sequence) >= max_steps[rollout_idx]
                    is_done = is_solved or is_dead_end or exceeds_horizon
                    # Update states.
                    trajectory_state.state_sequence.append(successor_state)
                    trajectory_state.action_sequence.append(action)
                    trajectory_state.value_sequence.append(value)
                    trajectory_state.q_value_sequence.append(q_value)
                    trajectory_state.reward_sequence.append(reward)
                    trajectory_state.solved = is_solved
                    trajectory_state.done = is_done
                    internal_state.closed_set.add(successor_state)

    def _finalize(self, trajectory_states: list[TrajectoryState], internal_states: list[Any]) -> None:
        pass  # No finalization needed for rollout sampler.

    def sample(self, initial_state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]], horizon: int) -> list[Trajectory]:
        """Generate trajectories for the given instances using the model."""
        trajectory_states, rollout_states = self._initialize(initial_state_goals)
        while any(not trajectory_state.done for trajectory_state in trajectory_states):
            max_steps = [horizon for _ in trajectory_states]
            self._internal_sample(trajectory_states, rollout_states, max_steps)
        self._finalize(trajectory_states, rollout_states)
        return self._to_trajectories(trajectory_states, self.reward_function)


class PolicyTrajectorySampler(PolicyRolloutSampler):
    """
    Values are treated as logits, and an action is sampled according to the resulting probability distribution.
    """
    def __init__(self,
                 model: ActionScalarModel,
                 reward_function: RewardFunction) -> None:
        super().__init__(model, reward_function)

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
                 epsilon: float) -> None:
        assert isinstance(epsilon, float), "Epsilon must be a float."
        assert epsilon >= 0.0, "Epsilon must be a probability."
        assert epsilon <= 1.0, "Epsilon must be a probability."
        super().__init__(model, reward_function)
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
                 temperature: float) -> None:
        assert isinstance(temperature, float), "Temperature must be a float."
        assert temperature > 0.0, "Temperature must be positive."
        super().__init__(model, reward_function)
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
                 temperature_steps: int) -> None:
        super().__init__(model, reward_function)
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
                 reward_function: RewardFunction) -> None:
        super().__init__(model, reward_function)

    def sample_action_index(self, state: mm.State, actions: list[mm.GroundAction], values: torch.Tensor) -> int:
        action_index = values.argmax().item()
        return action_index  # type: ignore
