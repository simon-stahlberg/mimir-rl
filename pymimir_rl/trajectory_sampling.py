import pymimir as mm
import torch

from abc import ABC, abstractmethod

from .models import ActionScalarModel
from .reward_functions import RewardFunction
from .trajectories import Trajectory


class TrajectorySampler(ABC):
    """
    Abstract base class for sampling trajectories.
    """
    def __init__(self, model: ActionScalarModel, reward_function: RewardFunction) -> None:
        assert isinstance(model, ActionScalarModel), "Model must be an instance of ActionScalarModel."
        assert isinstance(reward_function, RewardFunction), "Reward function must be an instance of RewardFunction."
        super().__init__()
        self.model = model
        self.reward_function = reward_function

    class RolloutContext:
        def __init__(self, initial_state: mm.State, goal_condition: mm.GroundConjunctiveCondition) -> None:
            self.state_sequence: list[mm.State] = [initial_state]
            self.action_sequence: list[mm.GroundAction] = []
            self.reward_sequence: list[float] = []
            self.closed_set: set[mm.State] = set([initial_state])
            self.goal_condition: mm.GroundConjunctiveCondition = goal_condition
            self.solved: bool = goal_condition.holds(initial_state)
            self.done = self.solved or (len(initial_state.generate_applicable_actions()) == 0)

    @abstractmethod
    def sample_action_index(self, values: torch.Tensor) -> int:
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
                    break  # All
                q_values_batch = self.model.forward(state_goals)
                for rollout_idx, (q_values, applicable_actions) in zip(context_indices, q_values_batch):
                    context = rollout_contexts[rollout_idx]
                    q_values = q_values.cpu()  # Move the result to CPU, the remaining operations are very cheap.
                    q_values_copy = q_values.clone()
                    current_state = context.state_sequence[-1]
                    # Reduce the value actions leading to already visited states.
                    for action_idx, action in enumerate(applicable_actions):
                        successor_state = action.apply(current_state)
                        if successor_state in context.closed_set:
                            q_values_copy[action_idx] = -1000000.0
                    # Sample an action to apply.
                    action_idx = self.sample_action_index(q_values_copy)
                    action = applicable_actions[action_idx]
                    successor_state = action.apply(current_state)
                    reward = self.reward_function(current_state, action, successor_state, context.goal_condition)
                    solves = context.goal_condition.holds(successor_state)
                    # Update context.
                    context.state_sequence.append(successor_state)
                    context.action_sequence.append(action)
                    context.reward_sequence.append(reward)
                    context.closed_set.add(successor_state)
                    context.solved = solves
                    context.done = solves or (len(successor_state.generate_applicable_actions()) == 0)
        # Create trajectories from contexts.
        trajectories: list[Trajectory] = []
        for context in rollout_contexts:
            trajectories.append(Trajectory(context.state_sequence, context.action_sequence, context.reward_sequence, self.reward_function, context.goal_condition))
        return trajectories


class PolicyTrajectorySampler(TrajectorySampler):
    """
    Values are treated as logits, and an action is sampled according to the resulting probability distribution.
    """

    def __init__(self, model: ActionScalarModel, reward_function: RewardFunction) -> None:
        super().__init__(model, reward_function)

    def sample_action_index(self, values: torch.Tensor) -> int:
        probabilities = values.softmax(0)
        action_index = probabilities.multinomial(num_samples=1)
        return action_index.item()  # type: ignore


class BoltzmannTrajectorySampler(TrajectorySampler):
    """
    Samples trajectories based on a Boltzmann distribution.
    """

    def __init__(self, model: ActionScalarModel, reward_function: RewardFunction, temperature: float) -> None:
        assert isinstance(temperature, float), "Temperature must be a float."
        assert temperature > 0.0, "Temperature must be positive."
        super().__init__(model, reward_function)
        self.temperature = temperature

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    def get_temperature(self) -> float:
        return self.temperature

    def sample_action_index(self, values: torch.Tensor) -> int:
        probabilities = (values / self.temperature).softmax(dim=0)
        action_index = probabilities.multinomial(num_samples=1)
        return action_index.item()  # type: ignore


class GreedyPolicyTrajectorySampler(TrajectorySampler):
    """
    Samples trajectories based on greedily and deterministically following the policy.
    """

    def __init__(self, model: ActionScalarModel, reward_function: RewardFunction) -> None:
        super().__init__(model, reward_function)

    def sample_action_index(self, values: torch.Tensor) -> int:
        action_index = values.argmax().item()
        return action_index  # type: ignore
