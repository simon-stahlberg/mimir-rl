import torch

from abc import ABC, abstractmethod
from torch.nn.functional import huber_loss

from .models import QValueModel
from .trajectories import Transition


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, transitions: list[Transition]) -> torch.Tensor:
        pass


class DQNLossFunction(LossFunction):
    def __init__(self, model: QValueModel, target_model: QValueModel, discount_factor: float) -> None:
        assert isinstance(model, QValueModel), "Model must be an instance of ModelWrapper."
        assert isinstance(target_model, QValueModel), "Target model must be an instance of ModelWrapper."
        assert isinstance(discount_factor, float), "Discount factor must be a float."
        assert discount_factor > 0.0, "Discount factor must be positive."
        assert discount_factor <= 1.0, "Discount factor must not be greater than 1."
        self.model = model
        self.target_model = target_model
        self.discount_factor = discount_factor

    def __call__(self, transitions: list[Transition]) -> torch.Tensor:
        dead_end_value = -10000
        targets = self._compute_targets(transitions, dead_end_value)
        predictions = self.model.forward_transitions(transitions)
        losses = huber_loss(predictions, targets, delta=1.0, reduction='none')
        return losses

    def _compute_targets(self, transitions: list[Transition], dead_end_value: float) -> torch.Tensor:
        # TODO: Replace target network with smooth maximum.
        with torch.no_grad():
            self.target_model.eval()
            device = next(self.target_model.parameters()).device
            state_goals = [(transition.current_state, transition.goal_condition) for transition in transitions]
            q_values_list = self.target_model.forward_state_goals(state_goals)
            max_values = torch.stack([q_values.max() if (q_values.numel() > 0) else torch.tensor(dead_end_value, device=device) for q_values, _ in q_values_list])
            rewards = torch.tensor([transition.reward for transition in transitions], requires_grad=False, device=device)
            achieves_goal = torch.tensor([transition.achieves_goal for transition in transitions], dtype=torch.float, requires_grad=False, device=device)
            return rewards + (1.0 - achieves_goal) * self.discount_factor * max_values