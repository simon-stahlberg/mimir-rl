import torch

from abc import ABC, abstractmethod
from torch.nn.functional import huber_loss

from .models import QValueModel
from .trajectories import Transition


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, model: QValueModel, transitions: list[Transition]) -> torch.Tensor:
        pass


class DQNLossFunction(LossFunction):
    def __init__(self, discount_factor: float) -> None:
        assert isinstance(discount_factor, float), "Discount factor must be a float."
        assert discount_factor > 0.0, "Discount factor must be positive."
        assert discount_factor <= 1.0, "Discount factor must not be greater than 1."
        self.discount_factor = discount_factor

    def __call__(self, model: QValueModel, transitions: list[Transition]) -> torch.Tensor:
        dead_end_value = -10000
        targets = self._compute_targets(model, transitions, dead_end_value)
        state_goals = [(transition.current_state, transition.goal_condition) for transition in transitions]
        result = model.forward(state_goals)
        predictions = torch.stack([q_values[actions.index(transition.selected_action)] for (q_values, actions), transition in zip(result, transitions)])
        losses = huber_loss(predictions, targets, delta=1.0, reduction='none')
        return losses

    def _mellowmax(self, values: torch.Tensor, omega: float, dim: int = -1) -> torch.Tensor:
        """
        MellowMax operator (smooth maximum).

        Args:
            q_values (torch.Tensor): Tensor of values.
            omega (float): Temperature parameter (should be > 0).
            dim (int): Dimension to apply the operator over (default: last).

        Returns:
            torch.Tensor: MellowMax values.
        """
        assert isinstance(values, torch.Tensor), "Values must be an instance of Tensor."
        assert omega > 0, "Omega must be positive."
        return (torch.logsumexp(omega * values, dim=dim) - torch.log(torch.tensor(float(values.size(dim)), device=values.device))) / omega

    def _compute_targets(self, model: QValueModel, transitions: list[Transition], dead_end_value: float) -> torch.Tensor:
        with torch.no_grad():
            model.eval()
            device = next(model.parameters()).device
            state_goals = [(transition.current_state, transition.goal_condition) for transition in transitions]
            q_values_list = model.forward(state_goals)
            max_values = torch.stack([self._mellowmax(q_values, 10.0) if (q_values.numel() > 0) else torch.tensor(dead_end_value, device=device) for q_values, _ in q_values_list])
            rewards = torch.tensor([transition.reward for transition in transitions], requires_grad=False, device=device)
            achieves_goal = torch.tensor([transition.achieves_goal for transition in transitions], dtype=torch.float, requires_grad=False, device=device)
            return rewards + (1.0 - achieves_goal) * self.discount_factor * max_values