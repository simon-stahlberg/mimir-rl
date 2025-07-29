import pymimir as mm
import torch

from abc import ABC, abstractmethod
from torch.nn.functional import huber_loss

from .models import QValueModel
from .trajectories import Transition


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, q_values: torch.Tensor, transitions: list[Transition]) -> torch.Tensor:
        pass


class DQNLossFunction(LossFunction):
    def __init__(self, target_model: QValueModel, discount_factor: float, mellowmax_factor: float) -> None:
        assert isinstance(target_model, QValueModel), "Target model must be an instance of QValueModel."
        assert isinstance(discount_factor, float), "Discount factor must be a float."
        assert discount_factor > 0.0, "Discount factor must be positive."
        assert discount_factor <= 1.0, "Discount factor must not be greater than 1."
        self.target_model = target_model
        self.discount_factor = discount_factor
        self.mellowmax_factor = mellowmax_factor

    def get_mellowmax_factor(self) -> float:
        return self.mellowmax_factor

    def set_mellowmax_factor(self, factor: float) -> None:
        self.mellowmax_factor = factor

    def __call__(self, q_values: torch.Tensor, transitions: list[Transition]) -> torch.Tensor:
        dead_end_value = -10000
        target_q_values = self._compute_targets(transitions, dead_end_value, q_values.device)
        losses = huber_loss(q_values, target_q_values, delta=1.0, reduction='none')
        return losses

    def _mellowmax(self, values: torch.Tensor, omega: float, dim: int = -1) -> torch.Tensor:
        """
        MellowMax operator (smooth maximum).

        Args:
            values (torch.Tensor): Tensor of values.
            omega (float): Temperature parameter (should be > 0).
            dim (int): Dimension to apply the operator over (default: last).

        Returns:
            torch.Tensor: MellowMax values.
        """
        assert isinstance(values, torch.Tensor), "Values must be an instance of Tensor."
        assert omega > 0, "Omega must be positive."
        return (torch.logsumexp(omega * values, dim=dim) - torch.log(torch.tensor(float(values.size(dim)), device=values.device))) / omega

    def _compute_targets(self, transitions: list[Transition], dead_end_value: float, device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            successor_state_goals = [(transition.successor_state, transition.goal_condition) for transition in transitions]
            all_quccessor_q_values = self.target_model.forward(successor_state_goals)
            successor_max_values = torch.stack([self._mellowmax(q_values, self.mellowmax_factor) if (q_values.numel() > 0) else torch.tensor(dead_end_value, dtype=torch.float, device=device) for q_values, _ in all_quccessor_q_values])
            rewards = torch.tensor([transition.reward for transition in transitions], requires_grad=False, dtype=torch.float, device=device)
            achieves_goal = torch.tensor([transition.achieves_goal for transition in transitions], dtype=torch.float, requires_grad=False, device=device)
            return rewards + (1.0 - achieves_goal) * self.discount_factor * successor_max_values
