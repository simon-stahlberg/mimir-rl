import torch

from abc import ABC, abstractmethod
from torch.nn.functional import huber_loss

from .models import ActionScalarModel
from .trajectories import Transition


class LossFunction(ABC):
    def get_value_bounds(self, transitions: list[Transition], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        for transition in transitions:
            reward_function = transition.reward_function
            lower_bound, upper_bound = reward_function.get_value_bounds(transition.immediate_reward, transition.future_rewards)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        lower_bounds_tensor = torch.tensor(lower_bounds, requires_grad=False, dtype=torch.float, device=device)
        upper_bounds_tensor = torch.tensor(upper_bounds, requires_grad=False, dtype=torch.float, device=device)
        return lower_bounds_tensor, upper_bounds_tensor

    @abstractmethod
    def __call__(self, transitions: list[Transition]) -> torch.Tensor:
        pass


class DQNLossFunction(LossFunction):
    def __init__(self, model: ActionScalarModel, target_model: ActionScalarModel, discount_factor: float, mellowmax_factor: float, use_bounds_loss: bool = False) -> None:
        assert isinstance(model, ActionScalarModel), "Model must be an instance of ActionScalarModel."
        assert isinstance(target_model, ActionScalarModel), "Target model must be an instance of ActionScalarModel."
        assert isinstance(discount_factor, float), "Discount factor must be a float."
        assert isinstance(mellowmax_factor, float), "Mellowmax factor must be a float."
        assert isinstance(use_bounds_loss, bool), "Option to use bounds loss must be a Boolean."
        assert discount_factor > 0.0, "Discount factor must be positive."
        assert discount_factor <= 1.0, "Discount factor must not be greater than 1."
        self.model = model
        self.target_model = target_model
        self.discount_factor = discount_factor
        self.mellowmax_factor = mellowmax_factor
        self.use_bounds_loss = use_bounds_loss

    def get_mellowmax_factor(self) -> float:
        return self.mellowmax_factor

    def set_mellowmax_factor(self, factor: float) -> None:
        self.mellowmax_factor = factor

    def __call__(self, transitions: list[Transition]) -> torch.Tensor:
        self.model.train()
        state_goals = [(transition.current_state, transition.goal_condition) for transition in transitions]
        all_q_values = self.model.forward(state_goals)
        selected_q_values = torch.stack([q_values[actions.index(transition.selected_action)] for (q_values, actions), transition in zip(all_q_values, transitions)])
        # Compute DQN loss.
        dead_end_value = -10000
        target_q_values = self._compute_targets(transitions, dead_end_value, selected_q_values.device)
        losses = huber_loss(selected_q_values, target_q_values, delta=1.0, reduction='none')
        # Compute bounds loss, if selected.
        if self.use_bounds_loss:
            lower_bounds, upper_bounds = self.get_value_bounds(transitions, selected_q_values.device)
            bounds_errors = selected_q_values - selected_q_values.clamp(lower_bounds, upper_bounds).detach()
            losses += huber_loss(bounds_errors, torch.zeros_like(bounds_errors), delta=1.0, reduction='none')
        return losses

    def _mellowmax(self, values: torch.Tensor, temperature: float, dim: int = -1) -> torch.Tensor:
        """
        MellowMax operator (smooth maximum).

        Args:
            values (torch.Tensor): Tensor of values.
            temperature (float): Temperature parameter (should be > 0).
            dim (int): Dimension to apply the operator over (default: last).

        Returns:
            torch.Tensor: MellowMax values.
        """
        assert isinstance(values, torch.Tensor), "Values must be an instance of Tensor."
        assert temperature > 0, "Omega must be positive."
        return (torch.logsumexp(temperature * values, dim=dim) - torch.log(torch.tensor(float(values.size(dim)), device=values.device))) / temperature

    def _compute_targets(self, transitions: list[Transition], dead_end_value: float, device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            successor_state_goals = [(transition.successor_state, transition.goal_condition) for transition in transitions]
            all_quccessor_q_values = self.target_model.forward(successor_state_goals)
            successor_max_values = torch.stack([self._mellowmax(q_values, self.mellowmax_factor) if (q_values.numel() > 0) else torch.tensor(dead_end_value, dtype=torch.float, device=device) for q_values, _ in all_quccessor_q_values])
            immediate_rewards = torch.tensor([transition.immediate_reward for transition in transitions], requires_grad=False, dtype=torch.float, device=device)
            achieves_goal = torch.tensor([transition.achieves_goal for transition in transitions], dtype=torch.float, requires_grad=False, device=device)
            return immediate_rewards + (1.0 - achieves_goal) * self.discount_factor * successor_max_values
