import torch

from torch.nn.functional import huber_loss

from .loss_functions import OptimizationFunction
from .models import ActionScalarModel
from .trajectories import Transition


class DQNOptimization(OptimizationFunction):

    def __init__(self,
                 model: ActionScalarModel,
                 model_optimizer: torch.optim.Optimizer,
                 model_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 target_model: ActionScalarModel,
                 discount_factor: float,
                 mellowmax_factor: float,
                 use_bounds_loss: bool = False) -> None:
        """
        Initializes the DQNOptimization class.

        Args:
            model (ActionScalarModel): The main model to be optimized.
            model_optimizer (torch.optim.Optimizer): The optimizer used for updating the model parameters.
            model_lr_scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler for the model optimizer.
            target_model (ActionScalarModel): The target model used for computing target values.
            discount_factor (float): The discount factor for future rewards. Must be in the range (0, 1].
            mellowmax_factor (float): The temperature parameter for the mellowmax function. Must be positive.
            use_bounds_loss (bool): If True, includes bounds loss in the optimization process. Defaults to False.

        Raises:
            AssertionError: If any of the arguments do not meet the specified type or value requirements.
        """
        assert isinstance(model, ActionScalarModel), "Model must be an instance of ActionScalarModel."
        assert isinstance(model_optimizer, torch.optim.Optimizer), "Model optimizer must be an instance of Optimizer."
        assert isinstance(model_lr_scheduler, torch.optim.lr_scheduler.LRScheduler), "Model learning rate scheduler must be an instance of LRScheduler."
        assert isinstance(target_model, ActionScalarModel), "Target model must be an instance of ActionScalarModel."
        assert isinstance(discount_factor, float), "Discount factor must be a float."
        assert isinstance(mellowmax_factor, float), "Mellowmax temperature must be a float."
        assert isinstance(use_bounds_loss, bool), "Option to use bounds loss must be a Boolean."
        assert discount_factor > 0.0, "Discount factor must be positive."
        assert discount_factor <= 1.0, "Discount factor must not be greater than 1."
        assert mellowmax_factor > 0.0, "Mellowmax temperature must be positive."
        self.model = model
        self.model_optimizer = model_optimizer
        self.model_lr_scheduler = model_lr_scheduler
        self.target_model = target_model
        self.discount_factor = discount_factor
        self.mellowmax_temperature = mellowmax_factor
        self.use_bounds_loss = use_bounds_loss

    def get_mellowmax_factor(self) -> float:
        """
        Get the mellowmax temperature factor.

        Returns:
            float: The current mellowmax temperature factor.
        """
        return self.mellowmax_temperature

    def set_mellowmax_factor(self, factor: float) -> None:
        """
        Set the mellowmax temperature factor.

        Args:
            factor (float): The new mellowmax temperature factor. Must be positive.
        """
        assert factor > 0.0, "Mellowmax temperature must be positive."
        self.mellowmax_temperature = factor

    def __call__(self, transitions: list[Transition], weights: torch.Tensor) -> torch.Tensor:
        """
        Execute the DQN optimization function.

        Args:
           transitions (list[Transition]): A list of transitions.
           weights (torch.Tensor): A tensor of weights corresponding to each transition.

        Returns:
           torch.Tensor: The loss tensor resulting from the optimization process, used for monitoring as no gradients are attached to it.
        """
        dead_end_value = -10000
        device = next(self.model.parameters()).device
        target_q_values = self._compute_targets(transitions, dead_end_value, device)
        state_goals = [(transition.current_state, transition.goal_condition) for transition in transitions]
        # Compute bounds, if selected.
        if self.use_bounds_loss:
            lower_bounds, upper_bounds = self.get_value_bounds(transitions, device)
        # Run the forward pass for each model, they share the target Q-values and bounds are only computed once.
        self.model.train()
        batch_q_values = self.model.forward(state_goals)
        selected_q_values = torch.stack([q_values[actions.index(transition.selected_action)] for (q_values, actions), transition in zip(batch_q_values, transitions)])
        # Compute DQN loss.
        losses = huber_loss(selected_q_values, target_q_values, delta=1.0, reduction='none')
        # Compute bounds loss, if selected.
        if self.use_bounds_loss:
            bounds_errors = selected_q_values - selected_q_values.clamp(lower_bounds, upper_bounds).detach()  # type: ignore
            bounds_losses = huber_loss(bounds_errors, torch.zeros_like(bounds_errors), delta=1.0, reduction='none')
            losses += bounds_losses
        losses *= weights.to(losses.device)
        self.model_optimizer.zero_grad()
        losses.mean().backward()
        self.model_optimizer.step()
        self.model_lr_scheduler.step()
        return losses.detach()

    def _mellowmax(self, values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        MellowMax operator (smooth maximum).

        Args:
            values (torch.Tensor): Tensor of values.
            dim (int): Dimension to apply the operator over (default: last).

        Returns:
            torch.Tensor: MellowMax values.
        """
        assert isinstance(values, torch.Tensor), "Values must be an instance of Tensor."
        return (torch.logsumexp(self.mellowmax_temperature * values, dim=dim) - torch.log(torch.tensor(float(values.size(dim)), device=values.device))) / self.mellowmax_temperature

    def _compute_targets(self, transitions: list[Transition], dead_end_value: float, device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            successor_state_goals = [(transition.successor_state, transition.goal_condition) for transition in transitions]
            batch_successor_q_values = self.target_model.forward(successor_state_goals)
            successor_max_values = torch.stack([self._mellowmax(q_values) if (q_values.numel() > 0) else torch.tensor(dead_end_value, dtype=torch.float, device=device) for q_values, _ in batch_successor_q_values])
            immediate_rewards = torch.tensor([transition.immediate_reward for transition in transitions], requires_grad=False, dtype=torch.float, device=device)
            achieves_goal = torch.tensor([transition.achieves_goal for transition in transitions], dtype=torch.float, requires_grad=False, device=device)
            return immediate_rewards + (1.0 - achieves_goal) * self.discount_factor * successor_max_values
