import math
import pymimir as mm
import torch

from abc import ABC, abstractmethod
from torch.nn.functional import huber_loss

from .models import ActionScalarModel
from .trajectories import Transition


"""
Abstract base class for optimization functions used in reinforcement learning.
"""
class OptimizationFunction(ABC):
    def get_value_bounds(self, transitions: list[Transition], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute lower and upper bounds for values in the given transitions.

        Args:
            transitions (list[Transition]): List of Transition objects containing reward information.
            device (torch.device): PyTorch device to place the returned tensors on.

        Returns:
            Tuple of (lower_bounds_tensor, upper_bounds_tensor) containing the computed bounds for each transition.
        """
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        for transition in transitions:
            reward_function = transition.reward_function
            lower_bound, upper_bound = reward_function.get_value_bounds(transition.immediate_reward, transition.future_rewards, transition.part_of_solution)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        lower_bounds_tensor = torch.tensor(lower_bounds, requires_grad=False, dtype=torch.float, device=device)
        upper_bounds_tensor = torch.tensor(upper_bounds, requires_grad=False, dtype=torch.float, device=device)
        return lower_bounds_tensor, upper_bounds_tensor

    @abstractmethod
    def __call__(self, transitions: list[Transition], weights: torch.Tensor) -> torch.Tensor:
        """
        Optimize the model parameters based on the given transitions and weights.

        Args:
            transitions (list[Transition]): List of Transition objects.
            weights (torch.Tensor): Tensor of weights for the transitions.

        Returns:
            Tensor containing the computed optimization errors. Gradient is not attached.
        """
        pass


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
        all_q_values = self.model.forward(state_goals)
        selected_q_values = torch.stack([q_values[actions.index(transition.selected_action)] for (q_values, actions), transition in zip(all_q_values, transitions)])
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
            all_successor_q_values = self.target_model.forward(successor_state_goals)
            successor_max_values = torch.stack([self._mellowmax(q_values) if (q_values.numel() > 0) else torch.tensor(dead_end_value, dtype=torch.float, device=device) for q_values, _ in all_successor_q_values])
            immediate_rewards = torch.tensor([transition.immediate_reward for transition in transitions], requires_grad=False, dtype=torch.float, device=device)
            achieves_goal = torch.tensor([transition.achieves_goal for transition in transitions], dtype=torch.float, requires_grad=False, device=device)
            return immediate_rewards + (1.0 - achieves_goal) * self.discount_factor * successor_max_values


class DiscreteSoftActorCriticOptimization(OptimizationFunction):
    def __init__(self,
                 policy_model: ActionScalarModel,
                 policy_optimizer: torch.optim.Optimizer,
                 policy_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 qvalue_target_1: ActionScalarModel,
                 qvalue_model_1: ActionScalarModel,
                 qvalue_optimizer_1: torch.optim.Optimizer,
                 qvalue_lr_scheduler_1: torch.optim.lr_scheduler.LRScheduler,
                 qvalue_target_2: ActionScalarModel,
                 qvalue_model_2: ActionScalarModel,
                 qvalue_optimizer_2: torch.optim.Optimizer,
                 qvalue_lr_scheduler_2: torch.optim.lr_scheduler.LRScheduler,
                 discount_factor: float,
                 polyak_factor: float = 0.005,
                 entropy_temperature: float = 1.0,
                 entropy_lr: float = 0.0003) -> None:
        """
        Initializes the DiscreteSoftActorCriticOptimization class.

        Args:
            policy_model (ActionScalarModel): The policy model to be optimized.
            policy_optimizer (torch.optim.Optimizer): The optimizer for the policy model.
            policy_lr_scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler for the policy optimizer.
            qvalue_target_1 (ActionScalarModel): The first Q-value model to compute targets.
            qvalue_model_1 (ActionScalarModel): The first Q-value model to be optimized.
            qvalue_optimizer_1 (torch.optim.Optimizer): The optimizer for the first Q-value model.
            qvalue_lr_scheduler_1 (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler for the first Q-value optimizer.
            qvalue_target_2 (ActionScalarModel): The second Q-value model to compute targets.
            qvalue_model_2 (ActionScalarModel): The second Q-value model to be optimized.
            qvalue_optimizer_2 (torch.optim.Optimizer): The optimizer for the second Q-value model.
            qvalue_lr_scheduler_2 (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler for the second Q-value optimizer.
            discount_factor (float): The discount factor for future rewards. Must be in the range (0, 1].
            polyak_factor (float): The rate at which the target models are updated. Must be in the range (0, 1]. Defaults to 0.005.
            entropy_temperature (float): The initial entropy value. Must be in the range (0, 1]. Defaults to 1.0.
            entropy_lr (float): The learning rate for the entropy temperature. Must be positive. Defaults to 0.0003.

        Raises:
            AssertionError: If any of the arguments do not meet the specified type or value requirements.
        """
        super().__init__()
        assert isinstance(policy_model, ActionScalarModel), "Policy model must be an instance of ActionScalarModel."
        assert isinstance(policy_optimizer, torch.optim.Optimizer), "Policy optimizer must be an instance of torch.optim.Optimizer."
        assert isinstance(policy_lr_scheduler, torch.optim.lr_scheduler.LRScheduler), "Policy LR scheduler must be an instance of torch.optim.lr_scheduler.LRScheduler."
        assert isinstance(qvalue_target_1, ActionScalarModel), "First Q-value model must be an instance of ActionScalarModel."
        assert isinstance(qvalue_model_1, ActionScalarModel), "First Q-value model must be an instance of ActionScalarModel."
        assert isinstance(qvalue_optimizer_1, torch.optim.Optimizer), "First Q-value optimizer must be an instance of torch.optim.Optimizer."
        assert isinstance(qvalue_lr_scheduler_1, torch.optim.lr_scheduler.LRScheduler), "First Q-value LR scheduler must be an instance of torch.optim.lr_scheduler.LRScheduler."
        assert isinstance(qvalue_target_2, ActionScalarModel), "Second Q-value model must be an instance of ActionScalarModel."
        assert isinstance(qvalue_model_2, ActionScalarModel), "Second Q-value model must be an instance of ActionScalarModel."
        assert isinstance(qvalue_optimizer_2, torch.optim.Optimizer), "Second Q-value optimizer must be an instance of torch.optim.Optimizer."
        assert isinstance(qvalue_lr_scheduler_2, torch.optim.lr_scheduler.LRScheduler), "Second Q-value LR scheduler must be an instance of torch.optim.lr_scheduler.LRScheduler."
        assert isinstance(discount_factor, float), "Discount factor must be a float."
        assert isinstance(polyak_factor, float), "Polyak factor must be a float."
        assert isinstance(entropy_temperature, float), "The initial entropy must be a float."
        assert isinstance(entropy_lr, float), "Learning rate for the entropy must be a float."
        assert discount_factor > 0.0, "Discount factor must be greater than 0.0."
        assert discount_factor <= 1.0, "Discount factor must be less than or equal to 1.0."
        assert polyak_factor > 0.0, "Polyak factor must be greater than 0.0."
        assert polyak_factor <= 1.0, "Polyak factor must be less than or equal to 1.0."
        assert entropy_temperature > 0.0, "Entropy factor must be greater than 0.0."
        assert entropy_lr > 0.0, "Learning rate for policy must be greater than 0.0."
        self.policy_model = policy_model
        self.policy_optimizer = policy_optimizer
        self.policy_lr_scheduler = policy_lr_scheduler
        self.qvalue_target_1 = qvalue_target_1
        self.qvalue_model_1 = qvalue_model_1
        self.qvalue_optimizer_1 = qvalue_optimizer_1
        self.qvalue_lr_scheduler_1 = qvalue_lr_scheduler_1
        self.qvalue_target_2 = qvalue_target_2
        self.qvalue_model_2 = qvalue_model_2
        self.qvalue_optimizer_2 = qvalue_optimizer_2
        self.qvalue_lr_scheduler_2 = qvalue_lr_scheduler_2
        self.discount_factor = discount_factor
        self.polyak_factor = polyak_factor
        device = next(self.policy_model.parameters()).device
        self.entropy_temperature = entropy_temperature
        self.log_entropy_fraction = torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        self.entropy_optimizer = torch.optim.Adam([self.log_entropy_fraction], lr=entropy_lr)
        self._update_target_models(1.0)  # Ensure that target models are initialized correctly.

    def get_entropy_temperature(self) -> float:
        return self.entropy_temperature

    def set_entropy_temperature(self, entropy_temperature: float):
        assert entropy_temperature > 0.0, "Entropy fraction must be greater than 0.0."
        self.entropy_temperature = entropy_temperature

    def _compute_qvalue_targets(self, transitions: list[Transition]) -> torch.Tensor:
        with torch.no_grad():
            device = next(self.policy_model.parameters()).device
            dead_end_value = -10000
            successor_state_goals = [(transition.successor_state, transition.goal_condition) for transition in transitions]
            all_policy_logits = self.policy_model.forward(successor_state_goals)
            all_qvalues_1 = self.qvalue_target_1.forward(successor_state_goals)
            all_qvalues_2 = self.qvalue_target_2.forward(successor_state_goals)
            target_qvalues: list[torch.Tensor] = []
            for (policy_logits, _), (qvalues_1, _), (qvalues_2, _) in zip(all_policy_logits, all_qvalues_1, all_qvalues_2):
                if policy_logits.numel() > 0:
                    entropy_factor = self.log_entropy_fraction.exp().item() * math.log(policy_logits.numel())  # TODO: -math.log(...)?
                    policy_entropy = entropy_factor * policy_logits.log_softmax(0)
                    min_qvalues = torch.min(qvalues_1, qvalues_2)
                    target_qvalue = (policy_logits.softmax(0) * (min_qvalues - policy_entropy)).sum()
                    target_qvalues.append(target_qvalue)
                else:  # If there are no successor states, it's a dead end.
                    target_qvalues.append(torch.tensor(dead_end_value, device=device))
            immediate_rewards = torch.tensor([transition.immediate_reward for transition in transitions], requires_grad=False, dtype=torch.float, device=device)
            achieves_goal = torch.tensor([transition.achieves_goal for transition in transitions], dtype=torch.float, requires_grad=False, device=device)
            discounted_targets = immediate_rewards + (1.0 - achieves_goal) * self.discount_factor * torch.stack(target_qvalues)
            return discounted_targets

    def _update_qvalue_models(self,
                              transitions: list[Transition],
                              all_qvalues_1: list[tuple[torch.Tensor, list[mm.GroundAction]]],
                              all_qvalues_2: list[tuple[torch.Tensor, list[mm.GroundAction]]],
                              qvalue_targets: torch.Tensor,
                              weights: torch.Tensor) -> torch.Tensor:
        selected_qvalues_1 = torch.stack([qvalues[actions.index(transition.selected_action)] for (qvalues, actions), transition in zip(all_qvalues_1, transitions)])
        selected_qvalues_2 = torch.stack([qvalues[actions.index(transition.selected_action)] for (qvalues, actions), transition in zip(all_qvalues_2, transitions)])
        qvalue_losses_1 = weights.to(selected_qvalues_1.device) * torch.nn.functional.huber_loss(selected_qvalues_1, qvalue_targets, reduction='none')
        qvalue_losses_2 = weights.to(selected_qvalues_2.device) * torch.nn.functional.huber_loss(selected_qvalues_2, qvalue_targets, reduction='none')
        # Optimize first Q-value model.
        self.qvalue_optimizer_1.zero_grad()
        (qvalue_losses_1).mean().backward()
        self.qvalue_optimizer_1.step()
        self.qvalue_lr_scheduler_1.step()
        # Optimize second Q-value model.
        self.qvalue_optimizer_2.zero_grad()
        (qvalue_losses_2).mean().backward()
        self.qvalue_optimizer_2.step()
        self.qvalue_lr_scheduler_2.step()
        # Return detached losses for monitoring.
        return (qvalue_losses_1 + qvalue_losses_2).detach()

    def _update_policy_model(self,
                             batch_qvalues_1: list[tuple[torch.Tensor, list[mm.GroundAction]]],
                             batch_qvalues_2: list[tuple[torch.Tensor, list[mm.GroundAction]]],
                             batch_policy_logits: list[tuple[torch.Tensor, list[mm.GroundAction]]],
                             weights: torch.Tensor) -> torch.Tensor:
        batch_policy_probs = [logits.softmax(0) for logits, _ in batch_policy_logits]
        batch_min_qvalues = [torch.min(qvalues_1, qvalues_2) for (qvalues_1, _), (qvalues_2, _) in zip(batch_qvalues_1, batch_qvalues_2)]
        batch_loss: list[torch.Tensor] = []
        for policy_probs, min_qvalues in zip(batch_policy_probs, batch_min_qvalues):
            entropy_factor = self.log_entropy_fraction.exp().item() * math.log(policy_probs.numel())
            entropy_term = entropy_factor * (policy_probs + 1E-8).log()
            policy_loss = (policy_probs * (entropy_term - min_qvalues.detach())).sum()
            batch_loss.append(policy_loss)
        batch_loss_tensor = torch.stack(batch_loss)
        losses = weights.to(batch_loss_tensor.device) * batch_loss_tensor
        self.policy_optimizer.zero_grad()
        losses.mean().backward()
        self.policy_optimizer.step()
        self.policy_lr_scheduler.step()
        return losses.detach()

    def _update_entropy_factor(self, batch_policy_logits: list[tuple[torch.Tensor, list[mm.GroundAction]]], weights: torch.Tensor):
        batch_loss: list[torch.Tensor] = []
        for logits, _ in batch_policy_logits:
            logits = logits.detach()
            entropy = -(logits.softmax(0) * logits.log_softmax(0)).sum(0)
            entropy_target = self.entropy_temperature * math.log(logits.numel())
            entropy_factor = self.log_entropy_fraction.exp() * math.log(logits.numel())
            entropy_loss = -(entropy_factor * (entropy.detach() - entropy_target))
            batch_loss.append(entropy_loss)
        batch_loss_tensor = torch.stack(batch_loss)
        losses = weights.to(batch_loss_tensor.device) * batch_loss_tensor
        self.entropy_optimizer.zero_grad()
        losses.mean().backward()
        self.entropy_optimizer.step()
        return losses.detach()

    def _update_target_models(self, polyak_factor: float):
        # Update target networks using polyak averaging
        with torch.no_grad():
            for target_param, param in zip(self.qvalue_target_1.parameters(), self.qvalue_model_1.parameters()):
                target_param.copy_(polyak_factor * param + (1.0 - polyak_factor) * target_param)
            for target_param, param in zip(self.qvalue_target_2.parameters(), self.qvalue_model_2.parameters()):
                target_param.copy_(polyak_factor * param + (1.0 - polyak_factor) * target_param)

    def __call__(self, transitions: list[Transition], weights: torch.Tensor) -> torch.Tensor:
        """
        Execute the SAC optimization step.

        Args:
            transitions (list[Transition]): List of transitions.
            weights (torch.Tensor): Weights for the transitions.

        Returns:
            torch.Tensor: The loss tensor resulting from the optimization process, used for monitoring as no gradients are attached to it.
        """
        state_goals = [(transition.current_state, transition.goal_condition) for transition in transitions]
        all_policy_logits = self.policy_model.forward(state_goals)
        all_qvalues_1 = self.qvalue_model_1.forward(state_goals)
        all_qvalues_2 = self.qvalue_model_2.forward(state_goals)
        qvalue_targets = self._compute_qvalue_targets(transitions)
        qvalue_losses = self._update_qvalue_models(transitions, all_qvalues_1, all_qvalues_2, qvalue_targets, weights)
        policy_losses = self._update_policy_model(all_qvalues_1, all_qvalues_2, all_policy_logits, weights)
        entropy_losses = self._update_entropy_factor(all_policy_logits, weights)
        self._update_target_models(self.polyak_factor)
        # TODO: Add bounds loss?
        return (qvalue_losses + policy_losses).detach()
