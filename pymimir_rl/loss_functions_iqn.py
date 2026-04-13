import pymimir as mm
import torch

from torch.nn.functional import huber_loss

from .loss_functions import OptimizationFunction
from .models import ActionScalarModel
from .trajectories import Transition


class ActionQuantileModel(ActionScalarModel):
    """
    Interface for IQN models.
    Output tensor shape must be: [num_actions, num_quantiles]
    """
    def forward(self,
                state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
                taus: torch.Tensor | None = None,
                num_quantiles: int = 32) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
        """
        Forward pass.
        Args:
            state_goals: List of state-goal pairs.
            taus: [batch_size, num_quantiles] tensor of quantile fractions.
        Returns:
            List of (QuantileValues, Actions).
        """
        raise NotImplementedError("Must be implemented by subclass.")


class IQNOptimization(OptimizationFunction):
    def __init__(self,
                 model: ActionQuantileModel,
                 model_optimizer: torch.optim.Optimizer,
                 model_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 target_model: ActionQuantileModel,
                 discount_factor: float,
                 num_quantiles: int = 64, # Quantiles for Online network (loss)
                 num_target_quantiles: int = 64, # Quantiles for Target network
                 num_selection_quantiles: int = 32, # Quantiles for Action Selection
                 use_bounds: bool = True # Clip targets to heuristic bounds
                 ) -> None:
        assert isinstance(model, ActionQuantileModel), "Model must be ActionQuantileModel"
        assert isinstance(model_optimizer, torch.optim.Optimizer), "Model optimizer must be Optimizer"
        assert isinstance(model_lr_scheduler, torch.optim.lr_scheduler.LRScheduler), "Model LR scheduler must be LRScheduler"
        assert isinstance(target_model, ActionQuantileModel), "Target model must be ActionQuantileModel"
        assert isinstance(discount_factor, float), "Discount factor must be float"
        assert discount_factor > 0.0, "Discount factor must be positive."
        assert discount_factor <= 1.0, "Discount factor must not be greater than 1."
        assert isinstance(num_quantiles, int) and num_quantiles > 0, "N must be positive integer."
        assert isinstance(num_target_quantiles, int) and num_target_quantiles > 0, "N_prime must be positive integer."
        assert isinstance(num_selection_quantiles, int) and num_selection_quantiles > 0, "K must be positive integer."
        assert isinstance(use_bounds, bool), "use_bounds must be boolean."
        self.model = model
        self.model_optimizer = model_optimizer
        self.model_lr_scheduler = model_lr_scheduler
        self.target_model = target_model
        self.discount_factor = discount_factor
        self.num_quantiles = num_quantiles
        self.num_target_quantiles = num_target_quantiles
        self.num_selection_quantiles = num_selection_quantiles
        self.use_bounds = use_bounds

    def _assert_matching_action_order(self,
                                      reference_actions: list[mm.GroundAction],
                                      candidate_actions: list[mm.GroundAction]) -> None:
        assert len(reference_actions) == len(candidate_actions), "Models must return the same number of applicable actions."
        for reference_action, candidate_action in zip(reference_actions, candidate_actions):
            assert reference_action.get_index() == candidate_action.get_index(), "Models must return applicable actions in the same order."

    def __call__(self, transitions: list[Transition], weights: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch_size = len(transitions)

        self.model.train()
        self.target_model.eval()

        taus = torch.rand(batch_size, self.num_quantiles, device=device)
        state_goals = [(t.current_state, t.goal_condition) for t in transitions]
        current_quantiles_batch = self.model.forward(state_goals, taus=taus)

        # Compute Targets (with bounds clipping if enabled)
        with torch.no_grad():
            target_quantiles = self._compute_target_distributions(transitions, device)

        # Compute IQN Loss
        losses = []
        iterator = zip(current_quantiles_batch, target_quantiles, transitions)
        for i, ((pred_qs, pred_actions), target_dist, transition) in enumerate(iterator):
            assert pred_qs.shape[0] == len(pred_actions), "Model outputs must align with the returned applicable actions."
            assert transition.selected_action in pred_actions, "Selected action must be present in the model's applicable-action list."
            action_idx = pred_actions.index(transition.selected_action)
            current_theta = pred_qs[action_idx]
            u = target_dist.unsqueeze(0) - current_theta.unsqueeze(1)
            huber = huber_loss(u, torch.zeros_like(u), reduction='none')
            tau_expanded = taus[i].unsqueeze(1).expand_as(u)
            diff = torch.abs(tau_expanded - (u < 0).float())
            element_loss = (diff * huber).sum(dim=1).mean(dim=0)
            losses.append(element_loss)

        # Optimization Step
        loss_tensor = torch.stack(losses)
        weighted_loss = loss_tensor * weights.to(device)

        self.model_optimizer.zero_grad()
        weighted_loss.mean().backward()
        self.model_optimizer.step()
        self.model_lr_scheduler.step()

        return loss_tensor.detach()

    def _compute_target_distributions(self, transitions: list[Transition], device: torch.device) -> list[torch.Tensor]:
        lower_bounds, upper_bounds = self.get_value_bounds(transitions, device) if self.use_bounds else (None, None)
        next_states = [(t.successor_state, t.goal_condition) for t in transitions]

        tau_selection = torch.rand(len(transitions), self.num_selection_quantiles, device=device)
        tau_target = torch.rand(len(transitions), self.num_target_quantiles, device=device)

        # Double DQN/IQN target computation: select actions with the online network and evaluate them with the target network.
        batch_qs_selection = self.model.forward(next_states, taus=tau_selection)
        batch_qs_target = self.target_model.forward(next_states, taus=tau_target)

        targets = []
        dead_end_value = -1000.0 # Adjust based on your domain

        for i, transition in enumerate(transitions):
            qs_k, actions_k = batch_qs_selection[i]
            qs_prime, actions_prime = batch_qs_target[i]

            self._assert_matching_action_order(actions_k, actions_prime)

            if len(actions_k) == 0:
                target_dist = torch.full((self.num_target_quantiles,), dead_end_value, device=device)
            else:
                action_means = qs_k.mean(dim=1)
                best_action_idx = torch.argmax(action_means)
                target_dist = qs_prime[best_action_idx]

            reward = transition.immediate_reward
            is_terminal = 1.0 if transition.achieves_goal else 0.0
            updated_dist = reward + self.discount_factor * (1.0 - is_terminal) * target_dist

            # Apply Bounds Clipping (Crucial for Planning)
            if self.use_bounds and lower_bounds is not None and upper_bounds is not None:
                updated_dist = torch.clamp(updated_dist, min=lower_bounds[i], max=upper_bounds[i])

            targets.append(updated_dist)

        return targets
