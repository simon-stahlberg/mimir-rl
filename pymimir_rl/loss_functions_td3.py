import torch
import torch.nn.functional as F
import pymimir as mm

from typing import Callable

from .loss_functions import OptimizationFunction
from .models import ActionScalarModel
from .trajectories import Transition


class DiscreteTD3Optimization(OptimizationFunction):
    def __init__(self,
                 policy_model: ActionScalarModel,
                 policy_optimizer: torch.optim.Optimizer,
                 policy_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 policy_target: ActionScalarModel,
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
                 policy_delay: int = 2) -> None:
        """
        Initializes the DiscreteTD3Optimization class.

        Args:
            policy_model (ActionScalarModel): The policy model to be optimized.
            policy_optimizer (torch.optim.Optimizer): The optimizer for the policy model.
            policy_lr_scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler for the policy optimizer.
            policy_target (ActionScalarModel): The target policy model.
            qvalue_target_1 (ActionScalarModel): The first Q-value model to compute targets.
            qvalue_model_1 (ActionScalarModel): The first Q-value model to be optimized.
            qvalue_optimizer_1 (torch.optim.Optimizer): The optimizer for the first Q-value model.
            qvalue_lr_scheduler_1 (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler for the first Q-value optimizer.
            qvalue_target_2 (ActionScalarModel): The second Q-value model to compute targets.
            qvalue_model_2 (ActionScalarModel): The second Q-value model to be optimized.
            qvalue_optimizer_2 (torch.optim.Optimizer): The optimizer for the second Q-value model.
            qvalue_lr_scheduler_2 (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler for the second Q-value optimizer.
            discount_factor (float): The discount factor for future rewards.
            polyak_factor (float): The rate at which the target models are updated. Defaults to 0.005.
            policy_delay (int): Number of critic updates before an actor/target update. Defaults to 2.
        """
        super().__init__()
        assert isinstance(policy_model, ActionScalarModel)
        assert isinstance(policy_optimizer, torch.optim.Optimizer)
        assert isinstance(policy_target, ActionScalarModel)
        assert isinstance(qvalue_target_1, ActionScalarModel)
        assert isinstance(qvalue_model_1, ActionScalarModel)
        assert isinstance(qvalue_optimizer_1, torch.optim.Optimizer)
        assert isinstance(qvalue_target_2, ActionScalarModel)
        assert isinstance(qvalue_model_2, ActionScalarModel)
        assert isinstance(qvalue_optimizer_2, torch.optim.Optimizer)
        assert isinstance(discount_factor, float) and 0.0 < discount_factor <= 1.0
        assert isinstance(polyak_factor, float) and 0.0 < polyak_factor <= 1.0
        assert isinstance(policy_delay, int) and policy_delay >= 1
        self.policy_model = policy_model
        self.policy_optimizer = policy_optimizer
        self.policy_lr_scheduler = policy_lr_scheduler
        self.policy_target = policy_target
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
        self.policy_delay = policy_delay
        self._update_step = 0  # Internal step counter for delayed updates
        self._update_targets(1.0)  # Initialize targets
        self._listeners_losses: list[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]] = []  # Listener lists

    def register_on_losses(self, callback: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]):
        """
        Register a callback to receive actor and critic losses.
        Args:
           callback (Callable): Function receiving (actor_loss, critic_loss_1, critic_loss_2).
        """
        self._listeners_losses.append(callback)

    def _notify_listeners_losses(self, actor_loss: torch.Tensor, critic_loss_1: torch.Tensor, critic_loss_2: torch.Tensor):
        for listener in self._listeners_losses:
            listener(actor_loss, critic_loss_1, critic_loss_2)

    def _compute_qvalue_targets(self, transitions: list[Transition]) -> torch.Tensor:
        """
        Computes the target Q-value: r + gamma * min(Q_t1(s', pi_t(s')), Q_t2(s', pi_t(s')))
        Uses deterministic selection (argmax) for the target policy.
        """
        with torch.no_grad():
            device = next(self.policy_model.parameters()).device
            dead_end_value = -10000
            successor_state_goals = [(transition.successor_state, transition.goal_condition) for transition in transitions]

            # 1. Get Target Policy Actions (Deterministic/Greedy)
            batch_target_policy_logits = self.policy_target.forward(successor_state_goals)

            # 2. Get Target Q-Values
            batch_target_q1 = self.qvalue_target_1.forward(successor_state_goals)
            batch_target_q2 = self.qvalue_target_2.forward(successor_state_goals)

            target_qvalues: list[torch.Tensor] = []

            # Zip everything: Logits, Q1, Q2
            iterator = zip(batch_target_policy_logits, batch_target_q1, batch_target_q2)

            for (policy_logits, _), (qvalues_1, _), (qvalues_2, _) in iterator:
                if policy_logits.numel() > 0:
                    # Discrete TD3: Select action with highest logit (Deterministic Policy)
                    # We do not strictly need "target policy smoothing" noise here as typically done in Continuous TD3,
                    # but relying on the min(Q1, Q2) provides the necessary pessimism.
                    best_action_idx = policy_logits.argmax(dim=-1)

                    q1_val = qvalues_1[best_action_idx]
                    q2_val = qvalues_2[best_action_idx]

                    min_q = torch.min(q1_val, q2_val)
                    target_qvalues.append(min_q)
                else:
                    # Dead end
                    target_qvalues.append(torch.tensor(dead_end_value, device=device, dtype=torch.float))

            immediate_rewards = torch.tensor([t.immediate_reward for t in transitions], device=device, dtype=torch.float)
            achieves_goal = torch.tensor([t.achieves_goal for t in transitions], device=device, dtype=torch.float)

            discounted_targets = immediate_rewards + (1.0 - achieves_goal) * self.discount_factor * torch.stack(target_qvalues)
            return discounted_targets

    def _compute_critic_losses(self,
                               transitions: list[Transition],
                               batch_qvalues_1: list[tuple[torch.Tensor, list[mm.GroundAction]]],
                               batch_qvalues_2: list[tuple[torch.Tensor, list[mm.GroundAction]]]) -> tuple[torch.Tensor, torch.Tensor]:

        target_q = self._compute_qvalue_targets(transitions)

        # Extract Q-values for the specifically selected actions in the trajectory
        selected_q1 = torch.stack([
            qvalues[actions.index(t.selected_action)]
            for (qvalues, actions), t in zip(batch_qvalues_1, transitions)
        ])
        selected_q2 = torch.stack([
            qvalues[actions.index(t.selected_action)]
            for (qvalues, actions), t in zip(batch_qvalues_2, transitions)
        ])

        loss_1 = F.huber_loss(selected_q1, target_q, reduction='none')
        loss_2 = F.huber_loss(selected_q2, target_q, reduction='none')

        return loss_1, loss_2

    def _compute_actor_loss(self,
                            batch_qvalues_1: list[tuple[torch.Tensor, list[mm.GroundAction]]],
                            batch_policy_logits: list[tuple[torch.Tensor, list[mm.GroundAction]]]) -> torch.Tensor:
        """
        Computes actor loss.
        We want to maximize Q1(s, pi(s)).
        Since argmax is not differentiable, we use Gumbel-Softmax (hard=True) or Softmax
        to weight the Q-values, allowing gradients to propagate back to policy logits.
        """
        batch_losses: list[torch.Tensor] = []

        for (policy_logits, _), (qvalues_1, _) in zip(batch_policy_logits, batch_qvalues_1):
            if policy_logits.numel() > 0:
                # Gumbel-Softmax with hard=True approximates the deterministic action selection
                # (one-hot vector) but allows gradient flow.
                # Alternatively, simple softmax can be used: probs = policy_logits.softmax(dim=-1)

                # Using standard softmax weighting for stability in sparse reward / discrete settings:
                probs = policy_logits.softmax(dim=-1)

                # We detach Q-values because we are updating the Actor, not the Critic here.
                actor_value = (probs * qvalues_1.detach()).sum()

                # We want to MAXIMIZE value, so loss is negative.
                batch_losses.append(-actor_value)
            else:
                batch_losses.append(torch.tensor(0.0, device=policy_logits.device))

        return torch.stack(batch_losses)

    def _update_targets(self, polyak: float) -> None:
        with torch.no_grad():
            # Update Critic Targets
            for target, param in zip(self.qvalue_target_1.parameters(), self.qvalue_model_1.parameters()):
                target.copy_(polyak * param + (1.0 - polyak) * target)
            for target, param in zip(self.qvalue_target_2.parameters(), self.qvalue_model_2.parameters()):
                target.copy_(polyak * param + (1.0 - polyak) * target)
            # Update Actor Target
            for target, param in zip(self.policy_target.parameters(), self.policy_model.parameters()):
                target.copy_(polyak * param + (1.0 - polyak) * target)

    def __call__(self, transitions: list[Transition], weights: torch.Tensor) -> torch.Tensor:
        self._update_step += 1

        state_goals = [(t.current_state, t.goal_condition) for t in transitions]

        # 1. Forward passes
        batch_qvalues_1 = self.qvalue_model_1.forward(state_goals)
        batch_qvalues_2 = self.qvalue_model_2.forward(state_goals)

        # 2. Update Critics (Always)
        critic_loss_1, critic_loss_2 = self._compute_critic_losses(transitions, batch_qvalues_1, batch_qvalues_2)

        self.qvalue_optimizer_1.zero_grad()
        (critic_loss_1 * weights.to(critic_loss_1.device)).mean().backward()
        self.qvalue_optimizer_1.step()
        self.qvalue_lr_scheduler_1.step()

        self.qvalue_optimizer_2.zero_grad()
        (critic_loss_2 * weights.to(critic_loss_2.device)).mean().backward()
        self.qvalue_optimizer_2.step()
        self.qvalue_lr_scheduler_2.step()

        # 3. Delayed Actor Updates
        actor_loss_val = torch.tensor(0.0, device=critic_loss_1.device)

        if self._update_step % self.policy_delay == 0:
            # Re-compute policy logits to generate graph for backprop
            batch_policy_logits = self.policy_model.forward(state_goals)

            # Note: We must re-fetch Q1 values or use the ones from above.
            # If we reuse batch_qvalues_1, we must ensure the graph is retained or we rely on detached Qs.
            # In _compute_actor_loss, we detach Qs, so reusing batch_qvalues_1 is safe.

            actor_losses = self._compute_actor_loss(batch_qvalues_1, batch_policy_logits)

            self.policy_optimizer.zero_grad()
            (actor_losses * weights.to(actor_losses.device)).mean().backward()
            self.policy_optimizer.step()
            self.policy_lr_scheduler.step()

            # Update Targets
            self._update_targets(self.polyak_factor)

            actor_loss_val = actor_losses.detach().mean()

        # Notify listeners
        self._notify_listeners_losses(
            actor_loss_val,
            critic_loss_1.detach().mean(),
            critic_loss_2.detach().mean()
        )

        return (critic_loss_1 + critic_loss_2).detach()
