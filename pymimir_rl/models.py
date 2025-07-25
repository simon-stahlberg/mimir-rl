import pymimir as mm
import torch

from .trajectories import Transition


class QValueModel(torch.nn.Module):
    """
    Abstract base class for model wrappers.
    """

    def forward_transitions(self, transitions: list[Transition]) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            transitions (list[Transition]): The transitions to process.

        Returns:
            list[torch.Tensor]: Q-values associated for each transition.
        """
        raise NotImplementedError("Forward transitions must be implemented by subclass.")

    def forward_state_goals(self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
        """
        Forward pass through the model.

        Args:
            state_goals (list[State, GroundConjunctiveCondition]): The state-goal pairs to process.

        Returns:
            list[tuple[torch.Tensor, list[GroundAction]]]: For each state-goal pair, Q-values with corresponding valid actions.
        """
        raise NotImplementedError("Forward states must be implemented by subclass.")