import pymimir as mm
import torch


class ActionScalarModel(torch.nn.Module):
    """
    Abstract base class for model wrappers.
    """

    def forward(self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
        """
        Forward pass through the model.

        Args:
            state_goals (list[State, GroundConjunctiveCondition]): The state-goal pairs to process.

        Returns:
            list[tuple[torch.Tensor, list[GroundAction]]]: For each state-goal pair, Q-values with corresponding valid actions.
        """
        raise NotImplementedError("Forward states must be implemented by subclass.")
