import torch

from abc import ABC, abstractmethod

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
