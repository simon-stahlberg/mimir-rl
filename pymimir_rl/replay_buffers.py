import torch

from abc import ABC, abstractmethod

from .trajectories import Transition


class ReplayBuffer(ABC):
    """
    Abstract base class for replay buffers.
    """

    @abstractmethod
    def push(self, transition: Transition) -> None:
        """
        Adds a transition to the replay buffer.
        """
        pass

    @abstractmethod
    def sample(self, n: int) -> tuple[list[Transition], torch.Tensor, torch.Tensor]:
        """
        Samples a batch of transitions from the replay buffer.

        Args:
            n (int): Number of transitions to sample.

        Returns:
            list[Transition]: A list of sampled transitions.
            torch.Tensor: Importance sampling weights.
            torch.Tensor: Indices of sampled transitions.
        """
        pass

    def update(self, indices: torch.Tensor, errors: torch.Tensor) -> None:
        """
        Updates the priorities of sampled transitions based on errors.

        Args:
            indices (torch.Tensor): Indices of sampled transitions.
            errors (torch.Tensor): Computed errors for sampled transitions.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the current number of stored transitions.
        """
        pass


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    A Prioritized Experience Replay (PER) buffer that prioritizes sampling transitions
    based on their temporal difference (TD) error.

    Attributes:
        capacity (int): Maximum number of transitions the buffer can store.
        alpha (float): Controls the level of prioritization (0 = uniform sampling, 1 = full prioritization).
        beta (float): Controls importance sampling correction for bias reduction.
        beta_start (float): Initial beta value.
        beta_frames (int): Number of frames over which beta is annealed.
        buffer (list): Stores the experience tuples (state, action, reward, next_state, done).
        priorities (np.array): Stores the priorities of the transitions.
        position (int): Tracks the current position for overwriting old transitions.
        frame (int): Tracks the number of updates to anneal beta over time.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        """
        Initializes the Prioritized Experience Replay buffer.

        Args:
            capacity (int): Maximum buffer size.
            alpha (float, optional): Degree of prioritization (default=0.6).
            beta_start (float, optional): Initial beta value for importance sampling (default=0.4).
            beta_frames (int, optional): Number of frames to anneal beta to 1.0 (default=100000).
        """
        self.capacity: int = capacity
        self.alpha: float = alpha
        self.beta_start: float = beta_start
        self.beta_frames: int = beta_frames
        self.buffer: list[Transition] = []
        self.position: int = 0
        self.priorities = torch.zeros(capacity, dtype=torch.float)
        self.beta: float = beta_start
        self.frame: int = 1  # Tracks number of updates for beta annealing.

    def push(self, transition: Transition) -> None:
        """
        Adds a transition to the replay buffer with the highest priority.
        Replaces the transition with the lowest priority when full.

        Args:
            state (np.array): The current state. action (int): The action taken.
            reward (float): The reward received. next_state (np.array): The next
            state. done (bool): Whether the episode ended.
        """
        max_priority = self.priorities.max() if self.buffer else 1.0  # New transitions get max priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def sample(self, n: int) -> tuple[list[Transition], torch.Tensor, torch.Tensor]:
        """
        Samples a batch of transitions based on their priorities.

        Args:
            n (int): Number of transitions to sample.

        Returns:
            tuple: (transitions, weights, indices)
                - transitions (list[Transition]): Sampled transitions.
                - weights (np.array): Importance sampling weights.
                - indices (np.array): Indices of sampled transitions.
        """
        if len(self.buffer) == 0:
            return ([], torch.tensor([]), torch.tensor([]))
        # Compute probability distribution.
        priorities = self.priorities[:len(self.buffer)] ** self.alpha
        probabilities = priorities / priorities.sum()
        # Sample indices based on priority probabilities.
        indices = torch.multinomial(probabilities, n, replacement=True)
        samples: list[Transition] = [self.buffer[idx] for idx in indices]
        # Compute importance sampling (IS) weights.
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** -self.beta  # Normalize.
        weights /= weights.max()
        # Anneal beta towards 1.0.
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        self.frame += 1
        # Return the sampled batch.
        return samples, weights, indices

    def update(self, indices: torch.Tensor, errors: torch.Tensor) -> None:
        """
        Updates the priorities of sampled transitions based on TD errors.

        Args:
            indices (np.array): Indices of sampled transitions.
            errors (np.array): Computed TD errors for sampled transitions.
        """
        self.priorities[indices] = errors.abs() + 1e-5

    def get_mean_priority(self) -> float:
        if len(self.buffer) == 0: return 0.0
        return self.priorities[:len(self.buffer)].mean().item()

    def get_std_priority(self) -> float:
        if len(self.buffer) == 0: return 0.0
        return self.priorities[:len(self.buffer)].std().item()

    def __len__(self) -> int:
        """
        Returns the current number of stored transitions.

        Returns:
            int: Number of transitions in the buffer.
        """
        return len(self.buffer)