import pymimir as mm
import torch

from abc import ABC, abstractmethod

from .models import ActionScalarModel
from .reward_functions import RewardFunction


class InitialStateSampler(ABC):
    """
    Abstract base class for sampling initial states.
    """

    @abstractmethod
    def sample(self, problems: list[mm.Problem]) -> list[mm.State]:
        """
        Sample initial states for the problems.

        Args:
            problems (list[mm.Problem]): A list of problem instances.

        Returns:
            list[mm.State]: A list of sampled initial states.
        """
        pass


class OriginalInitialStateSampler(InitialStateSampler):
    """
    Samples the initial state that is specified in the PDDL file.
    """

    def sample(self, problems: list[mm.Problem]) -> list[mm.State]:
        return [problem.get_initial_state() for problem in problems]


class TopValueInitialStateSampler(InitialStateSampler):
    """
    Samples the initial state based on their predicted Q-values with respect to the original goal.
    """

    def __init__(self, problems: list[mm.Problem], model: ActionScalarModel, reward_function: RewardFunction, sample_original_probability: float, temperature: float, max_buffer_size: int) -> None:
        super().__init__()
        assert isinstance(sample_original_probability, float), "Sample original probability must be a float."
        assert isinstance(temperature, float), "Temperature must be a float."
        assert isinstance(max_buffer_size, int), "Max buffer size must be an integer."
        assert sample_original_probability >= 0.0, "Sample original probability must not be negative."
        assert sample_original_probability <= 1.0, "Sample original probability must be at most 1."
        assert max_buffer_size > 0, "Max buffer size must be positive."
        self.problems = problems
        self.model = model
        self.reward_function = reward_function
        self.sample_original_probability = sample_original_probability
        self.temperature = temperature
        self.max_buffer_size = max_buffer_size
        self.state_buffers: dict[mm.Problem, list[mm.State]] = {problem: [problem.get_initial_state()] for problem in problems}
        self.value_buffers: dict[mm.Problem, list[float]] = {problem: [0.0] for problem in problems}

    def _update_state_values(self, problem: mm.Problem) -> None:
        with torch.no_grad():
            # TODO: This code can likely be cleaned up.
            self.model.eval()
            device = next(self.model.parameters()).device
            states = self.state_buffers[problem]
            values = self.value_buffers[problem]
            goal_condition = problem.get_goal_condition()
            state_goals = [(state, goal_condition) for state in states]
            qvalues_actions_list = self.model.forward(state_goals)
            assert len(qvalues_actions_list) == len(states), "Model forward must return one result per input state."
            new_value_list: list[torch.Tensor] = []
            for state, (q_values, actions) in zip(states, qvalues_actions_list):
                assert q_values.ndim == 1, "Initial-state sampling requires one scalar Q-value per action."
                assert q_values.numel() == len(actions), "Q-values and applicable actions must have equal lengths."
                if goal_condition.holds(state):
                    value = torch.tensor(0.0, dtype=torch.float, device=device)
                elif (len(state.generate_applicable_actions()) == 0 or
                      self.reward_function.is_dead_end(state, goal_condition)):
                    value = torch.tensor(RewardFunction.get_dead_end_reward(), dtype=torch.float, device=device)
                else:
                    assert q_values.numel() > 0, "A live state must have at least one applicable action."
                    value = q_values.max()
                new_value_list.append(value)
            new_values = torch.stack(new_value_list)
            assert len(values) == new_values.numel()
            for idx in range(len(values)):
                values[idx] = new_values[idx].item()

    def add_state(self, state: mm.State, value: float) -> None:
        def argmin(xs: list[float]) -> int:
            return min(range(len(xs)), key=lambda x: xs[x])
        states = self.state_buffers[state.get_problem()]
        values = self.value_buffers[state.get_problem()]
        if len(values) >= self.max_buffer_size:
            replace_idx = argmin(values)
            replace_value = values[replace_idx]
            if value > replace_value:
                states[replace_idx] = state
                values[replace_idx] = value
        else:
            states.append(state)
            values.append(value)

    def sample(self, problems: list[mm.Problem]) -> list[mm.State]:
        sampled_initial_states: list[mm.State] = []
        updated_problems: set[mm.Problem] = set()
        for problem in problems:
            sample_original_initial_state = torch.rand(1).item() < self.sample_original_probability
            if sample_original_initial_state:
                sampled_initial_states.append(problem.get_initial_state())
            else:
                if problem not in updated_problems:
                    self._update_state_values(problem)
                    updated_problems.add(problem)
                states = self.state_buffers[problem]
                values = self.value_buffers[problem]
                values_tensor = torch.tensor(values, dtype=torch.float) / self.temperature
                probabilities = values_tensor.softmax(dim=0)
                state_idx = probabilities.multinomial(num_samples=1)
                sampled_initial_states.append(states[state_idx])
        return sampled_initial_states
