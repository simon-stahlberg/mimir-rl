import pymimir as mm
import torch

from typing import Callable

from .goal_condition_sampling import GoalConditionSampler, OriginalGoalConditionSampler
from .initial_state_sampling import InitialStateSampler, OriginalInitialStateSampler
from .loss_functions import LossFunction
from .models import ActionScalarModel
from .problem_sampling import ProblemSampler, UniformProblemSampler
from .replay_buffers import ReplayBuffer
from .reward_functions import RewardFunction
from .trajectories import Trajectory, Transition
from .trajectory_refinements import TrajectoryRefiner, IdentityTrajectoryRefiner
from .trajectory_sampling import TrajectorySampler


class OffPolicyAlgorithm:
    """
    Modular off-policy RL algorithm.
    """

    def __init__(self,
                 problems: list[mm.Problem],
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 loss_function: LossFunction,
                 reward_function: RewardFunction,
                 replay_buffer: ReplayBuffer,
                 trajectory_sampler: TrajectorySampler,
                 horizon: int,
                 rollout_count: int,
                 batch_size: int,
                 train_steps: int,
                 problem_sampler: ProblemSampler | None = None,
                 initial_state_sampler: InitialStateSampler |  None = None,
                 goal_condition_sampler: GoalConditionSampler |  None = None,
                 trajectory_refiner: TrajectoryRefiner |  None = None) -> None:
        """
        Initialize the off-policy RL algorithm with the specified components.

        Args:
            problems (list[mm.Problem]): List of problem instances.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            lr_scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler for the optimizer.
            loss_function (LossFunction): Function to compute losses.
            reward_function (RewardFunction): Function to compute rewards.
            replay_buffer (ReplayBuffer): Buffer to store experience transitions.
            trajectory_sampler (TrajectorySampler): Sampler for generating trajectories.
            horizon (int): Maximum length of the sampled trajectories.
            rollout_count (int): Number of sampled trajectories.
            batch_size (int): Mini-batch size used for each optimization step.
            train_steps (int): Number of optimization steps to perform.
            problem_sampler (ProblemSampler, optional): Sampler for selecting problems. Defaults to UniformProblemSampler.
            initial_state_sampler (InitialStateSampler, optional): Sampler for initial states. Defaults to OriginalInitialStateSampler.
            goal_condition_sampler (GoalConditionSampler, optional): Sampler for goal conditions. Defaults to OriginalGoalConditionSampler.
            trajectory_refiner (TrajectoryRefiner, optional): Refiner for trajectories. Defaults to IdentityTrajectoryRefiner.
        """
        assert isinstance(problems, list) and all(isinstance(problem, mm.Problem) for problem in problems), "Problems must be a list of mm.Problem instances."
        assert len(problems) > 0, "At least one problem must be provided."
        assert isinstance(optimizer, torch.optim.Optimizer), "Optimizer must be an instance of torch.optim.Optimizer."
        assert isinstance(lr_scheduler, torch.optim.lr_scheduler.LRScheduler), "LR Scheduler must be an instance of torch.optim.lr_scheduler.LRScheduler."
        assert isinstance(loss_function, LossFunction), "Loss function must be an instance of LossFunction."
        assert isinstance(reward_function, RewardFunction), "Reward function must be an instance of RewardFunction."
        assert isinstance(replay_buffer, ReplayBuffer), "Replay buffer must be an instance of RewardBuffer."
        assert isinstance(trajectory_sampler, TrajectorySampler), "Trajectory sampler must be an instance of TrajectorySampler."
        assert isinstance(horizon, int), "Horizon must be an integer."
        assert horizon > 0, "Horizon must be positive."
        assert isinstance(train_steps, int), "Train steps must be an integer."
        assert train_steps > 0, "Train steps must be positive."
        self.problems = problems
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.reward_function = reward_function
        self.replay_buffer = replay_buffer
        self.trajectory_sampler = trajectory_sampler
        self.horizon = horizon
        self.rollout_count = rollout_count
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.problem_sampler = problem_sampler or UniformProblemSampler()
        self.initial_state_sampler = initial_state_sampler or OriginalInitialStateSampler()
        self.goal_condition_sampler = goal_condition_sampler or OriginalGoalConditionSampler()
        self.trajectory_refiner = trajectory_refiner or IdentityTrajectoryRefiner()
        assert isinstance(self.problem_sampler, ProblemSampler), "Problem sampler must be an instance of ProblemSampler."
        assert isinstance(self.initial_state_sampler, InitialStateSampler), "Initial state sampler must be an instance of InitialStateSampler."
        assert isinstance(self.goal_condition_sampler, GoalConditionSampler), "Goal condition sampler must be an instance of GoalConditionSampler."
        assert isinstance(self.trajectory_refiner, TrajectoryRefiner), "Trajectory refiner must be an instance of TrajectoryRefiner."
        # Initialize listener lists.
        self._listeners_sample_problems: list[Callable[[list[mm.Problem]], None]] = []
        self._listeners_sample_initial_states: list[Callable[[list[mm.State]], None]] = []
        self._listeners_sample_goal_conditions: list[Callable[[list[mm.GroundConjunctiveCondition]], None]] = []
        self._listeners_sample_trajectories: list[Callable[[list[Trajectory]], None]] = []
        self._listeners_refine_trajectories: list[Callable[[list[Trajectory]], None]] = []
        self._listeners_pre_collect_experience: list[Callable[[], None]] = []
        self._listeners_post_collect_experience: list[Callable[[], None]] = []
        self._listeners_train_step: list[Callable[[list[Transition], torch.Tensor], None]] = []
        self._listeners_pre_optimize_model: list[Callable[[], None]] = []
        self._listeners_post_optimize_model: list[Callable[[], None]] = []

    def _notify_sample_problems(self, result: list[mm.Problem]) -> None:
        for listener in self._listeners_sample_problems:
            listener(result)

    def _notify_sample_initial_states(self, result: list[mm.State]) -> None:
        for listener in self._listeners_sample_initial_states:
            listener(result)

    def _notify_sample_goal_conditions(self, result: list[mm.GroundConjunctiveCondition]) -> None:
        for listener in self._listeners_sample_goal_conditions:
            listener(result)

    def _notify_sample_trajectories(self, result: list[Trajectory]) -> None:
        for listener in self._listeners_sample_trajectories:
            listener(result)

    def _notify_refine_trajectories(self, result: list[Trajectory]) -> None:
        for listener in self._listeners_refine_trajectories:
            listener(result)

    def _notify_pre_collect_experience(self) -> None:
        for listener in self._listeners_pre_collect_experience:
            listener()

    def _notify_post_collect_experience(self) -> None:
        for listener in self._listeners_post_collect_experience:
            listener()

    def _notify_train_step(self, transitions: list[Transition], losses: torch.Tensor) -> None:
        for listener in self._listeners_train_step:
            listener(transitions, losses)

    def _notify_pre_optimize_model(self) -> None:
        for listener in self._listeners_pre_optimize_model:
            listener()

    def _notify_post_optimize_model(self) -> None:
        for listener in self._listeners_post_optimize_model:
            listener()

    def register_sample_problems(self, callback: Callable[[list[mm.Problem]], None]) -> None:
        self._listeners_sample_problems.append(callback)

    def register_sample_initial_states(self, callback: Callable[[list[mm.State]], None]) -> None:
        self._listeners_sample_initial_states.append(callback)

    def register_sample_goal_conditions(self, callback: Callable[[list[mm.GroundConjunctiveCondition]], None]) -> None:
        self._listeners_sample_goal_conditions.append(callback)

    def register_sample_trajectories(self, callback: Callable[[list[Trajectory]], None]) -> None:
        self._listeners_sample_trajectories.append(callback)

    def register_refine_trajectories(self, callback: Callable[[list[Trajectory]], None]) -> None:
        self._listeners_refine_trajectories.append(callback)

    def register_pre_collect_experience(self, callback: Callable[[], None]) -> None:
        self._listeners_pre_collect_experience.append(callback)

    def register_post_collect_experience(self, callback: Callable[[], None]) -> None:
        self._listeners_post_collect_experience.append(callback)

    def register_train_step(self, callback: Callable[[list[Transition], torch.Tensor], None]) -> None:
        self._listeners_train_step.append(callback)

    def register_pre_optimize_model(self, callback: Callable[[], None]) -> None:
        self._listeners_pre_optimize_model.append(callback)

    def register_post_optimize_model(self, callback: Callable[[], None]) -> None:
        self._listeners_post_optimize_model.append(callback)

    def sample_problems(self, n: int) -> list[mm.Problem]:
        """
        Sample problems for the strategy.

        Args:
            n (int): The number of problems to sample.

        Returns:
            list[mm.Problem]: A list of sampled problems.
        """
        result = self.problem_sampler.sample(self.problems, n)
        self._notify_sample_problems(result)
        return result

    def sample_initial_states(self, problems: list[mm.Problem]) -> list[mm.State]:
        """
        Sample an initial state for the strategy.

        Args:
            problem (mm.Problem): The problem instance to sample the initial state from.

        Returns:
            mm.State: An initial state.
        """
        result = self.initial_state_sampler.sample(problems)
        self._notify_sample_initial_states(result)
        return result

    def sample_goal_conditions(self, problems: list[mm.Problem]) -> list[mm.GroundConjunctiveCondition]:
        """
        Sample goal conditions for the strategy.

        Args:
            problems (list[mm.Problem]): A list of problem instances to sample goal conditions from.

        Returns:
            list[mm.GroundConjunctiveCondition]: A list of sampled goal conditions.
        """
        result = self.goal_condition_sampler.sample(problems)
        self._notify_sample_goal_conditions(result)
        return result

    def sample_trajectories(self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]) -> list[Trajectory]:
        """
        Sample trajectories from the given problems.

        Args:
            state_goals (list[tuple[mm.State, mm.GroundConjunctiveCondition]]): A list of tuples containing states and their corresponding goals.
            model (ModelWrapper): A model used to sample successor states in the rollout process.
            reward_function (RewardFunction): A reward function to compute the rewards along the trajectory.

        Returns:
            list[Trajectory]: A list of sampled trajectories.
        """
        result = self.trajectory_sampler.sample(state_goals, self.horizon)
        self._notify_sample_trajectories(result)
        return result

    def refine_trajectories(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """
        Refine the sampled trajectories.

        Args:
            trajectories (list[Trajectory]): A list of sampled trajectories.

        Returns:
            list[Trajectory]: A refined list of trajectories.
        """
        result = self.trajectory_refiner.refine(trajectories)
        self._notify_refine_trajectories(result)
        return result

    def fit(self) -> None:
        """
        Run one step of experience collection with refinement and model optimization.
        """
        self.collect_experience(self.rollout_count)
        self.optimize_model(self.batch_size)

    def collect_experience(self, rollout_size: int) -> None:
        """
        Collect experience from the sampled trajectories.

        Args:
            rollout_size (int): The number of trajectories to sample.
        """
        self._notify_pre_collect_experience()
        with torch.no_grad():
            problems = self.sample_problems(rollout_size)
            initial_states = self.sample_initial_states(problems)
            goal_conditions = self.sample_goal_conditions(problems)
            state_goals = list(zip(initial_states, goal_conditions))
            trajectories = self.sample_trajectories(state_goals)
            refined_trajectories = self.refine_trajectories(trajectories)
            for refined_trajectory in refined_trajectories:
                for refined_transition in refined_trajectory:
                    self.replay_buffer.push(refined_transition)
        self._notify_post_collect_experience()

    def optimize_model(self, batch_size: int) -> None:
        """
        Update the strategy using a batch of data.

        Args:
            batch_size (int): The batch size for model optimization.
        """
        self._notify_pre_optimize_model()
        if len(self.replay_buffer) > 0:
            for _ in range(self.train_steps):
                transitions, weights, indices = self.replay_buffer.sample(batch_size)
                self.optimizer.zero_grad()
                losses = self.loss_function(transitions)
                loss = (losses * weights).mean()
                loss.backward()  # type: ignore
                self.optimizer.step()
                self.replay_buffer.update(indices, losses.detach().cpu())
                self._notify_train_step(transitions, losses.detach())
            self.lr_scheduler.step()
        self._notify_post_optimize_model()
