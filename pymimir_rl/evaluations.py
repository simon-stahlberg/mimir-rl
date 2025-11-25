import math
import pymimir as mm

from abc import ABC, abstractmethod

from .trajectories import Trajectory
from .trajectory_sampling import TrajectorySampler


class EvaluationCriteria(ABC):
    @abstractmethod
    def evaluate(self, trajectories: list[Trajectory]) -> int:
        pass

    @abstractmethod
    def compare(self, x: int, y: int) -> int:
        """
        Compares two evaluation values.

        Args:
            x (int): The first evaluation value.
            y (int): The second evaluation value.

        Returns:
            int: -1 if x is worse than y, 0 if x and y are equal, and 1 if x is better than y.
        """
        pass


class CoverageCriteria(EvaluationCriteria):
    def evaluate(self, trajectories: list[Trajectory]) -> int:
        return sum([(1 if trajectory.is_solution() else 0) for trajectory in trajectories])

    def compare(self, x: int, y: int) -> int:
        return (x > y) - (x < y)


class LengthCriteria(EvaluationCriteria):
    def __init__(self, only_solutions: bool = True) -> None:
        super().__init__()
        self.only_solutions = only_solutions

    def evaluate(self, trajectories: list[Trajectory]) -> int:
        return sum([(len(trajectory) if (not self.only_solutions or trajectory.is_solution()) else 0) for trajectory in trajectories])

    def compare(self, x: int, y: int) -> int:
        return (x < y) - (x > y)


class TDErrorCriteria(EvaluationCriteria):
    def __init__(self, only_solutions: bool = True) -> None:
        super().__init__()
        self.only_solutions = only_solutions

    def evaluate(self, trajectories: list[Trajectory]) -> int:
        total_error = 0.0
        for trajectory in trajectories:
            if not self.only_solutions or trajectory.is_solution():
                for idx in range(len(trajectory) - 1):
                    curr = trajectory[idx]
                    succ = trajectory[idx + 1]
                    q_curr = curr.predicted_q_value
                    q_succ = succ.predicted_q_value + curr.immediate_reward
                    total_error += abs(q_succ - q_curr)
                total_error += abs(trajectory[-1].predicted_q_value)  # Terminal state should have Q-value 0.
        return round(total_error)

    def compare(self, x: int, y: int) -> int:
        return (x < y) - (x > y)


class ProbabilityCriteria(EvaluationCriteria):
    def __init__(self, only_solutions: bool = True) -> None:
        super().__init__()
        self.only_solutions = only_solutions

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def evaluate(self, trajectories: list[Trajectory]) -> int:
        total_error = 0.0
        for trajectory in trajectories:
            if not self.only_solutions or trajectory.is_solution():
                for transition in trajectory:
                    # If the logit is negative, then the transition is undesirable and results in some error.
                    total_error += 1.0 - self._sigmoid(transition.predicted_q_value)
        return round(total_error)

    def compare(self, x: int, y: int) -> int:
        return (x < y) - (x > y)


class PolicyEvaluation:
    """
    Tracks and evaluates the performance of a policy across a set of problems using specified evaluation criteria.

    Args:
        problems (list[mm.Problem]): A list of problem instances to evaluate the policy on.
        criterias (list[EvaluationCriteria]): A list of evaluation criteria to assess policy performance.
        trajectory_sampler (TrajectorySampler): A specific strategy for generating trajectories with the model.
        reward_function (RewardFunction): A function for computing rewards along the generated trajectories.
        horizon (int): Maximum length of the generated trajectories.
    """
    def __init__(self,
                 problems: list[mm.Problem],
                 criterias: list[EvaluationCriteria],
                 trajectory_sampler: TrajectorySampler,
                 horizon: int) -> None:
        self.problems = problems
        self.criterias = criterias
        self.trajectory_sampler = trajectory_sampler
        self.horizon = horizon
        self.best_evaluation = None

    def evaluate(self) -> tuple[bool, list[int]]:
        """
        Evaluates the given model on the set of problems using the specified criteria.

        Args:
            model (ModelWrapper): The model to be evaluated.

        Returns:
            tuple[bool, list[int]]:
                - A boolean indicating whether the evaluation improved upon the best seen so far.
                - A list of evaluation scores for each criterion.
        """
        state_goals = [(problem.get_initial_state(), problem.get_goal_condition()) for problem in self.problems]
        trajectories = self.trajectory_sampler.sample(state_goals, self.horizon)
        # Evaluate on all trajectories.
        evaluation = [criteria.evaluate(trajectories) for criteria in self.criterias]
        if self.best_evaluation is None:
            self.best_evaluation = evaluation
            return True, evaluation
        # Compare evaluations lexicographically.
        for idx, criteria in enumerate(self.criterias):
            comparison = criteria.compare(evaluation[idx], self.best_evaluation[idx])
            if comparison < 0:
                return False, evaluation
            if comparison > 0:
                self.best_evaluation = evaluation
                return True, evaluation
        return False, evaluation


class SequentialPolicyEvaluation:
    """
    Tracks and evaluates the performance of a policy across a set of problems using specified evaluation criteria.
    In contrast to PolicyEvaluation, problems are evaluated sequentially in increasing order of difficulty, and evaluation stops as soon as one problem fails.
    The difficulty is determined by the size of the goal condition, and in case of ties, by the number of objects.
    The given criteria are only applied to the successfully solved problems.

    Args:
        problems (list[mm.Problem]): A list of problem instances to evaluate the policy on.
        criterias (list[EvaluationCriteria]): A list of evaluation criteria to assess policy performance.
        trajectory_sampler (TrajectorySampler): A specific strategy for generating trajectories with the model.
        reward_function (RewardFunction): A function for computing rewards along the generated trajectories.
        horizon (int): Maximum length of the generated trajectories.
    """
    def __init__(self,
                 problems: list[mm.Problem],
                 criterias: list[EvaluationCriteria],
                 trajectory_sampler: TrajectorySampler,
                 horizon: int) -> None:
        # Sort problems by difficulty: first by goal size, then by number of objects.
        self.problems = sorted(problems, key=lambda p: (len(p.get_goal_condition()), len(p.get_objects())))
        self.criterias = criterias
        self.trajectory_sampler = trajectory_sampler
        self.horizon = horizon
        self.best_evaluation = None

    def evaluate(self) -> tuple[bool, list[int]]:
        """
        Evaluates the given model on the set of problems using the specified criteria.

        Args:
            model (ModelWrapper): The model to be evaluated.

        Returns:
            tuple[bool, list[int]]:
                - A boolean indicating whether the evaluation improved upon the best seen so far.
                - A list of evaluation scores for each criterion.
        """
        # Generate trajectories for each problem sequentially.
        successful_trajectories = []
        for problem in self.problems:
            state_goal = (problem.get_initial_state(), problem.get_goal_condition())
            trajectory = self.trajectory_sampler.sample([state_goal], self.horizon)[0]
            if trajectory.is_solution():
                successful_trajectories.append(trajectory)
            else:
                break
        # Evaluate only on the successfully solved problems.
        evaluation = [criteria.evaluate(successful_trajectories) for criteria in self.criterias]
        if self.best_evaluation is None:
            self.best_evaluation = evaluation
            return True, evaluation
        # Compare evaluations lexicographically.
        for idx, criteria in enumerate(self.criterias):
            comparison = criteria.compare(evaluation[idx], self.best_evaluation[idx])
            if comparison < 0:
                return False, evaluation
            if comparison > 0:
                self.best_evaluation = evaluation
                return True, evaluation
        return False, evaluation
