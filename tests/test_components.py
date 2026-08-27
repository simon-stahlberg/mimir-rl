import math
from typing import Any, Callable, cast
import pymimir as mm
import pytest
import torch
import pymimir_rl.subtrajectory_sampling as subtrajectory_sampling_module

from pathlib import Path
from pymimir_rgnn import (
    RelationalGraphNeuralNetwork,
    HyperparameterConfig,
    ModuleConfig,
    SumAggregation,
    PredicateMLPMessages,
    MLPUpdates,
    StateEncoder,
    GroundActionsEncoder,
    GoalEncoder,
    ActionScalarDecoder
)
from pymimir_rl import *
from pymimir_rl.reward_functions import RewardFunction
from pymimir_rl.trajectory_sampling import TrajectoryState


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'


class RGNNWrapper(ActionScalarModel):
    def __init__(self, domain: mm.Domain) -> None:
        super().__init__()  # type: ignore

        hparam_config = HyperparameterConfig(
            domain=domain,
            num_layers=4,
            embedding_size=8
        )

        input_spec = (StateEncoder(), GroundActionsEncoder(), GoalEncoder())
        output_spec = [('q', ActionScalarDecoder(hparam_config))]

        module_config = ModuleConfig(
            aggregation_function=SumAggregation(),
            message_function=PredicateMLPMessages(hparam_config, input_spec),
            update_function=MLPUpdates(hparam_config)
        )

        self.rgnn = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore

    def forward(self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
        input_list: list[tuple[mm.State, list[mm.GroundAction], mm.GroundConjunctiveCondition]] = []
        actions_list: list[list[mm.GroundAction]] = []
        for state, goal in state_goals:
            actions = list(state.applicable_actions())
            input_list.append((state, actions, goal))
            actions_list.append(actions)
        q_values_list: list[torch.Tensor] = self.rgnn.forward(input_list).readout('q')  # type: ignore
        output = list(zip(q_values_list, actions_list))
        for tensor, _ in output:
            assert not tensor.isnan().any()
            assert not tensor.isinf().any()
        return output


class DummyIQNWrapper(ActionQuantileModel):
    def __init__(self, action_values: list[float]) -> None:
        super().__init__()  # type: ignore
        self.action_values = torch.nn.Parameter(torch.tensor(action_values, dtype=torch.float))

    def forward(self,
                state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]],
                taus: torch.Tensor | None = None,
                num_quantiles: int = 32) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
        quantile_count = num_quantiles if taus is None else taus.shape[1]
        output: list[tuple[torch.Tensor, list[mm.GroundAction]]] = []
        for state, goal in state_goals:
            actions = list(state.applicable_actions())
            assert len(actions) <= self.action_values.numel()
            if len(actions) == 0:
                quantiles = torch.zeros((0, quantile_count), device=self.action_values.device)
            else:
                quantiles = self.action_values[:len(actions)].unsqueeze(1).repeat(1, quantile_count)
            output.append((quantiles, actions))
        return output


class FakeBeamState:
    def __init__(self, name: str) -> None:
        self.name = name
        self._actions: list[FakeBeamAction] = []

    def set_actions(self, actions: list['FakeBeamAction']) -> None:
        self._actions = actions

    def applicable_actions(self) -> tuple['FakeBeamAction', ...]:
        return tuple(self._actions)

    @property
    def problem(self) -> None:
        return None

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeBeamState) and self.name == other.name

    def __repr__(self) -> str:
        return f"FakeBeamState({self.name})"


class FakeBeamAction:
    def __init__(self, name: str, successor_state: FakeBeamState, reward: float = 0.0) -> None:
        self.name = name
        self.successor_state = successor_state
        self.reward = reward

    def apply(self, state: FakeBeamState) -> FakeBeamState:
        return self.successor_state

    def __repr__(self) -> str:
        return f"FakeBeamAction({self.name})"


class FakeSearchTransition:
    def __init__(self,
                 source: FakeBeamState,
                 action: FakeBeamAction,
                 target: FakeBeamState) -> None:
        self.source = source
        self.action = action
        self.target = target


class FakeBeamGoalCondition:
    def __init__(self, goal_states: set[FakeBeamState] | None = None) -> None:
        self.goal_states = goal_states or set()

    def holds(self, state: FakeBeamState) -> bool:
        return state in self.goal_states


class DummyBeamModel(ActionScalarModel):
    def forward(self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
        raise NotImplementedError("Beam internal tests do not call the model.")


class FixedBeamModel(ActionScalarModel):
    def __init__(self, action_values: list[float], dtype: torch.dtype = torch.float) -> None:
        super().__init__()
        self.action_values = torch.nn.Parameter(torch.tensor(action_values, dtype=dtype))

    def forward(self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
        output: list[tuple[torch.Tensor, list[mm.GroundAction]]] = []
        for state, _ in state_goals:
            actions = list(state.applicable_actions())
            output.append((self.action_values[:len(actions)], actions))
        return output


class ShortBatchBeamModel(FixedBeamModel):
    def forward(self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
        return super().forward(state_goals)[:-1]


class DummyBeamRewardFunction(RewardFunction):
    def __init__(self, constant: float, dead_end_states: set[FakeBeamState] | None = None) -> None:
        self.constant = constant
        self.dead_end_states = dead_end_states or set()

    def __call__(self,
                 current_state: mm.State,
                 action: mm.GroundAction,
                 successor_state: mm.State,
                 goal_condition: mm.GroundConjunctiveCondition) -> float:
        return self.constant

    def is_dead_end(self, state: mm.State, goal_condition: mm.GroundConjunctiveCondition) -> bool:
        return state in self.dead_end_states


class ActionBeamRewardFunction(RewardFunction):
    def __init__(self, dead_end_states: set[FakeBeamState] | None = None) -> None:
        self.dead_end_states = dead_end_states or set()

    def __call__(self,
                 current_state: mm.State,
                 action: mm.GroundAction,
                 successor_state: mm.State,
                 goal_condition: mm.GroundConjunctiveCondition) -> float:
        return cast(Any, action).reward

    def is_dead_end(self, state: mm.State, goal_condition: mm.GroundConjunctiveCondition) -> bool:
        return state in self.dead_end_states


def test_model_wrapper():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    model = RGNNWrapper(domain)
    current_state = problem.initial_state
    goal_condition = problem.goal
    output = model.forward([(current_state, goal_condition)])
    assert output is not None
    assert isinstance(output, list)
    assert len(output) > 0
    assert isinstance(output[0][0], torch.Tensor)
    assert isinstance(output[0][1], list)
    assert len(output[0][1]) > 0


def test_dqn_loss():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    model = RGNNWrapper(domain)
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loss = DQNOptimization(model, optimizer, lr_scheduler, model, 0.999, 10.0, True)
    transitions: list[Transition] = []
    current_state = problem.initial_state
    reward_function = ConstantRewardFunction(-1.0)
    for selected_action in current_state.applicable_actions():
        successor_state = selected_action.apply(current_state)
        goal_condition = problem.goal
        reward = reward_function(current_state, selected_action, successor_state, goal_condition)
        transitions.append(Transition(current_state, successor_state, selected_action, -1.0, -1.0, reward, 0.0, reward_function, goal_condition, False))
    losses = loss(transitions, torch.ones(len(transitions)))
    assert losses is not None
    assert len(losses) == len(transitions)


def test_sac_loss():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    policy_model = RGNNWrapper(domain)
    qvalue_model_1 = RGNNWrapper(domain)
    qvalue_model_2 = RGNNWrapper(domain)
    qvalue_target_1 = RGNNWrapper(domain)
    qvalue_target_2 = RGNNWrapper(domain)
    policy_optimizer = torch.optim.Adam(policy_model.parameters())
    qvalue_optimizer_1 = torch.optim.Adam(qvalue_model_1.parameters())
    qvalue_optimizer_2 = torch.optim.Adam(qvalue_model_2.parameters())
    policy_lr_scheduler = torch.optim.lr_scheduler.StepLR(policy_optimizer, step_size=10, gamma=0.9)
    qvalue_lr_scheduler_1 = torch.optim.lr_scheduler.StepLR(qvalue_optimizer_1, step_size=10, gamma=0.9)
    qvalue_lr_scheduler_2 = torch.optim.lr_scheduler.StepLR(qvalue_optimizer_2, step_size=10, gamma=0.9)
    discount_factor = 0.999
    polyak_factor = 0.005
    entropy_temperature = 1.0
    entropy_lr = 0.0003
    loss = DiscreteSoftActorCriticOptimization(policy_model,
                                               policy_optimizer,
                                               policy_lr_scheduler,
                                               qvalue_target_1,
                                               qvalue_model_1,
                                               qvalue_optimizer_1,
                                               qvalue_lr_scheduler_1,
                                               qvalue_target_2,
                                               qvalue_model_2,
                                               qvalue_optimizer_2,
                                               qvalue_lr_scheduler_2,
                                               discount_factor,
                                               polyak_factor,
                                               entropy_temperature,
                                               entropy_lr)
    transitions: list[Transition] = []
    current_state = problem.initial_state
    reward_function = ConstantRewardFunction(-1.0)
    for selected_action in current_state.applicable_actions():
        successor_state = selected_action.apply(current_state)
        goal_condition = problem.goal
        reward = reward_function(current_state, selected_action, successor_state, goal_condition)
        transitions.append(Transition(current_state, successor_state, selected_action, -1.0, -1.0, reward, 0.0, reward_function, goal_condition, False))
    losses = loss(transitions, torch.ones(len(transitions)))
    assert losses is not None
    assert len(losses) == len(transitions)


def test_sac_loss_sets_training_modes():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    policy_model = RGNNWrapper(domain)
    qvalue_model_1 = RGNNWrapper(domain)
    qvalue_model_2 = RGNNWrapper(domain)
    qvalue_target_1 = RGNNWrapper(domain)
    qvalue_target_2 = RGNNWrapper(domain)
    policy_optimizer = torch.optim.Adam(policy_model.parameters())
    qvalue_optimizer_1 = torch.optim.Adam(qvalue_model_1.parameters())
    qvalue_optimizer_2 = torch.optim.Adam(qvalue_model_2.parameters())
    policy_lr_scheduler = torch.optim.lr_scheduler.StepLR(policy_optimizer, step_size=10, gamma=0.9)
    qvalue_lr_scheduler_1 = torch.optim.lr_scheduler.StepLR(qvalue_optimizer_1, step_size=10, gamma=0.9)
    qvalue_lr_scheduler_2 = torch.optim.lr_scheduler.StepLR(qvalue_optimizer_2, step_size=10, gamma=0.9)
    loss = DiscreteSoftActorCriticOptimization(policy_model,
                                               policy_optimizer,
                                               policy_lr_scheduler,
                                               qvalue_target_1,
                                               qvalue_model_1,
                                               qvalue_optimizer_1,
                                               qvalue_lr_scheduler_1,
                                               qvalue_target_2,
                                               qvalue_model_2,
                                               qvalue_optimizer_2,
                                               qvalue_lr_scheduler_2,
                                               0.999,
                                               0.005,
                                               1.0,
                                               0.0003)
    policy_model.eval()
    qvalue_model_1.eval()
    qvalue_model_2.eval()
    qvalue_target_1.train()
    qvalue_target_2.train()
    transitions: list[Transition] = []
    current_state = problem.initial_state
    reward_function = ConstantRewardFunction(-1.0)
    for selected_action in current_state.applicable_actions():
        successor_state = selected_action.apply(current_state)
        goal_condition = problem.goal
        reward = reward_function(current_state, selected_action, successor_state, goal_condition)
        transitions.append(Transition(current_state, successor_state, selected_action, -1.0, -1.0, reward, 0.0, reward_function, goal_condition, False))
    loss(transitions, torch.ones(len(transitions)))
    assert policy_model.training
    assert qvalue_model_1.training
    assert qvalue_model_2.training
    assert not qvalue_target_1.training
    assert not qvalue_target_2.training


def test_sac_entropy_loss_uses_exact_temperature_objective():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    policy_model = RGNNWrapper(domain)
    qvalue_model_1 = RGNNWrapper(domain)
    qvalue_model_2 = RGNNWrapper(domain)
    qvalue_target_1 = RGNNWrapper(domain)
    qvalue_target_2 = RGNNWrapper(domain)
    policy_optimizer = torch.optim.Adam(policy_model.parameters())
    qvalue_optimizer_1 = torch.optim.Adam(qvalue_model_1.parameters())
    qvalue_optimizer_2 = torch.optim.Adam(qvalue_model_2.parameters())
    policy_lr_scheduler = torch.optim.lr_scheduler.StepLR(policy_optimizer, step_size=10, gamma=0.9)
    qvalue_lr_scheduler_1 = torch.optim.lr_scheduler.StepLR(qvalue_optimizer_1, step_size=10, gamma=0.9)
    qvalue_lr_scheduler_2 = torch.optim.lr_scheduler.StepLR(qvalue_optimizer_2, step_size=10, gamma=0.9)
    loss = DiscreteSoftActorCriticOptimization(policy_model,
                                               policy_optimizer,
                                               policy_lr_scheduler,
                                               qvalue_target_1,
                                               qvalue_model_1,
                                               qvalue_optimizer_1,
                                               qvalue_lr_scheduler_1,
                                               qvalue_target_2,
                                               qvalue_model_2,
                                               qvalue_optimizer_2,
                                               qvalue_lr_scheduler_2,
                                               0.999,
                                               0.005,
                                               0.5,
                                               0.0003)
    current_state = problem.initial_state
    actions = current_state.applicable_actions()
    num_actions = len(actions)
    logits = torch.zeros(num_actions, dtype=torch.float)
    loss.log_entropy_alpha.data.fill_(2.0)
    entropy_losses = loss._compute_entropy_loss([(logits, actions)])
    expected_entropy = math.log(num_actions)
    expected_target = 0.5 * math.log(num_actions)
    expected_loss = 2.0 * (expected_entropy - expected_target)
    assert torch.isclose(entropy_losses[0], torch.tensor(expected_loss, dtype=entropy_losses.dtype), atol=1e-6)


def test_td3_loss():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    policy_model = RGNNWrapper(domain)
    policy_target = RGNNWrapper(domain)
    qvalue_model_1 = RGNNWrapper(domain)
    qvalue_model_2 = RGNNWrapper(domain)
    qvalue_target_1 = RGNNWrapper(domain)
    qvalue_target_2 = RGNNWrapper(domain)
    policy_optimizer = torch.optim.Adam(policy_model.parameters())
    qvalue_optimizer_1 = torch.optim.Adam(qvalue_model_1.parameters())
    qvalue_optimizer_2 = torch.optim.Adam(qvalue_model_2.parameters())
    policy_lr_scheduler = torch.optim.lr_scheduler.StepLR(policy_optimizer, step_size=10, gamma=0.9)
    qvalue_lr_scheduler_1 = torch.optim.lr_scheduler.StepLR(qvalue_optimizer_1, step_size=10, gamma=0.9)
    qvalue_lr_scheduler_2 = torch.optim.lr_scheduler.StepLR(qvalue_optimizer_2, step_size=10, gamma=0.9)
    loss = DiscreteTD3Optimization(policy_model,
                                   policy_optimizer,
                                   policy_lr_scheduler,
                                   policy_target,
                                   qvalue_target_1,
                                   qvalue_model_1,
                                   qvalue_optimizer_1,
                                   qvalue_lr_scheduler_1,
                                   qvalue_target_2,
                                   qvalue_model_2,
                                   qvalue_optimizer_2,
                                   qvalue_lr_scheduler_2,
                                   0.999,
                                   0.005,
                                   2)
    transitions: list[Transition] = []
    current_state = problem.initial_state
    reward_function = ConstantRewardFunction(-1.0)
    for selected_action in current_state.applicable_actions():
        successor_state = selected_action.apply(current_state)
        goal_condition = problem.goal
        reward = reward_function(current_state, selected_action, successor_state, goal_condition)
        transitions.append(Transition(current_state, successor_state, selected_action, -1.0, -1.0, reward, 0.0, reward_function, goal_condition, False))
    losses = loss(transitions, torch.ones(len(transitions)))
    assert losses is not None
    assert len(losses) == len(transitions)


def test_td3_loss_with_gumbel_softmax():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    policy_model = RGNNWrapper(domain)
    policy_target = RGNNWrapper(domain)
    qvalue_model_1 = RGNNWrapper(domain)
    qvalue_model_2 = RGNNWrapper(domain)
    qvalue_target_1 = RGNNWrapper(domain)
    qvalue_target_2 = RGNNWrapper(domain)
    policy_optimizer = torch.optim.Adam(policy_model.parameters())
    qvalue_optimizer_1 = torch.optim.Adam(qvalue_model_1.parameters())
    qvalue_optimizer_2 = torch.optim.Adam(qvalue_model_2.parameters())
    policy_lr_scheduler = torch.optim.lr_scheduler.StepLR(policy_optimizer, step_size=10, gamma=0.9)
    qvalue_lr_scheduler_1 = torch.optim.lr_scheduler.StepLR(qvalue_optimizer_1, step_size=10, gamma=0.9)
    qvalue_lr_scheduler_2 = torch.optim.lr_scheduler.StepLR(qvalue_optimizer_2, step_size=10, gamma=0.9)
    torch.manual_seed(0)
    loss = DiscreteTD3Optimization(policy_model,
                                   policy_optimizer,
                                   policy_lr_scheduler,
                                   policy_target,
                                   qvalue_target_1,
                                   qvalue_model_1,
                                   qvalue_optimizer_1,
                                   qvalue_lr_scheduler_1,
                                   qvalue_target_2,
                                   qvalue_model_2,
                                   qvalue_optimizer_2,
                                   qvalue_lr_scheduler_2,
                                   0.999,
                                   0.005,
                                   2,
                                   True)
    transitions: list[Transition] = []
    current_state = problem.initial_state
    reward_function = ConstantRewardFunction(-1.0)
    for selected_action in current_state.applicable_actions():
        successor_state = selected_action.apply(current_state)
        goal_condition = problem.goal
        reward = reward_function(current_state, selected_action, successor_state, goal_condition)
        transitions.append(Transition(current_state, successor_state, selected_action, -1.0, -1.0, reward, 0.0, reward_function, goal_condition, False))
    losses = loss(transitions, torch.ones(len(transitions)))
    assert losses is not None
    assert len(losses) == len(transitions)


def test_td3_loss_sets_training_modes():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    policy_model = RGNNWrapper(domain)
    policy_target = RGNNWrapper(domain)
    qvalue_model_1 = RGNNWrapper(domain)
    qvalue_model_2 = RGNNWrapper(domain)
    qvalue_target_1 = RGNNWrapper(domain)
    qvalue_target_2 = RGNNWrapper(domain)
    policy_optimizer = torch.optim.Adam(policy_model.parameters())
    qvalue_optimizer_1 = torch.optim.Adam(qvalue_model_1.parameters())
    qvalue_optimizer_2 = torch.optim.Adam(qvalue_model_2.parameters())
    policy_lr_scheduler = torch.optim.lr_scheduler.StepLR(policy_optimizer, step_size=10, gamma=0.9)
    qvalue_lr_scheduler_1 = torch.optim.lr_scheduler.StepLR(qvalue_optimizer_1, step_size=10, gamma=0.9)
    qvalue_lr_scheduler_2 = torch.optim.lr_scheduler.StepLR(qvalue_optimizer_2, step_size=10, gamma=0.9)
    loss = DiscreteTD3Optimization(policy_model,
                                   policy_optimizer,
                                   policy_lr_scheduler,
                                   policy_target,
                                   qvalue_target_1,
                                   qvalue_model_1,
                                   qvalue_optimizer_1,
                                   qvalue_lr_scheduler_1,
                                   qvalue_target_2,
                                   qvalue_model_2,
                                   qvalue_optimizer_2,
                                   qvalue_lr_scheduler_2,
                                   0.999,
                                   0.005,
                                   1)
    policy_model.eval()
    qvalue_model_1.eval()
    qvalue_model_2.eval()
    policy_target.train()
    qvalue_target_1.train()
    qvalue_target_2.train()
    transitions: list[Transition] = []
    current_state = problem.initial_state
    reward_function = ConstantRewardFunction(-1.0)
    for selected_action in current_state.applicable_actions():
        successor_state = selected_action.apply(current_state)
        goal_condition = problem.goal
        reward = reward_function(current_state, selected_action, successor_state, goal_condition)
        transitions.append(Transition(current_state, successor_state, selected_action, -1.0, -1.0, reward, 0.0, reward_function, goal_condition, False))
    loss(transitions, torch.ones(len(transitions)))
    assert policy_model.training
    assert qvalue_model_1.training
    assert qvalue_model_2.training
    assert not policy_target.training
    assert not qvalue_target_1.training
    assert not qvalue_target_2.training


def test_iqn_loss():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    model = DummyIQNWrapper([float(i) for i in range(16)])
    target_model = DummyIQNWrapper([float(i) for i in range(16)])
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loss = IQNOptimization(model, optimizer, lr_scheduler, target_model, 0.999, use_bounds=False)
    transitions: list[Transition] = []
    current_state = problem.initial_state
    reward_function = ConstantRewardFunction(-1.0)
    for selected_action in current_state.applicable_actions():
        successor_state = selected_action.apply(current_state)
        goal_condition = problem.goal
        reward = reward_function(current_state, selected_action, successor_state, goal_condition)
        transitions.append(Transition(current_state, successor_state, selected_action, -1.0, -1.0, reward, 0.0, reward_function, goal_condition, False))
    losses = loss(transitions, torch.ones(len(transitions)))
    assert losses is not None
    assert len(losses) == len(transitions)


def test_iqn_loss_sets_training_modes():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    model = DummyIQNWrapper([float(i) for i in range(16)])
    target_model = DummyIQNWrapper([float(i) for i in range(16)])
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loss = IQNOptimization(model, optimizer, lr_scheduler, target_model, 0.999, use_bounds=False)
    model.eval()
    target_model.train()
    transitions: list[Transition] = []
    current_state = problem.initial_state
    reward_function = ConstantRewardFunction(-1.0)
    for selected_action in current_state.applicable_actions():
        successor_state = selected_action.apply(current_state)
        goal_condition = problem.goal
        reward = reward_function(current_state, selected_action, successor_state, goal_condition)
        transitions.append(Transition(current_state, successor_state, selected_action, -1.0, -1.0, reward, 0.0, reward_function, goal_condition, False))
    loss(transitions, torch.ones(len(transitions)))
    assert model.training
    assert not target_model.training


def test_iqn_target_uses_online_action_selection():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    model = DummyIQNWrapper([10.0, 0.0] + [0.0 for _ in range(14)])
    target_model = DummyIQNWrapper([1.0, 50.0] + [0.0 for _ in range(14)])
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loss = IQNOptimization(model,
                           optimizer,
                           lr_scheduler,
                           target_model,
                           1.0,
                           num_quantiles=4,
                           num_target_quantiles=4,
                           num_selection_quantiles=4,
                           use_bounds=False)
    current_state = problem.initial_state
    reward_function = ConstantRewardFunction(-1.0)
    transition: Transition | None = None
    for selected_action in current_state.applicable_actions():
        successor_state = selected_action.apply(current_state)
        if len(successor_state.applicable_actions()) > 1:
            goal_condition = problem.goal
            reward = reward_function(current_state, selected_action, successor_state, goal_condition)
            transition = Transition(current_state, successor_state, selected_action, -1.0, -1.0, reward, 0.0, reward_function, goal_condition, False)
            break
    assert transition is not None
    device = next(model.parameters()).device
    target_distributions = loss._compute_target_distributions([transition], device)
    expected_value = transition.immediate_reward + target_model.action_values[0].item()
    expected_distribution = torch.full((loss.num_target_quantiles,), expected_value, device=device)
    assert torch.allclose(target_distributions[0], expected_distribution)


def test_dead_end_targets_do_not_bootstrap():
    current_state = FakeBeamState('current')
    proven_dead_end = FakeBeamState('proven_dead_end')
    stay = FakeBeamAction('stay', proven_dead_end)
    selected_action = FakeBeamAction('to_proven_dead_end', proven_dead_end)
    current_state.set_actions([selected_action])
    proven_dead_end.set_actions([stay])
    goal_condition = FakeBeamGoalCondition()
    reward_function = DummyBeamRewardFunction(0.0, {proven_dead_end})
    transition = Transition(cast(mm.State, current_state),
                            cast(mm.State, proven_dead_end),
                            cast(mm.GroundAction, selected_action),
                            100.0,
                            100.0,
                            0.0,
                            0.0,
                            reward_function,
                            cast(mm.GroundConjunctiveCondition, goal_condition),
                            False)

    assert transition.enters_dead_end
    assert transition.is_terminal
    assert transition.immediate_reward == RewardFunction.get_dead_end_reward()

    scalar_model = FixedBeamModel([100.0])
    scalar_optimizer = torch.optim.Adam(scalar_model.parameters())
    scalar_scheduler = torch.optim.lr_scheduler.ConstantLR(scalar_optimizer)
    dqn = DQNOptimization(scalar_model, scalar_optimizer, scalar_scheduler, scalar_model, 0.9, 1.0)
    scalar_device = next(scalar_model.parameters()).device
    dqn_target = dqn._compute_targets([transition], RewardFunction.get_dead_end_reward(), scalar_device)
    assert torch.equal(dqn_target, torch.tensor([RewardFunction.get_dead_end_reward()], device=scalar_device))

    iqn_model = DummyIQNWrapper([100.0])
    iqn_target_model = DummyIQNWrapper([100.0])
    iqn_optimizer = torch.optim.Adam(iqn_model.parameters())
    iqn_scheduler = torch.optim.lr_scheduler.ConstantLR(iqn_optimizer)
    iqn = IQNOptimization(iqn_model,
                          iqn_optimizer,
                          iqn_scheduler,
                          iqn_target_model,
                          0.9,
                          num_quantiles=4,
                          num_target_quantiles=4,
                          num_selection_quantiles=4,
                          use_bounds=False)
    iqn_device = next(iqn_model.parameters()).device
    iqn_target = iqn._compute_target_distributions([transition], iqn_device)[0]
    assert torch.equal(iqn_target, torch.full((4,), RewardFunction.get_dead_end_reward(), device=iqn_device))

    sac_policy = FixedBeamModel([100.0])
    sac_qvalue_1 = FixedBeamModel([100.0])
    sac_qvalue_2 = FixedBeamModel([100.0])
    sac_policy_optimizer = torch.optim.Adam(sac_policy.parameters())
    sac_qvalue_optimizer_1 = torch.optim.Adam(sac_qvalue_1.parameters())
    sac_qvalue_optimizer_2 = torch.optim.Adam(sac_qvalue_2.parameters())
    sac = DiscreteSoftActorCriticOptimization(
        sac_policy,
        sac_policy_optimizer,
        torch.optim.lr_scheduler.ConstantLR(sac_policy_optimizer),
        FixedBeamModel([100.0]),
        sac_qvalue_1,
        sac_qvalue_optimizer_1,
        torch.optim.lr_scheduler.ConstantLR(sac_qvalue_optimizer_1),
        FixedBeamModel([100.0]),
        sac_qvalue_2,
        sac_qvalue_optimizer_2,
        torch.optim.lr_scheduler.ConstantLR(sac_qvalue_optimizer_2),
        0.9,
    )
    sac_target = sac._compute_qvalue_targets([transition])
    assert torch.equal(sac_target, torch.tensor([RewardFunction.get_dead_end_reward()]))

    td3_policy = FixedBeamModel([100.0])
    td3_qvalue_1 = FixedBeamModel([100.0])
    td3_qvalue_2 = FixedBeamModel([100.0])
    td3_policy_optimizer = torch.optim.Adam(td3_policy.parameters())
    td3_qvalue_optimizer_1 = torch.optim.Adam(td3_qvalue_1.parameters())
    td3_qvalue_optimizer_2 = torch.optim.Adam(td3_qvalue_2.parameters())
    td3 = DiscreteTD3Optimization(
        td3_policy,
        td3_policy_optimizer,
        torch.optim.lr_scheduler.ConstantLR(td3_policy_optimizer),
        FixedBeamModel([100.0]),
        FixedBeamModel([100.0]),
        td3_qvalue_1,
        td3_qvalue_optimizer_1,
        torch.optim.lr_scheduler.ConstantLR(td3_qvalue_optimizer_1),
        FixedBeamModel([100.0]),
        td3_qvalue_2,
        td3_qvalue_optimizer_2,
        torch.optim.lr_scheduler.ConstantLR(td3_qvalue_optimizer_2),
        0.9,
    )
    td3_target = td3._compute_qvalue_targets([transition])
    assert torch.equal(td3_target, torch.tensor([RewardFunction.get_dead_end_reward()]))


def test_trajectory_rejects_misaligned_value_sequences_when_empty():
    domain = mm.Domain.from_file(DATA_DIR / 'gripper' / 'domain.pddl')
    problem = mm.Problem.from_file(domain, DATA_DIR / 'gripper' / 'problem.pddl')
    state = problem.initial_state
    goal_condition = problem.goal
    reward_function = ConstantRewardFunction(-1.0)

    with pytest.raises(AssertionError, match="value sequence"):
        Trajectory([state], [], [1.0], [], [], reward_function, goal_condition)
    with pytest.raises(AssertionError, match="Q-value sequence"):
        Trajectory([state], [], [], [1.0], [], reward_function, goal_condition)


def test_zero_transition_goal_trajectory_validates():
    domain = mm.Domain.from_file(DATA_DIR / 'gripper' / 'domain.pddl')
    problem = mm.Problem.from_file(domain, DATA_DIR / 'gripper' / 'problem.pddl')
    state = problem.initial_state
    for action_name in (
        '(pick ball2 rooma left)',
        '(move rooma roomb)',
        '(drop ball2 roomb left)',
    ):
        action = next(action for action in state.applicable_actions() if str(action) == action_name)
        state = action.apply(state)
    goal_condition = problem.goal
    assert goal_condition.holds(state)

    trajectory = Trajectory(
        [state],
        [],
        [],
        [],
        [],
        ConstantRewardFunction(-1.0),
        goal_condition,
    )

    assert trajectory.is_solution()
    trajectory.validate()
    assert TDErrorCriteria(False).evaluate([trajectory]) == 0


def test_trajectory_rejects_suffix_after_explicit_dead_end():
    domain = mm.Domain.from_file(DATA_DIR / 'gripper' / 'domain.pddl')
    problem = mm.Problem.from_file(domain, DATA_DIR / 'gripper' / 'problem.pddl')
    start = problem.initial_state
    to_dead_end = next(action for action in start.applicable_actions()
                       if str(action) == '(pick ball2 rooma left)')
    dead_end = to_dead_end.apply(start)
    after_dead_end = next(action for action in dead_end.applicable_actions()
                          if str(action) == '(move rooma roomb)')
    tail = after_dead_end.apply(dead_end)
    reward_function = DummyBeamRewardFunction(-1.0, cast(Any, {dead_end}))

    with pytest.raises(AssertionError, match="cannot continue"):
        Trajectory(
            [start, dead_end, tail],
            [to_dead_end, after_dead_end],
            [0.0, 0.0],
            [0.0, 0.0],
            [-1.0, -1.0],
            reward_function,
            problem.goal,
        )


def test_iw_subtrajectory_does_not_continue_after_explicit_dead_end(monkeypatch: pytest.MonkeyPatch):
    domain = mm.Domain.from_file(DATA_DIR / 'gripper' / 'domain.pddl')
    problem = mm.Problem.from_file(domain, DATA_DIR / 'gripper' / 'problem.pddl')
    start = problem.initial_state
    pick = next(action for action in start.applicable_actions()
                if str(action) == '(pick ball2 rooma left)')
    dead_end = pick.apply(start)
    move = next(action for action in dead_end.applicable_actions()
                if str(action) == '(move rooma roomb)')
    after_dead_end = move.apply(dead_end)
    drop = next(action for action in after_dead_end.applicable_actions()
                if str(action) == '(drop ball2 roomb left)')
    goal = drop.apply(after_dead_end)

    def fake_iw(*args: Any,
                on_generate: Callable[..., None] | None = None,
                on_discover: Callable[..., None] | None = None,
                **kwargs: Any) -> None:
        assert on_generate is not None
        assert on_discover is not None
        for transition in (
            FakeSearchTransition(cast(Any, start), cast(Any, pick), cast(Any, dead_end)),
            FakeSearchTransition(cast(Any, dead_end), cast(Any, move), cast(Any, after_dead_end)),
            FakeSearchTransition(cast(Any, after_dead_end), cast(Any, drop), cast(Any, goal)),
        ):
            on_generate(transition)
            on_discover(transition)

    monkeypatch.setattr(mm, 'iw', fake_iw)
    reward_function = DummyBeamRewardFunction(-1.0, cast(Any, {dead_end}))
    sampler = IWSubtrajectorySampler(reward_function, 1)

    assert sampler.sample(start, problem.goal) is None


def test_iw_subtrajectory_preserves_live_route_to_shared_state(monkeypatch: pytest.MonkeyPatch):
    start = FakeBeamState('start')
    dead_end = FakeBeamState('dead_end')
    live = FakeBeamState('live')
    shared = FakeBeamState('shared')
    to_dead_end = FakeBeamAction('to_dead_end', dead_end)
    dead_to_shared = FakeBeamAction('dead_to_shared', shared)
    to_live = FakeBeamAction('to_live', live)
    live_to_shared = FakeBeamAction('live_to_shared', shared)
    start.set_actions([to_dead_end, to_live])
    dead_end.set_actions([dead_to_shared])
    live.set_actions([live_to_shared])
    shared.set_actions([FakeBeamAction('stay', shared)])

    class EmptyGoalCondition(FakeBeamGoalCondition):
        def __iter__(self):
            return iter(())

    goal_condition = EmptyGoalCondition()

    def fake_iw(*args: Any,
                on_generate: Callable[..., None] | None = None,
                on_discover: Callable[..., None] | None = None,
                **kwargs: Any) -> None:
        assert on_generate is not None
        assert on_discover is not None
        generated_edges = [
            (start, to_dead_end, dead_end, True),
            (dead_end, dead_to_shared, shared, True),
            (start, to_live, live, True),
            (live, live_to_shared, shared, False),
        ]
        for current_state, action, successor_state, is_new in generated_edges:
            transition = FakeSearchTransition(current_state, action, successor_state)
            on_generate(transition)
            if is_new:
                on_discover(transition)

    captured: dict[str, Any] = {}

    def capture_trajectory(state_sequence: list[FakeBeamState], action_sequence: list[FakeBeamAction], *args: Any) -> Any:
        captured['states'] = state_sequence
        captured['actions'] = action_sequence
        return captured

    monkeypatch.setattr(mm, 'iw', fake_iw)
    monkeypatch.setattr(subtrajectory_sampling_module, 'Trajectory', capture_trajectory)
    sampler = IWSubtrajectorySampler(ActionBeamRewardFunction({dead_end}), 1)
    monkeypatch.setattr(sampler, '_get_goal_improvement',
                        lambda candidate, achieved, unachieved: int(candidate == shared))

    result = sampler.sample(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))

    assert result is captured
    assert captured['states'] == [start, live, shared]
    assert captured['actions'] == [to_live, live_to_shared]


def test_td_error_uses_immediate_reward_as_terminal_target():
    domain = mm.Domain.from_file(DATA_DIR / 'gripper' / 'domain.pddl')
    problem = mm.Problem.from_file(domain, DATA_DIR / 'gripper' / 'problem.pddl')
    start = problem.initial_state
    action = next(action for action in start.applicable_actions()
                  if str(action) == '(pick ball2 rooma left)')
    dead_end = action.apply(start)
    reward_function = DummyBeamRewardFunction(0.0, cast(Any, {dead_end}))
    trajectory = Trajectory(
        [start, dead_end],
        [action],
        [RewardFunction.get_dead_end_reward()],
        [RewardFunction.get_dead_end_reward()],
        [0.0],
        reward_function,
        problem.goal,
    )

    assert trajectory[0].is_terminal
    assert TDErrorCriteria(False).evaluate([trajectory]) == 0


def test_actionless_dead_end_transition_normalizes_reward():
    current_state = FakeBeamState('current')
    dead_end = FakeBeamState('dead_end')
    selected_action = FakeBeamAction('to_dead_end', dead_end)
    current_state.set_actions([selected_action])
    goal_condition = FakeBeamGoalCondition()
    reward_function = DummyBeamRewardFunction(-1.0)
    transition = Transition(cast(mm.State, current_state),
                            cast(mm.State, dead_end),
                            cast(mm.GroundAction, selected_action),
                            0.0,
                            0.0,
                            -1.0,
                            0.0,
                            reward_function,
                            cast(mm.GroundConjunctiveCondition, goal_condition),
                            False)

    assert transition.enters_dead_end
    assert transition.is_terminal
    assert transition.immediate_reward == RewardFunction.get_dead_end_reward()


def test_dead_end_cost_does_not_imply_dead_end_state():
    current_state = FakeBeamState('current')
    live = FakeBeamState('live')
    live.set_actions([FakeBeamAction('stay', live)])
    selected_action = FakeBeamAction('to_live', live)
    current_state.set_actions([selected_action])
    goal_condition = FakeBeamGoalCondition()
    reward_function = DummyBeamRewardFunction(RewardFunction.get_dead_end_reward())
    transition = Transition(cast(mm.State, current_state),
                            cast(mm.State, live),
                            cast(mm.GroundAction, selected_action),
                            0.0,
                            0.0,
                            RewardFunction.get_dead_end_reward(),
                            0.0,
                            reward_function,
                            cast(mm.GroundConjunctiveCondition, goal_condition),
                            False)

    assert not transition.enters_dead_end
    assert not transition.is_terminal
    assert transition.immediate_reward == RewardFunction.get_dead_end_reward()


def test_sum_reward_function_propagates_dead_end_proof():
    dead_end = FakeBeamState('dead_end')
    goal_condition = FakeBeamGoalCondition()
    reward_function = SumRewardFunction([
        DummyBeamRewardFunction(100.0, {dead_end}),
        DummyBeamRewardFunction(20_000.0),
    ])

    assert reward_function.is_dead_end(cast(mm.State, dead_end), cast(mm.GroundConjunctiveCondition, goal_condition))


def test_beam_search_finalize_handles_initial_dead_end():
    start = FakeBeamState('dead')
    start.set_actions([])
    goal_condition = FakeBeamGoalCondition()
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(-1.0), 2)
    trajectory_state = TrajectoryState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    search_state = sampler.SearchState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    sampler._finalize_state(trajectory_state, search_state)
    assert trajectory_state.state_sequence == [start]
    assert trajectory_state.action_sequence == []
    assert trajectory_state.reward_sequence == []


def test_beam_search_sample_handles_initial_dead_end():
    domain_path = DATA_DIR / 'spanner' / 'domain.pddl'
    problem_path = DATA_DIR / 'spanner' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    goal_condition = problem.goal
    dead_end_state = problem.initial_state
    while len(dead_end_state.applicable_actions()) > 0:
        walk_actions = [action for action in dead_end_state.applicable_actions() if str(action).startswith('(walk ')]
        assert len(walk_actions) == 1
        dead_end_state = walk_actions[0].apply(dead_end_state)

    assert not goal_condition.holds(dead_end_state)
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), ConstantRewardFunction(-1.0), 2)
    trajectory = sampler.sample([(dead_end_state, goal_condition)], 5)[0]

    assert len(trajectory) == 0
    assert trajectory.start_state == dead_end_state
    assert trajectory.final_state == dead_end_state
    assert trajectory.is_unsolvable()
    trajectory.validate()


@pytest.mark.parametrize('max_beam_size', [0, -1])
def test_beam_search_rejects_non_positive_beam_size(max_beam_size: int):
    with pytest.raises(AssertionError):
        BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(-1.0), max_beam_size)


def test_beam_search_rejects_non_integer_beam_size():
    with pytest.raises(AssertionError):
        BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(-1.0), 1.5)  # type: ignore


@pytest.mark.parametrize('horizon', [0, -1, 1.5])
def test_beam_search_rejects_invalid_horizon(horizon: int | float):
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(-1.0), 1)
    with pytest.raises(AssertionError):
        sampler.sample([], horizon)  # type: ignore[arg-type]


def test_beam_search_checks_horizon_before_expansion():
    start = FakeBeamState('start')
    successor = FakeBeamState('successor')
    successor.set_actions([FakeBeamAction('stay', successor)])
    action = FakeBeamAction('advance', successor)
    start.set_actions([action])
    goal_condition = FakeBeamGoalCondition()
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(0.0), 1)
    trajectory_state = TrajectoryState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    search_state = sampler.SearchState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))

    sampler._internal_sample([trajectory_state], [search_state], [0])

    assert trajectory_state.done
    assert search_state.depth == 0
    assert search_state.beam_list == [start]
    assert successor not in search_state.transition_map


def test_beam_search_rejects_misaligned_q_values_and_actions():
    start = FakeBeamState('start')
    successor = FakeBeamState('successor')
    action = FakeBeamAction('advance', successor)
    start.set_actions([action])
    goal_condition = FakeBeamGoalCondition()
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(0.0), 1)
    trajectory_state = TrajectoryState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    search_state = sampler.SearchState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))

    with pytest.raises(AssertionError, match="equal lengths"):
        sampler._beam_step(trajectory_state,
                           search_state,
                           max_depth=1,
                           beam_successor_values=[(torch.tensor([1.0, 2.0]), [cast(mm.GroundAction, action)])])


def test_beam_search_rejects_incomplete_model_batch():
    starts = [FakeBeamState('start_1'), FakeBeamState('start_2')]
    goal_condition = FakeBeamGoalCondition()
    for idx, start in enumerate(starts):
        successor = FakeBeamState(f'successor_{idx}')
        successor.set_actions([FakeBeamAction('stay', successor)])
        start.set_actions([FakeBeamAction('advance', successor)])
    sampler = BeamSearchTrajectorySampler(ShortBatchBeamModel([1.0]), DummyBeamRewardFunction(0.0), 1)
    state_goals = [(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition)) for start in starts]
    trajectory_states, search_states = sampler._initialize(state_goals)

    with pytest.raises(AssertionError, match="one result per input state"):
        sampler._internal_sample(trajectory_states, search_states, [1, 1])


def test_beam_search_ranks_goal_successor_by_q_value():
    start = FakeBeamState('start')
    goal = FakeBeamState('goal')
    live = FakeBeamState('live')
    live.set_actions([FakeBeamAction('stay', live)])
    to_goal = FakeBeamAction('to_goal', goal)
    to_live = FakeBeamAction('to_live', live)
    start.set_actions([to_goal, to_live])
    goal_condition = FakeBeamGoalCondition({goal})
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(0.0), 1)
    trajectory_state = TrajectoryState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    search_state = sampler.SearchState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))

    sampler._beam_step(trajectory_state,
                       search_state,
                       max_depth=5,
                       beam_successor_values=[(torch.tensor([-10.0, 10.0]), [cast(mm.GroundAction, to_goal), cast(mm.GroundAction, to_live)])])

    assert not trajectory_state.solved
    assert search_state.beam_list == [live]


def test_beam_search_returns_goal_once_it_enters_beam():
    start = FakeBeamState('start')
    goal = FakeBeamState('goal')
    live = FakeBeamState('live')
    live.set_actions([FakeBeamAction('stay', live)])
    to_goal = FakeBeamAction('to_goal', goal)
    to_live = FakeBeamAction('to_live', live)
    start.set_actions([to_goal, to_live])
    goal_condition = FakeBeamGoalCondition({goal})
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(0.0), 2)
    trajectory_state = TrajectoryState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    search_state = sampler.SearchState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))

    sampler._beam_step(trajectory_state,
                       search_state,
                       max_depth=5,
                       beam_successor_values=[(torch.tensor([-10.0, 10.0]), [cast(mm.GroundAction, to_goal), cast(mm.GroundAction, to_live)])])
    sampler._finalize_state(trajectory_state, search_state)

    assert trajectory_state.solved
    assert search_state.beam_list == [live, goal]
    assert trajectory_state.state_sequence == [start, goal]


def test_beam_search_follows_high_q_dead_end():
    start = FakeBeamState('start')
    dead_end = FakeBeamState('dead_end')
    live = FakeBeamState('live')
    live.set_actions([FakeBeamAction('stay', live)])
    to_dead_end = FakeBeamAction('to_dead_end', dead_end)
    to_live = FakeBeamAction('to_live', live)
    start.set_actions([to_dead_end, to_live])
    goal_condition = FakeBeamGoalCondition()
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(0.0), 1)
    trajectory_state = TrajectoryState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    search_state = sampler.SearchState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))

    sampler._beam_step(trajectory_state,
                       search_state,
                       max_depth=5,
                       beam_successor_values=[(torch.tensor([10.0, -10.0]), [cast(mm.GroundAction, to_dead_end), cast(mm.GroundAction, to_live)])])

    sampler._finalize_state(trajectory_state, search_state)

    assert trajectory_state.done
    assert search_state.beam_list == [dead_end]
    assert trajectory_state.state_sequence == [start, dead_end]
    assert trajectory_state.reward_sequence == [RewardFunction.get_dead_end_reward()]


def test_beam_search_forgets_dead_end_while_live_beam_remains():
    start = FakeBeamState('start')
    dead_end = FakeBeamState('dead_end')
    live = FakeBeamState('live')
    tail = FakeBeamState('tail')
    to_dead_end = FakeBeamAction('to_dead_end', dead_end)
    to_live = FakeBeamAction('to_live', live)
    to_tail = FakeBeamAction('to_tail', tail)
    start.set_actions([to_dead_end, to_live])
    live.set_actions([to_tail])
    tail.set_actions([FakeBeamAction('stay', tail)])
    goal_condition = FakeBeamGoalCondition()
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(0.0), 2)
    trajectory_state = TrajectoryState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    search_state = sampler.SearchState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))

    sampler._beam_step(trajectory_state,
                       search_state,
                       max_depth=2,
                       beam_successor_values=[(torch.tensor([10.0, 9.0]), [cast(mm.GroundAction, to_dead_end), cast(mm.GroundAction, to_live)])])
    assert not trajectory_state.done
    assert search_state.beam_list == [dead_end, live]
    assert search_state.open_list == [live]

    sampler._beam_step(trajectory_state,
                       search_state,
                       max_depth=2,
                       beam_successor_values=[(torch.tensor([8.0]), [cast(mm.GroundAction, to_tail)])])
    sampler._finalize_state(trajectory_state, search_state)

    assert search_state.beam_list == [tail]
    assert trajectory_state.state_sequence == [start, live, tail]


def test_beam_search_returns_live_branch_at_horizon_instead_of_dead_end():
    start = FakeBeamState('start')
    dead_end = FakeBeamState('dead_end')
    live = FakeBeamState('live')
    start.set_actions([
        FakeBeamAction('to_dead_end', dead_end),
        FakeBeamAction('to_live', live),
    ])
    live.set_actions([FakeBeamAction('stay', live)])
    goal_condition = FakeBeamGoalCondition()
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(0.0), 2)
    trajectory_state = TrajectoryState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    search_state = sampler.SearchState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))

    sampler._beam_step(
        trajectory_state,
        search_state,
        max_depth=1,
        beam_successor_values=[(
            torch.tensor([10.0, 9.0]),
            [cast(mm.GroundAction, start.applicable_actions()[0]),
             cast(mm.GroundAction, start.applicable_actions()[1])],
        )],
    )
    sampler._finalize_state(trajectory_state, search_state)

    assert trajectory_state.done
    assert search_state.beam_list == [dead_end, live]
    assert search_state.open_list == [live]
    assert trajectory_state.state_sequence == [start, live]


def test_beam_search_terminates_safely_proven_dead_end():
    start = FakeBeamState('start')
    proven_dead_end = FakeBeamState('proven_dead_end')
    live = FakeBeamState('live')
    proven_dead_end.set_actions([FakeBeamAction('proven_stay', proven_dead_end)])
    live.set_actions([FakeBeamAction('live_stay', live)])
    to_proven_dead_end = FakeBeamAction('to_proven_dead_end', proven_dead_end, 0.0)
    to_live = FakeBeamAction('to_live', live, 0.0)
    start.set_actions([to_proven_dead_end, to_live])
    goal_condition = FakeBeamGoalCondition()
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), ActionBeamRewardFunction({proven_dead_end}), 1)
    trajectory_state = TrajectoryState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    search_state = sampler.SearchState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))

    sampler._beam_step(trajectory_state,
                       search_state,
                       max_depth=5,
                       beam_successor_values=[(torch.tensor([10.0, -10.0]), [cast(mm.GroundAction, to_proven_dead_end), cast(mm.GroundAction, to_live)])])
    sampler._finalize_state(trajectory_state, search_state)

    assert trajectory_state.done
    assert search_state.beam_list == [proven_dead_end]
    assert search_state.open_list == []
    assert trajectory_state.state_sequence == [start, proven_dead_end]
    assert trajectory_state.reward_sequence == [RewardFunction.get_dead_end_reward()]


def test_beam_search_does_not_infer_dead_end_from_reward_value():
    start = FakeBeamState('start')
    live = FakeBeamState('live')
    live.set_actions([FakeBeamAction('stay', live)])
    action = FakeBeamAction('to_live', live)
    start.set_actions([action])
    goal_condition = FakeBeamGoalCondition()
    sampler = BeamSearchTrajectorySampler(
        DummyBeamModel(),
        DummyBeamRewardFunction(RewardFunction.get_dead_end_reward()),
        1,
    )
    trajectory_state = TrajectoryState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    search_state = sampler.SearchState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))

    sampler._beam_step(
        trajectory_state,
        search_state,
        max_depth=5,
        beam_successor_values=[(torch.tensor([5.0]), [cast(mm.GroundAction, action)])],
    )

    assert not trajectory_state.done
    assert search_state.open_list == [live]


def test_beam_search_recognizes_safely_proven_initial_dead_end():
    start = FakeBeamState('start')
    start.set_actions([FakeBeamAction('stay', start)])
    goal_condition = FakeBeamGoalCondition()
    reward_function = DummyBeamRewardFunction(0.0, {start})
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), reward_function, 1)

    trajectory_states, search_states = sampler._initialize([
        (cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition)),
    ])

    assert trajectory_states[0].done
    assert not trajectory_states[0].solved
    assert search_states[0].open_list == []


def test_beam_search_keeps_dead_end_successor_in_final_trajectory():
    start = FakeBeamState('start')
    dead_end = FakeBeamState('dead')
    dead_end.set_actions([])
    action = FakeBeamAction('to_dead', dead_end)
    start.set_actions([action])
    goal_condition = FakeBeamGoalCondition()
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(-1.0), 2)
    trajectory_state = TrajectoryState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    search_state = sampler.SearchState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    beam_successor_values = [(torch.tensor([7.0]), [cast(mm.GroundAction, action)])]
    sampler._beam_step(trajectory_state, search_state, max_depth=5, beam_successor_values=beam_successor_values)
    sampler._finalize_state(trajectory_state, search_state)
    assert trajectory_state.done
    assert trajectory_state.state_sequence == [start, dead_end]
    assert trajectory_state.action_sequence == [action]
    assert trajectory_state.reward_sequence == [RewardFunction.get_dead_end_reward()]
    assert trajectory_state.q_value_sequence == [7.0]
    assert trajectory_state.value_sequence == [7.0]


def test_beam_search_keeps_best_duplicate_successor():
    state_1 = FakeBeamState('state_1')
    state_2 = FakeBeamState('state_2')
    shared = FakeBeamState('shared')
    shared.set_actions([FakeBeamAction('stay', shared)])
    action_1 = FakeBeamAction('to_shared_from_state_1', shared)
    action_2 = FakeBeamAction('to_shared_from_state_2', shared)
    state_1.set_actions([action_1])
    state_2.set_actions([action_2])
    goal_condition = FakeBeamGoalCondition()
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(0.0), 2)
    trajectory_state = TrajectoryState(cast(mm.State, state_1), cast(mm.GroundConjunctiveCondition, goal_condition))
    search_state = cast(Any, sampler.SearchState(cast(mm.State, state_1), cast(mm.GroundConjunctiveCondition, goal_condition)))
    search_state.beam_list = [state_1, state_2]
    search_state.open_list = [state_1, state_2]
    search_state.closed_set = {state_1, state_2}
    beam_successor_values = [
        (torch.tensor([1.0]), [cast(mm.GroundAction, action_1)]),
        (torch.tensor([10.0]), [cast(mm.GroundAction, action_2)]),
    ]
    sampler._beam_step(trajectory_state,
                       search_state,
                       max_depth=5,
                       beam_successor_values=beam_successor_values)
    predecessor_state, _, _, q_value = search_state.transition_map[shared]
    assert predecessor_state == state_2
    assert q_value == 10.0


def test_beam_search_does_not_collapse_to_revisited_state():
    start = FakeBeamState('start')
    middle = FakeBeamState('middle')
    action_to_middle = FakeBeamAction('to_middle', middle)
    action_back_to_start = FakeBeamAction('back_to_start', start)
    start.set_actions([action_to_middle])
    middle.set_actions([action_back_to_start])
    goal_condition = FakeBeamGoalCondition()
    sampler = BeamSearchTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(0.0), 1)
    trajectory_state = TrajectoryState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    search_state = sampler.SearchState(cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition))
    sampler._beam_step(trajectory_state,
                       search_state,
                       max_depth=2,
                       beam_successor_values=[(torch.tensor([5.0]), [cast(mm.GroundAction, action_to_middle)])])
    sampler._beam_step(trajectory_state,
                       search_state,
                       max_depth=2,
                       beam_successor_values=[(torch.tensor([4.0]), [cast(mm.GroundAction, action_back_to_start)])])
    sampler._finalize_state(trajectory_state, search_state)
    assert trajectory_state.state_sequence == [start, middle]
    assert trajectory_state.action_sequence == [action_to_middle]
    assert trajectory_state.reward_sequence == [0.0]


def test_policy_rollout_records_max_q_as_state_value():
    start = FakeBeamState('start')
    successor_1 = FakeBeamState('successor_1')
    successor_2 = FakeBeamState('successor_2')
    successor_1.set_actions([FakeBeamAction('stay_1', successor_1)])
    successor_2.set_actions([FakeBeamAction('stay_2', successor_2)])
    action_1 = FakeBeamAction('action_1', successor_1, 0.0)
    action_2 = FakeBeamAction('action_2', successor_2, 100.0)
    start.set_actions([action_1, action_2])
    goal_condition = FakeBeamGoalCondition()
    sampler = GreedyPolicyTrajectorySampler(FixedBeamModel([5.0, 4.0]), ActionBeamRewardFunction())
    trajectory_states, rollout_states = sampler._initialize([
        (cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition)),
    ])

    sampler._internal_sample(trajectory_states, rollout_states, [5])

    assert trajectory_states[0].value_sequence == [5.0]
    assert trajectory_states[0].action_sequence == [action_1]


def test_policy_rollout_revisit_mask_supports_float16():
    start = FakeBeamState('start')
    stay = FakeBeamAction('stay', start)
    start.set_actions([stay])
    goal_condition = FakeBeamGoalCondition()
    sampler = GreedyPolicyTrajectorySampler(
        FixedBeamModel([0.0], dtype=torch.float16),
        DummyBeamRewardFunction(0.0),
    )
    trajectory_states, rollout_states = sampler._initialize([
        (cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition)),
    ])

    sampler._internal_sample(trajectory_states, rollout_states, [1])

    assert trajectory_states[0].action_sequence == [stay]


def test_policy_rollout_revisit_mask_suppresses_revisit_below_any_finite_score():
    start = FakeBeamState('start')
    live = FakeBeamState('live')
    stay = FakeBeamAction('stay', start)
    advance = FakeBeamAction('advance', live)
    start.set_actions([stay, advance])
    live.set_actions([FakeBeamAction('live_stay', live)])
    goal_condition = FakeBeamGoalCondition()
    sampler = GreedyPolicyTrajectorySampler(
        FixedBeamModel([0.0, -2_000_000.0]),
        DummyBeamRewardFunction(0.0),
    )
    trajectory_states, rollout_states = sampler._initialize([
        (cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition)),
    ])

    sampler._internal_sample(trajectory_states, rollout_states, [1])

    assert trajectory_states[0].action_sequence == [advance]


@pytest.mark.parametrize('temperature', [0.5, 1.0, 2.0])
@pytest.mark.parametrize('sampler_factory', [
    lambda model, reward_function, temperature: BoltzmannTrajectorySampler(model, reward_function, temperature),
    lambda model, reward_function, temperature: StateBoltzmannTrajectorySampler(model, reward_function, temperature, temperature, 10),
], ids=['boltzmann', 'state-boltzmann'])
def test_boltzmann_policy_rollout_suppresses_partial_revisits(sampler_factory, temperature):
    start = FakeBeamState('start')
    live = FakeBeamState('live')
    stay = FakeBeamAction('stay', start)
    advance = FakeBeamAction('advance', live)
    start.set_actions([stay, advance])
    live.set_actions([FakeBeamAction('live_stay', live)])
    goal_condition = FakeBeamGoalCondition()
    sampler = sampler_factory(
        FixedBeamModel([1000.0, 0.0]),
        DummyBeamRewardFunction(0.0),
        temperature,
    )
    trajectory_states, rollout_states = sampler._initialize([
        (cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition)),
    ])

    sampler._internal_sample(trajectory_states, rollout_states, [1])

    assert trajectory_states[0].action_sequence == [advance]


@pytest.mark.parametrize('temperature', [0.5, 1.0, 2.0])
@pytest.mark.parametrize('sampler_factory', [
    lambda model, reward_function, temperature: BoltzmannTrajectorySampler(model, reward_function, temperature),
    lambda model, reward_function, temperature: StateBoltzmannTrajectorySampler(model, reward_function, temperature, temperature, 10),
], ids=['boltzmann', 'state-boltzmann'])
def test_boltzmann_policy_rollout_falls_back_when_all_successors_are_revisits(sampler_factory, temperature):
    start = FakeBeamState('start')
    stay = FakeBeamAction('stay', start)
    start.set_actions([stay])
    goal_condition = FakeBeamGoalCondition()
    sampler = sampler_factory(
        FixedBeamModel([0.0]),
        DummyBeamRewardFunction(0.0),
        temperature,
    )
    trajectory_states, rollout_states = sampler._initialize([
        (cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition)),
    ])

    sampler._internal_sample(trajectory_states, rollout_states, [1])

    assert trajectory_states[0].action_sequence == [stay]


def test_policy_rollout_terminates_explicit_dead_end_with_dead_end_cost():
    start = FakeBeamState('start')
    dead_end = FakeBeamState('dead_end')
    dead_end.set_actions([FakeBeamAction('stay', dead_end)])
    action = FakeBeamAction('to_dead_end', dead_end, 25.0)
    start.set_actions([action])
    goal_condition = FakeBeamGoalCondition()
    reward_function = ActionBeamRewardFunction({dead_end})
    sampler = GreedyPolicyTrajectorySampler(FixedBeamModel([5.0]), reward_function)
    trajectory_states, rollout_states = sampler._initialize([
        (cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition)),
    ])

    sampler._internal_sample(trajectory_states, rollout_states, [5])

    assert trajectory_states[0].done
    assert not trajectory_states[0].solved
    assert trajectory_states[0].action_sequence == [action]
    assert trajectory_states[0].reward_sequence == [RewardFunction.get_dead_end_reward()]


def test_policy_rollout_handles_explicit_initial_dead_end():
    domain = mm.Domain.from_file(DATA_DIR / 'gripper' / 'domain.pddl')
    problem = mm.Problem.from_file(domain, DATA_DIR / 'gripper' / 'problem.pddl')
    start = problem.initial_state
    reward_function = DummyBeamRewardFunction(0.0, cast(Any, {start}))
    sampler = GreedyPolicyTrajectorySampler(DummyBeamModel(), reward_function)

    trajectory = sampler.sample([(start, problem.goal)], 5)[0]

    assert len(trajectory) == 0
    assert trajectory.is_unsolvable()
    trajectory.validate()


@pytest.mark.parametrize('horizon', [0, -1, 1.5])
def test_policy_rollout_rejects_invalid_horizon(horizon: int | float):
    sampler = GreedyPolicyTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(0.0))
    with pytest.raises(AssertionError):
        sampler.sample([], horizon)  # type: ignore[arg-type]


def test_policy_rollout_checks_horizon_before_model_evaluation():
    start = FakeBeamState('start')
    successor = FakeBeamState('successor')
    start.set_actions([FakeBeamAction('advance', successor)])
    goal_condition = FakeBeamGoalCondition()
    sampler = GreedyPolicyTrajectorySampler(DummyBeamModel(), DummyBeamRewardFunction(0.0))
    trajectory_states, rollout_states = sampler._initialize([
        (cast(mm.State, start), cast(mm.GroundConjunctiveCondition, goal_condition)),
    ])

    sampler._internal_sample(trajectory_states, rollout_states, [0])

    assert trajectory_states[0].done
    assert trajectory_states[0].action_sequence == []


def test_ff_reward_function():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    reward_function = FFRewardFunction()
    current_state = problem.initial_state
    goal_condition = problem.goal
    for action in current_state.applicable_actions():
        successor_state = action.apply(current_state)
        reward = reward_function(current_state, action, successor_state, goal_condition)
        assert isinstance(reward, float)


def test_sum_reward_function():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    reward_function_1 = ConstantRewardFunction(-1.0)
    reward_function_2 = FFRewardFunction()
    sum_reward_function = SumRewardFunction([reward_function_1, reward_function_2])
    current_state = problem.initial_state
    goal_condition = problem.goal
    for action in current_state.applicable_actions():
        successor_state = action.apply(current_state)
        reward = sum_reward_function(current_state, action, successor_state, goal_condition)
        assert isinstance(reward, float)


@pytest.mark.parametrize("domain_name, trajectory_sampler_creator", [
    ('blocks', lambda model, reward_function: PolicyTrajectorySampler(model, reward_function)),
    ('blocks', lambda model, reward_function: BoltzmannTrajectorySampler(model, reward_function, 1.0)),
    ('blocks', lambda model, reward_function: StateBoltzmannTrajectorySampler(model, reward_function, 1.0, 0.1, 10)),
    ('blocks', lambda model, reward_function: GreedyPolicyTrajectorySampler(model, reward_function)),
    ('blocks', lambda model, reward_function: EpsilonGreedyTrajectorySampler(model, reward_function, 0.5)),
    ('blocks', lambda model, reward_function: BeamSearchTrajectorySampler(model, reward_function, 8)),
    ('gripper', lambda model, reward_function: PolicyTrajectorySampler(model, reward_function)),
    ('gripper', lambda model, reward_function: BoltzmannTrajectorySampler(model, reward_function, 1.0)),
    ('gripper', lambda model, reward_function: GreedyPolicyTrajectorySampler(model, reward_function)),
    ('gripper', lambda model, reward_function: StateBoltzmannTrajectorySampler(model, reward_function, 1.0, 0.1, 10)),
    ('gripper', lambda model, reward_function: EpsilonGreedyTrajectorySampler(model, reward_function, 0.5)),
    ('gripper', lambda model, reward_function: BeamSearchTrajectorySampler(model, reward_function, 16)),
    ('blocks-hard', lambda model, reward_function: PolicyTrajectorySampler(model, reward_function)),
    ('blocks-hard', lambda model, reward_function: BoltzmannTrajectorySampler(model, reward_function, 1.0)),
    ('blocks-hard', lambda model, reward_function: StateBoltzmannTrajectorySampler(model, reward_function, 1.0, 0.1, 10)),
    ('blocks-hard', lambda model, reward_function: GreedyPolicyTrajectorySampler(model, reward_function)),
    ('blocks-hard', lambda model, reward_function: EpsilonGreedyTrajectorySampler(model, reward_function, 0.5)),
    ('blocks-hard', lambda model, reward_function: BeamSearchTrajectorySampler(model, reward_function, 8)),
    ('gripper-hard', lambda model, reward_function: PolicyTrajectorySampler(model, reward_function)),
    ('gripper-hard', lambda model, reward_function: BoltzmannTrajectorySampler(model, reward_function, 1.0)),
    ('gripper-hard', lambda model, reward_function: GreedyPolicyTrajectorySampler(model, reward_function)),
    ('gripper-hard', lambda model, reward_function: StateBoltzmannTrajectorySampler(model, reward_function, 1.0, 0.1, 10)),
    ('gripper-hard', lambda model, reward_function: EpsilonGreedyTrajectorySampler(model, reward_function, 0.5)),
    ('gripper-hard', lambda model, reward_function: BeamSearchTrajectorySampler(model, reward_function, 16))
])
def test_trajectory_sampler(domain_name: str, trajectory_sampler_creator: Callable[[ActionScalarModel, RewardFunction], TrajectorySampler]):
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    model = RGNNWrapper(domain)
    reward_function = GoalTransitionRewardFunction(1)
    trajectory_sampler = trajectory_sampler_creator(model, reward_function)
    trajectories = trajectory_sampler.sample([(problem.initial_state, problem.goal)], 10)
    assert isinstance(trajectories, list)
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert isinstance(trajectory, Trajectory)
    assert trajectory.is_solution() or len(trajectory) <= 10
    trajectory.validate()  # Performs asserts internally.


@pytest.mark.parametrize("domain_name", ['blocks', 'blocks-hard', 'gripper', 'gripper-hard', 'spanner', 'spanner-hard'])
def test_trajectory_sampler_multiple(domain_name: str):
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    model = RGNNWrapper(domain)
    reward_function = GoalTransitionRewardFunction(1)
    trajectory_samplers: list[TrajectorySampler] = [
        GreedyPolicyTrajectorySampler(model, reward_function),
        BeamSearchTrajectorySampler(model, reward_function, 8)
    ]
    trajectory_probabilities = [0.5, 0.5]
    trajectory_sampler = MultipleTrajectorySampler(reward_function, trajectory_samplers, trajectory_probabilities, 3)
    trajectories = trajectory_sampler.sample([(problem.initial_state, problem.goal)], 10)
    assert isinstance(trajectories, list)
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert isinstance(trajectory, Trajectory)
    assert trajectory.is_solution() or len(trajectory) <= 10
    trajectory.validate()  # Performs asserts internally.


@pytest.mark.parametrize('horizon', [0, -1, 1.5])
def test_multiple_trajectory_sampler_rejects_invalid_horizon(horizon: int | float):
    reward_function = DummyBeamRewardFunction(0.0)
    child_sampler = GreedyPolicyTrajectorySampler(DummyBeamModel(), reward_function)
    sampler = MultipleTrajectorySampler(reward_function, [child_sampler], [1.0])

    with pytest.raises(AssertionError):
        sampler.sample([], horizon)  # type: ignore[arg-type]


@pytest.mark.parametrize("domain_name", ['blocks-hard', 'gripper-hard'])
def test_state_hindsight(domain_name: str):
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    model = RGNNWrapper(domain)
    reward_function = ConstantRewardFunction(-1)
    trajectory_sampler = PolicyTrajectorySampler(model, reward_function)
    trajectory_refiner = StateHindsightTrajectoryRefiner(10)
    original_trajectories = trajectory_sampler.sample([(problem.initial_state, problem.goal)], 100)
    refined_trajectories = trajectory_refiner.refine(original_trajectories)
    assert len(refined_trajectories) < 10
    for refined_trajectory in refined_trajectories:
        assert refined_trajectory.is_solution()
        refined_trajectory.validate()


@pytest.mark.parametrize("domain_name", ['blocks-hard', 'gripper-hard'])
def test_propositional_hindsight(domain_name: str):
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    model = RGNNWrapper(domain)
    reward_function = ConstantRewardFunction(-1)
    trajectory_sampler = PolicyTrajectorySampler(model, reward_function)
    trajectory_refiner = PropositionalHindsightTrajectoryRefiner([problem], 10)
    original_trajectories = trajectory_sampler.sample([(problem.initial_state, problem.goal)], 100)
    refined_trajectories = trajectory_refiner.refine(original_trajectories)
    assert len(refined_trajectories) < 10
    for refined_trajectory in refined_trajectories:
        assert refined_trajectory.is_solution()
        refined_trajectory.validate()


@pytest.mark.parametrize("domain_name", ['blocks-hard', 'gripper-hard'])
def test_lifted_hindsight(domain_name: str):
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    model = RGNNWrapper(domain)
    reward_function = ConstantRewardFunction(-1)
    trajectory_sampler = PolicyTrajectorySampler(model, reward_function)
    trajectory_refiner = LiftedHindsightTrajectoryRefiner([problem], 10)
    original_trajectories = trajectory_sampler.sample([(problem.initial_state, problem.goal)], 100)
    refined_trajectories = trajectory_refiner.refine(original_trajectories)
    assert len(refined_trajectories) < 10
    for refined_trajectory in refined_trajectories:
        assert refined_trajectory.is_solution()
        refined_trajectory.validate()


@pytest.mark.parametrize("domain_name", ['blocks', 'gripper'])
def test_off_policy_algorithm(domain_name: str):
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    problems = [problem]
    model = RGNNWrapper(domain)
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    discount_factor = 0.999
    loss_function = DQNOptimization(model, optimizer, lr_scheduler, model, discount_factor, 10.0)
    reward_function = ConstantRewardFunction(-1)
    replay_buffer = PrioritizedReplayBuffer(100)
    trajectory_sampler = PolicyTrajectorySampler(model, reward_function)
    horizon = 100
    rollout_count = 2
    batch_size = 4
    train_steps = 8
    problem_sampler = UniformProblemSampler()
    initial_state_sampler = OriginalInitialStateSampler()
    goal_condition_sampler = OriginalGoalConditionSampler()
    trajectory_refiner = PropositionalHindsightTrajectoryRefiner(problems, 10)
    algorithm = OffPolicyAlgorithm(problems,
                                   loss_function,
                                   reward_function,
                                   replay_buffer,
                                   replay_buffer,
                                   trajectory_sampler,
                                   horizon,
                                   rollout_count,
                                   batch_size,
                                   train_steps,
                                   problem_sampler,
                                   initial_state_sampler,
                                   goal_condition_sampler,
                                   trajectory_refiner)
    algorithm.fit()
    algorithm.fit()
    algorithm.fit()

def test_algorithm_hooks():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    problems = [problem]
    model = RGNNWrapper(domain)
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    discount_factor = 0.999
    loss_function = DQNOptimization(model, optimizer, lr_scheduler, model, discount_factor, 100.0)
    reward_function = ConstantRewardFunction(-1)
    replay_buffer = PrioritizedReplayBuffer(100)
    trajectory_sampler = PolicyTrajectorySampler(model, reward_function)
    horizon = 10
    rollout_count = 2
    batch_size = 4
    train_steps = 8
    problem_sampler = UniformProblemSampler()
    initial_state_sampler = OriginalInitialStateSampler()
    goal_condition_sampler = OriginalGoalConditionSampler()
    trajectory_refiner = PropositionalHindsightTrajectoryRefiner(problems, 10)
    algorithm = OffPolicyAlgorithm(problems,
                                   loss_function,
                                   reward_function,
                                   replay_buffer,
                                   replay_buffer,
                                   trajectory_sampler,
                                   horizon,
                                   rollout_count,
                                   batch_size,
                                   train_steps,
                                   problem_sampler,
                                   initial_state_sampler,
                                   goal_condition_sampler,
                                   trajectory_refiner)
    sample_problems: list[bool] = []
    sample_initial_states: list[bool] = []
    sample_goal_conditions: list[bool] = []
    sample_trajectories: list[bool] = []
    refine_trajectories: list[bool] = []
    pre_collect_experience: list[bool] = []
    post_collect_experience: list[bool] = []
    pre_optimize_model: list[bool] = []
    post_optimize_model: list[bool] = []
    train_step: list[bool] = []
    algorithm.register_on_sample_problems(lambda x: sample_problems.append(True))
    algorithm.register_on_sample_initial_states(lambda x: sample_initial_states.append(True))
    algorithm.register_on_sample_goal_conditions(lambda x: sample_goal_conditions.append(True))
    algorithm.register_on_sample_trajectories(lambda x: sample_trajectories.append(True))
    algorithm.register_on_refine_trajectories(lambda x: refine_trajectories.append(True))
    algorithm.register_on_pre_collect_experience(lambda: pre_collect_experience.append(True))
    algorithm.register_on_post_collect_experience(lambda: post_collect_experience.append(True))
    algorithm.register_on_pre_optimize_model(lambda: pre_optimize_model.append(True))
    algorithm.register_on_post_optimize_model(lambda: post_optimize_model.append(True))
    algorithm.register_on_train_step(lambda x, l: train_step.append(True))
    algorithm.fit()
    assert len(sample_problems) == 1
    assert len(sample_initial_states) == 1
    assert len(sample_goal_conditions) == 1
    assert len(sample_trajectories) == 1
    assert len(refine_trajectories) == 1
    assert len(pre_collect_experience) == 1
    assert len(post_collect_experience) == 1
    assert len(pre_optimize_model) == 1
    assert len(post_optimize_model) == 1
    assert (len(train_step) == train_steps) or (len(train_step) == 0)


def test_value_based_initial_state_sampler():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    problems: list[mm.Problem] = [problem] * 100
    model: ActionScalarModel = RGNNWrapper(domain)
    reward_function: RewardFunction = ConstantRewardFunction(-1)
    initial_state_sampler = TopValueInitialStateSampler(problems, model, reward_function, 0.5, 0.1, 10)
    # First, test without any additional states to the pool of initial states.
    # We expect to get the original initial state.
    sampled_initial_states_1 = initial_state_sampler.sample(problems)
    assert len(sampled_initial_states_1) == len(problems)
    assert all(sampled_initial_states_1[idx] == problem.initial_state for idx, problem in enumerate(problems))
    # Second, test adding a bunch of states.
    trajectory_sampler: TrajectorySampler = BoltzmannTrajectorySampler(model, reward_function, 1.0)
    sampled_trajectory = trajectory_sampler.sample([(problem.initial_state, problem.goal) for problem in problems], 100)
    for trajectory in sampled_trajectory:
        for transition in trajectory:
            initial_state_sampler.add_state(transition.current_state, transition.predicted_value)
    sampled_initial_states_2 = initial_state_sampler.sample(problems)
    assert len(sampled_initial_states_2) == len(problems)


def test_value_based_initial_state_sampler_uses_raw_q_and_terminal_state_values():
    domain = mm.Domain.from_file(DATA_DIR / 'gripper' / 'domain.pddl')
    problem = mm.Problem.from_file(domain, DATA_DIR / 'gripper' / 'problem.pddl')
    initial_state = problem.initial_state
    to_dead_end = next(action for action in initial_state.applicable_actions()
                       if str(action) == '(pick ball1 rooma left)')
    explicit_dead_end = to_dead_end.apply(initial_state)

    goal_state = initial_state
    for action_name in (
        '(pick ball2 rooma left)',
        '(move rooma roomb)',
        '(drop ball2 roomb left)',
    ):
        action = next(action for action in goal_state.applicable_actions() if str(action) == action_name)
        goal_state = action.apply(goal_state)
    assert problem.goal.holds(goal_state)

    model = FixedBeamModel([5.0, 4.0] + [0.0] * 14)
    reward_function = DummyBeamRewardFunction(20_000.0, cast(Any, {explicit_dead_end}))
    sampler = TopValueInitialStateSampler([problem], model, reward_function, 0.0, 1.0, 10)
    sampler.state_buffers[problem] = [initial_state, explicit_dead_end, goal_state]
    sampler.value_buffers[problem] = [float('nan')] * 3

    sampler._update_state_values(problem)

    assert sampler.value_buffers[problem] == [
        5.0,
        RewardFunction.get_dead_end_reward(),
        0.0,
    ]


def test_evaluation():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    problems: list[mm.Problem] = [problem]
    model: ActionScalarModel = RGNNWrapper(domain)
    reward_function: RewardFunction = ConstantRewardFunction(-1)
    trajectory_sampler: TrajectorySampler = GreedyPolicyTrajectorySampler(model, reward_function)
    criterias: list[EvaluationCriteria] = [CoverageCriteria(), LengthCriteria(False), TDErrorCriteria(False)]
    horizon: int = 100
    evaluation = PolicyEvaluation(problems, criterias, trajectory_sampler, horizon)
    best1, result1 = evaluation.evaluate()
    best2, result2 = evaluation.evaluate()
    assert best1
    assert not best2
    assert result1 == result2


def test_sequential_evaluation():
    domain_path = DATA_DIR / 'gripper' / 'domain.pddl'
    problem_path = DATA_DIR / 'gripper' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    problems: list[mm.Problem] = [problem] * 10
    model: ActionScalarModel = RGNNWrapper(domain)
    reward_function: RewardFunction = ConstantRewardFunction(-1)
    trajectory_sampler: TrajectorySampler = GreedyPolicyTrajectorySampler(model, reward_function)
    criterias: list[EvaluationCriteria] = [CoverageCriteria(), LengthCriteria(False), TDErrorCriteria(False)]
    horizon: int = 100
    evaluation = SequentialPolicyEvaluation(problems, criterias, trajectory_sampler, horizon, 3)
    best1, result1 = evaluation.evaluate()
    best2, result2 = evaluation.evaluate()
    assert best1
    assert not best2
    assert result1 == result2
