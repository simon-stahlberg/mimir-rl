import pymimir as mm
import pymimir_rl as rl
import pytest
import torch


@pytest.fixture
def problem():
    domain = mm.Domain.from_pddl('''
        (define (domain dead-regions) (:requirements :strips)
          (:predicates (ready) (live) (dead0) (dead1) (dead2) (goal))
          (:action advance :parameters () :precondition (ready)
            :effect (and (not (ready)) (live)))
          (:action lose :parameters () :precondition (live)
            :effect (and (not (live)) (dead0)))
          (:action finish :parameters () :precondition (live) :effect (goal))
          (:action explore0 :parameters () :precondition (dead0)
            :effect (and (not (dead0)) (dead1)))
          (:action explore1 :parameters () :precondition (dead1)
            :effect (and (not (dead1)) (dead2)))
          (:action loop :parameters () :precondition (dead2) :effect (dead2)))
    ''')
    return mm.Problem.from_pddl(domain, '''
        (define (problem p) (:domain dead-regions) (:init (ready)) (:goal (goal)))
    ''', generator="grounded")


class PreferenceModel(rl.ActionScalarModel):
    def __init__(self, preferred="lose", events=None, value=0.0):
        super().__init__()
        self.preferred = preferred
        self.events = events if events is not None else []
        self.value = torch.nn.Parameter(torch.tensor(value))

    def forward(self, state_goals):
        self.events.append("model")
        result = []
        for state, _ in state_goals:
            actions = list(state.applicable_actions())
            values = torch.tensor([float(action.schema.name == self.preferred) for action in actions])
            result.append((values + self.value, actions))
        return result


@pytest.mark.parametrize("start_in_dead_end", [False, True])
def test_h2_labels_only_after_rollout_and_relabels_hindsight(problem, start_in_dead_end):
    events = []
    created = []
    checked = []

    def factory(problem):
        created.append(problem)
        return mm.H2DeadEndDetector(problem)

    cached_detector = rl.CachedDeadEndDetector(factory)

    def detector(state, goal):
        events.append("detect")
        checked.append(state)
        return cached_detector(state, goal)

    start = problem.initial_state
    if start_in_dead_end:
        start = start.applicable_actions()[0].apply(start)
        lose = next(a for a in start.applicable_actions() if a.schema.name == "lose")
        start = lose.apply(start)
    sampler = rl.GreedyPolicyTrajectorySampler(
        PreferenceModel(events=events), rl.ConstantRewardFunction(-1.0),
        dead_end_detector=detector,
    )
    trajectory = sampler.sample([(start, problem.goal)], 5)[0]

    assert len(trajectory) == 5
    assert events[:5] == ["model"] * 5
    assert all(event == "detect" for event in events[5:])
    assert len(checked) == (1 if start_in_dead_end else 3)
    assert created == [problem]
    assert trajectory.is_unsolvable()
    assert [t.successor_is_dead_end for t in trajectory] == ([True] * 5 if start_in_dead_end else [False, True, True, True, True])
    assert not any(t.is_terminal for t in trajectory)
    assert [t.immediate_reward for t in trajectory] == [-1.0] * 5
    assert trajectory[0].future_rewards == -4.0
    trajectory.validate(False)

    # This goal is reachable along the recorded suffix, despite the original goal being impossible.
    hindsight_goal = problem.ground_condition(problem.fact("dead2"))
    end_index = 1 if start_in_dead_end else 3
    checked_before_hindsight = len(checked)
    hindsight = trajectory.clone_with_goal(0, end_index, hindsight_goal)
    assert hindsight.is_solution()
    assert not hindsight.is_unsolvable()
    assert not any(t.successor_is_dead_end for t in hindsight)
    assert hindsight[-1].is_terminal
    hindsight.validate()
    assert created == [problem]
    assert len(checked) == checked_before_hindsight


def test_forward_proof_does_not_require_detector_to_recognize_later_states(problem):
    checked = []

    def detector(state, goal):
        checked.append(state)
        return state.holds(problem.fact("dead0"))

    trajectory = rl.GreedyPolicyTrajectorySampler(
        PreferenceModel(), rl.ConstantRewardFunction(-1.0), dead_end_detector=detector,
    ).sample([(problem.initial_state, problem.goal)], 5)[0]
    assert len(checked) == 3
    assert not trajectory[0].successor_is_dead_end
    assert all(t.successor_is_dead_end for t in trajectory.transitions[1:])
    assert trajectory.is_unsolvable()


class RecordingOptimization(rl.OptimizationFunction):
    def __init__(self):
        self.batches = []

    def __call__(self, transitions, weights):
        self.batches.append(transitions)
        assert len(transitions) == len(weights)
        return torch.ones(len(transitions))


def make_algorithm(problem, sampler, loss, dead_buffer, hindsight_buffer, *, dead_end_q_factor=0.25):
    return rl.OffPolicyAlgorithm(
        [problem], loss, sampler.reward_function, dead_buffer, hindsight_buffer,
        sampler, horizon=5, rollout_count=1, batch_size=4, train_steps=1,
        trajectory_refiner=rl.StateHindsightTrajectoryRefiner(100),
        dead_end_q_factor=dead_end_q_factor,
    )


def test_replay_admits_only_proven_trajectories_and_relabels_all_outcomes(problem, monkeypatch):
    sampler = rl.GreedyPolicyTrajectorySampler(
        PreferenceModel(), rl.ConstantRewardFunction(-1.0),
        dead_end_detector=rl.CachedDeadEndDetector(mm.H2DeadEndDetector),
    )
    state_goals = [(problem.initial_state, problem.goal)]
    truncated = sampler.sample(state_goals, 1)[0]
    dead = sampler.sample(state_goals, 5)[0]
    sampler.model.preferred = "finish"
    solved = sampler.sample(state_goals, 5)[0]
    assert not truncated.is_unsolvable() and not truncated.is_solution()
    assert solved.is_solution()
    truncated[0].predicted_value = -10000.0
    solved[0].predicted_value = -10000.0
    initially_dead = dead.clone_with_goal(2, 4, problem.goal)
    assert initially_dead.is_unsolvable()
    assert all(t.successor_is_dead_end for t in initially_dead)
    empty = rl.Trajectory(
        [dead.final_state], [], [], [], [], sampler.reward_function, problem.goal,
        sampler.dead_end_detector,
    )
    assert empty.is_unsolvable()
    dead_buffer = rl.PrioritizedReplayBuffer(100)
    hindsight_buffer = rl.PrioritizedReplayBuffer(100)
    algorithm = make_algorithm(problem, sampler, RecordingOptimization(), dead_buffer, hindsight_buffer)
    trajectories = [truncated, solved, dead, initially_dead, empty]
    monkeypatch.setattr(algorithm, "sample_trajectories", lambda pairs: trajectories)
    refine = algorithm.trajectory_refiner.refine

    def check_full_trajectories(sampled):
        assert sampled == trajectories
        assert len(dead) == 5
        return refine(sampled)

    monkeypatch.setattr(algorithm.trajectory_refiner, "refine", check_full_trajectories)
    algorithm.collect_experience(1)
    assert dead_buffer.buffer == dead.transitions[1:] + initially_dead.transitions
    assert len(hindsight_buffer) > 0
    assert all(t.part_of_solution and not t.successor_is_dead_end for t in hindsight_buffer.buffer)
    assert all(t not in list(truncated) + list(solved) + list(dead) + list(initially_dead)
               for t in hindsight_buffer.buffer)


@pytest.mark.parametrize("values,factor,expected_cutoff", [
    ([0.0] * 5, 0.25, 3),
    ([0.0, 0.0, -2500.0, 0.0, 0.0], 0.25, 1),
    ([0.0, 0.0, -2499.0, 0.0, 0.0], 0.25, 3),
    ([-2500.0, 0.0, 0.0, 0.0, 0.0], 0.25, 0),
    ([0.0, -3000.0, 0.0, 0.0, 0.0], 0.25, 0),
    ([0.0, 0.0, 0.0, -2500.0, 0.0], 0.25, 2),
    ([0.0, 0.0, 0.0, 0.0, -2500.0], 0.25, 3),
    ([0.0, 0.0, -2500.0, 0.0, 0.0], 0.5, 3),
    ([0.0, 0.0, -5000.0, 0.0, 0.0], 0.5, 1),
    ([0.0, 0.0, -10000.0, 0.0, 0.0], 1.0, 1),
    ([0.0, float("nan"), float("inf"), float("-inf"), 0.0], 0.25, 3),
])
def test_dead_end_replay_cutoff_preserves_labels_and_targets(problem, monkeypatch, values, factor, expected_cutoff):
    # Only the end of the dead region is recognized, leaving room for an earlier Q-based cutoff.
    sampler = rl.GreedyPolicyTrajectorySampler(
        PreferenceModel(), rl.ConstantRewardFunction(-1.0),
        dead_end_detector=lambda state, goal: state.holds(problem.fact("dead2")),
    )
    trajectory = sampler.sample([(problem.initial_state, problem.goal)], 5)[0]
    for transition, value in zip(trajectory, values, strict=True):
        transition.predicted_value = value
        transition.predicted_q_value = -10000.0
    labels = [t.successor_is_dead_end for t in trajectory]
    assert labels == [False, False, False, True, True]
    dead_buffer = rl.PrioritizedReplayBuffer(100)
    algorithm = make_algorithm(
        problem, sampler, RecordingOptimization(), dead_buffer, rl.PrioritizedReplayBuffer(100),
        dead_end_q_factor=factor,
    )
    monkeypatch.setattr(algorithm, "sample_trajectories", lambda pairs: [trajectory])

    def unexpected_inference(*args, **kwargs):
        pytest.fail("Replay selection must use recorded predictions")

    monkeypatch.setattr(sampler.model, "forward", unexpected_inference)
    algorithm.collect_experience(1)
    assert dead_buffer.buffer == trajectory.transitions[expected_cutoff:]
    assert [t.successor_is_dead_end for t in trajectory] == labels
    assert [t.immediate_reward for t in trajectory] == [-1.0] * 5

    model = PreferenceModel(preferred="", value=-10000.0)
    optimizer = torch.optim.Adam(model.parameters())
    loss = rl.DQNOptimization(
        model, optimizer, torch.optim.lr_scheduler.ConstantLR(optimizer), model, 1.0, 10.0,
    )
    targets = loss._compute_targets(dead_buffer.buffer, -10000.0, torch.device("cpu"))
    expected = torch.tensor([-10001.0, -10001.0, -10001.0, -10000.0, -10000.0])
    torch.testing.assert_close(targets, expected[expected_cutoff:])


@pytest.mark.parametrize("factor", [0.0, -0.25, 1.01, float("nan"), float("inf"), float("-inf")])
def test_dead_end_replay_rejects_invalid_q_factor(problem, factor):
    sampler = rl.GreedyPolicyTrajectorySampler(PreferenceModel(), rl.ConstantRewardFunction(-1.0))
    with pytest.raises(ValueError, match="dead_end_q_factor"):
        make_algorithm(
            problem, sampler, RecordingOptimization(), rl.PrioritizedReplayBuffer(10),
            rl.PrioritizedReplayBuffer(10), dead_end_q_factor=factor,
        )


@pytest.mark.parametrize("dead_count,hindsight_count,batch_size", [(0, 0, 4), (1, 0, 4), (0, 1, 4), (1, 1, 4), (1, 1, 1)])
def test_replay_sampling_with_empty_or_small_buffers(problem, dead_count, hindsight_count, batch_size):
    sampler = rl.GreedyPolicyTrajectorySampler(PreferenceModel(), rl.ConstantRewardFunction(-1.0))
    transition = sampler.sample([(problem.initial_state, problem.goal)], 1)[0][0]
    dead_buffer = rl.PrioritizedReplayBuffer(10)
    hindsight_buffer = rl.PrioritizedReplayBuffer(10)
    for buffer, count in [(dead_buffer, dead_count), (hindsight_buffer, hindsight_count)]:
        for _ in range(count):
            buffer.push(transition)
    loss = RecordingOptimization()
    algorithm = make_algorithm(problem, sampler, loss, dead_buffer, hindsight_buffer)
    algorithm.optimize_model(batch_size)
    assert [len(batch) for batch in loss.batches] == ([batch_size] if dead_count + hindsight_count else [])


@pytest.mark.parametrize("initially_solved", [False, True])
def test_solutions_skip_dead_end_detection(problem, initially_solved, monkeypatch):
    def detector(state, goal):
        pytest.fail("A solution trajectory must not invoke dead-end detection")

    reward_function = rl.ConstantRewardFunction(-1.0)
    monkeypatch.setattr(reward_function, "is_dead_end", detector)
    sampler = rl.GreedyPolicyTrajectorySampler(
        PreferenceModel("finish"), reward_function, dead_end_detector=detector,
    )
    goal = problem.ground_condition(problem.fact("ready")) if initially_solved else problem.goal
    trajectory = sampler.sample([(problem.initial_state, goal)], 5)[0]
    assert len(trajectory) == (0 if initially_solved else 2)
    assert trajectory.is_solution()
    assert not trajectory.is_unsolvable()
    assert not any(t.successor_is_dead_end for t in trajectory)


def test_actionless_successor_is_terminal_without_h2():
    domain = mm.Domain.from_pddl('''
        (define (domain actionless) (:requirements :strips)
          (:predicates (ready) (goal))
          (:action lose :parameters () :precondition (ready) :effect (not (ready))))
    ''')
    problem = mm.Problem.from_pddl(domain, '''
        (define (problem p) (:domain actionless) (:init (ready)) (:goal (goal)))
    ''', generator="lifted")
    sampler = rl.GreedyPolicyTrajectorySampler(PreferenceModel(), rl.ConstantRewardFunction(-1.0))
    trajectory = sampler.sample([(problem.initial_state, problem.goal)], 5)[0]
    assert len(trajectory) == 1
    assert trajectory.is_unsolvable()
    assert trajectory[0].successor_is_dead_end
    assert trajectory[0].is_terminal
    assert trajectory[0].immediate_reward == -1.0


def test_detector_cache_keeps_problems_separate(problem):
    other = mm.Problem.from_pddl(problem.domain, '''
        (define (problem other) (:domain dead-regions) (:init (dead0)) (:goal (goal)))
    ''', generator="grounded")
    created = []

    def factory(problem):
        created.append(problem)
        return mm.H2DeadEndDetector(problem)

    detector = rl.CachedDeadEndDetector(factory)
    assert not detector(problem.initial_state, problem.goal)
    assert detector(other.initial_state, other.goal)
    assert not detector(problem.initial_state, problem.goal)
    assert created == [problem, other]
