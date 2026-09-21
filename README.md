# Mimir-RL

Mimir-RL is a Python library that implements RL algorithms using PyTorch and PyTorch RL that are tightly integrated with Mimir.

## Dead-end detection

Pass an optional state/goal detector callable to a trajectory sampler:

```python
import pymimir as mm
from pymimir_rl import BoltzmannTrajectorySampler, CachedDeadEndDetector

detector = CachedDeadEndDetector(mm.H2DeadEndDetector)
sampler = BoltzmannTrajectorySampler(
    model, reward_function, temperature=0.5, dead_end_detector=detector,
)
```

For h², construct problems with `generator="grounded"`. The cache creates one
native detector per problem. `Trajectory` calls it after sampling, in state order,
and propagates each proof forward for the same goal. Existing reward-function
proofs and actionless non-goal states also provide dead-end evidence.

Detected states with applicable actions do not terminate or prune rollouts.
`Transition.successor_is_dead_end` is separate from `is_terminal`, and ordinary
rewards are preserved. Optimizers assign fixed dead-end targets without
bootstrapping from those successors. Hindsight cloning recomputes labels for its
new goal using the same detector cache.

`OffPolicyAlgorithm` accepts `dead_end_replay_buffer` and `hindsight_replay_buffer`.
The former receives a suffix of each trajectory containing a proven dead state.
By default it starts with the transition entering the first proven dead state.
An earlier state whose recorded maximum Q-value is at most
`-10000 * dead_end_q_factor` moves the cutoff to the transition entering that
state. `dead_end_q_factor` defaults to `0.25` (threshold `-2500`) and must be in
`(0, 1]`. Nonfinite predictions do not move the cutoff. Selection uses the
recorded maximum over all applicable actions and requires no additional inference.

The Q-based cutoff only selects replay experience; it does not create dead-end
labels or change targets. All sampled trajectories remain intact for hindsight
refinement. Unproven horizon cutoffs do not qualify for the dead-end buffer,
even if their Q-values are low.
