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
The former receives each trajectory containing a proven dead state, from the
index returned by the optional `dead_end_cutoff(trajectory)` callable onwards.
Without a callable, the whole trajectory is stored. Each optimization step
samples `batch_size` hindsight transitions and adds up to `dead_end_batch_size`
dead-end transitions (default `batch_size // 2`, at most one per stored
transition), passing both to the loss as one batch.

The cutoff only selects replay experience; it does not create dead-end labels
or change targets. All sampled trajectories remain intact for hindsight
refinement. Unproven horizon cutoffs do not qualify for the dead-end buffer.
