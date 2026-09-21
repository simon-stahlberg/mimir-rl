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
The former receives full trajectories that contain a proven dead state, including
exploration after that state. All sampled trajectories remain available for
hindsight refinement. An unproven horizon cutoff alone does not qualify for the
dead-end buffer.
