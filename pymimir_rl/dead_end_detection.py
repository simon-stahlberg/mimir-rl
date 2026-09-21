from collections.abc import Callable

import pymimir as mm


class CachedDeadEndDetector:
    """Reuse one native detector per problem across trajectories and goals."""

    def __init__(self, factory: Callable[[mm.Problem], mm.DeadEndDetector]) -> None:
        self.factory = factory
        self.detectors: dict[mm.Problem, mm.DeadEndDetector] = {}

    def __call__(self, state: mm.State, goal: mm.GroundConjunctiveCondition) -> bool:
        if state.problem not in self.detectors:
            self.detectors[state.problem] = self.factory(state.problem)
        return self.detectors[state.problem].is_dead_end(state, goal)
