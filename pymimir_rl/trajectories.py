import pymimir as mm

from .reward_functions import RewardFunction


class Transition:
    def __init__(self,
                 current_state: mm.State,
                 successor_state: mm.State,
                 selected_action: mm.GroundAction,
                 reward: float,
                 future_rewards: float,
                 reward_function: RewardFunction,
                 goal_condition: mm.GroundConjunctiveCondition,
                 part_of_solution: bool) -> None:
        assert current_state.get_problem() == successor_state.get_problem(), "Origin and destination states must belong to the same problem."
        self.current_state = current_state
        self.successor_state = successor_state
        self.selected_action = selected_action
        self.immediate_reward = reward
        self.future_rewards = future_rewards
        self.reward_function = reward_function
        self.goal_condition = goal_condition
        self.part_of_solution = part_of_solution
        self.achieves_goal = goal_condition.holds(successor_state)

    def __str__(self) -> str:
        return f'Transition({str(self.current_state)}, {str(self.selected_action)}, {str(self.successor_state)}, {str(self.goal_condition)})'


class Trajectory:
    def __init__(self,
                 state_sequence: list[mm.State],
                 action_sequence: list[mm.GroundAction],
                 reward_sequence: list[float],
                 reward_function: RewardFunction,
                 goal_condition: mm.GroundConjunctiveCondition):
        assert len(state_sequence) == len(action_sequence) + 1, "State sequence must have one more element than action sequence."
        assert len(action_sequence) == len(reward_sequence), "State sequence and reward sequence must have the same length."
        assert isinstance(goal_condition, mm.GroundConjunctiveCondition), "Goal condition must be a GroundConjunctiveCondition."
        self.problem = state_sequence[0].get_problem()
        self.reward_function = reward_function
        self.transitions: list[Transition] = []
        part_of_solution = goal_condition.holds(state_sequence[-1])
        for idx in range(len(action_sequence)):
            current_state = state_sequence[idx]
            successor_state = state_sequence[idx + 1]
            selected_action = action_sequence[idx]
            reward = reward_sequence[idx]
            future_rewards = sum(reward_sequence[idx + 1:])
            transition = Transition(current_state, successor_state, selected_action, reward, future_rewards, reward_function, goal_condition, part_of_solution)
            assert (idx >= len(action_sequence) - 1) or (not transition.achieves_goal), "The trajectory must terminate on goal states."
            self.transitions.append(transition)

    def __iter__(self):
        return iter(self.transitions)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, index: int) -> Transition:
        return self.transitions[index]

    def is_solution(self) -> bool:
        return self.transitions[-1].achieves_goal

    def clone_with_goal(self, start_index_incl: int, end_index_incl: int, goal_condition: mm.GroundConjunctiveCondition) -> 'Trajectory':
        assert start_index_incl >= 0 and end_index_incl < len(self.transitions), "Indices must be within the bounds of the trajectory."
        assert start_index_incl <= end_index_incl, "Start index must be less than or equal to end index."
        assert goal_condition.get_problem() == self.problem, "Goal condition must belong to the same problem as the trajectory."
        assert isinstance(goal_condition, mm.GroundConjunctiveCondition), "Goal condition must be a GroundConjunctiveCondition."
        cloned_state_sequence: list[mm.State] = []
        cloned_action_sequence: list[mm.GroundAction] = []
        cloned_reward_sequence: list[float] = []
        for transition_index in range(start_index_incl, end_index_incl + 1):
            transition = self.transitions[transition_index]
            current_state = transition.current_state
            selected_action = transition.selected_action
            successor_state = transition.successor_state
            reward = self.reward_function(current_state, selected_action, successor_state, goal_condition)
            cloned_state_sequence.append(current_state)
            cloned_action_sequence.append(selected_action)
            cloned_reward_sequence.append(reward)
        cloned_state_sequence.append(self.transitions[end_index_incl].successor_state)
        return Trajectory(cloned_state_sequence, cloned_action_sequence, cloned_reward_sequence, self.reward_function, goal_condition)

    def validate(self) -> None:
        """
        Validates the trajectory. Raises an AssertionError if any validation fails.
        """
        last_transition = self.transitions[-1]
        goal_condition = last_transition.goal_condition
        if goal_condition.holds(last_transition.successor_state):
            assert last_transition.achieves_goal
        if last_transition.achieves_goal:
            assert goal_condition.holds(last_transition.successor_state)
        for i in range(len(self.transitions) - 1):
            assert self.transitions[i].successor_state == self.transitions[i + 1].current_state
        for transition in self.transitions:
            assert isinstance(transition, Transition)
            assert isinstance(transition.goal_condition, mm.GroundConjunctiveCondition)
            assert transition.current_state.get_problem() == self.problem
            assert transition.successor_state.get_problem() == self.problem
            actions = transition.current_state.generate_applicable_actions()
            assert transition.selected_action in actions
            assert transition.selected_action.get_precondition().holds(transition.current_state)
            assert transition.selected_action.apply(transition.current_state) == transition.successor_state
            assert not transition.goal_condition.holds(transition.current_state)
        if len(self.transitions) > 0:
            expected_rewards = sum(transition.immediate_reward for transition in self.transitions)
            actual_rewards = self.transitions[0].immediate_reward + self.transitions[0].future_rewards
            assert actual_rewards == expected_rewards, "Expected and actual rewards must not differ."


    def __str__(self) -> str:
        return '[' + str.join(', ', [str(transition) for transition in self.transitions]) + ']'
