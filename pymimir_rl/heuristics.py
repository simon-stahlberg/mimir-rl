# import pymimir as mm

# from pyswip import Prolog


# class LiftedFFHeuristic(mm.Heuristic):
#     def __init__(self, problem: mm.Problem, debug_mode: bool = False) -> None:
#         super().__init__()
#         self.problem = problem
#         self.debug_mode = debug_mode
#         self.rules = self._compute_rules()
#         self.goal = self._compute_query(self.problem.get_goal_condition())

#     def _to_prolog_term_name(self, name: str) -> str:
#         return name.removeprefix('?').upper() if name.startswith('?') else name.lower()

#     def _to_prolog_string(self, atom: mm.Atom | mm.GroundAtom) -> str:
#         predicate_name = atom.get_predicate().get_name()
#         term_names = [self._to_prolog_term_name(term.get_name()) for term in atom.get_terms()]
#         return f'{predicate_name}({str.join(', ', term_names)})'

#     def _compute_rules(self) -> list[str]:
#         rules: list[str] = []
#         for action in self.problem.get_domain().get_actions():
#             # Compute the body.
#             body = [self._to_prolog_string(literal.get_atom()) for literal in action.get_precondition().get_literals() if literal.get_polarity()]
#             # Compute the heads.
#             heads: list[str] = []
#             for conditional_effect in action.get_conditional_effect():
#                 assert len(conditional_effect.get_condition().get_parameters()) == 0, "Lifted FF does not support conditional effects with parameters."
#                 heads.extend([self._to_prolog_string(literal.get_atom()) for literal in conditional_effect.get_effect().get_literals() if literal.get_polarity()])
#             # Combine all heads with the body.
#             for head in heads:
#                 rules.append(f'{head} :- {str.join(', ', body)}')
#         return rules

#     def _compute_query(self, goal_condition: mm.GroundConjunctiveCondition) -> str:
#         body = [self._to_prolog_string(literal.get_atom()) for literal in goal_condition if literal.get_polarity()]
#         return f'has_relaxed_solution() :- {str.join(', ', body)}'

#     def compute_value(self, state: mm.State, is_goal_state: bool) -> float:
#         facts: list[str] = []
#         for atom in state.get_atoms():
#             predicate_name = atom.get_predicate().get_name().lower()
#             term_names = [obj.get_name() for obj in atom.get_terms()]
#             facts.append(f'{predicate_name}({str.join(', ', term_names)})')
#         # Write program to Prolog.
#         pl = Prolog()  # TODO: retractall to clear?
#         for fact in facts:
#             pl.assertz(fact, catcherrors=self.debug_mode)
#             print(fact)
#         for rule in self.rules:
#             pl.assertz(rule, catcherrors=self.debug_mode)
#             print(rule)
#         pl.assertz(self.goal)
#         print(self.goal)
#         results = list(pl.query('has_relaxed_solution()', maxresult=1))
#         pass
#         raise NotImplementedError()
