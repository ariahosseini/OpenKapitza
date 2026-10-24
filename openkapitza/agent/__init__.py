"""Agent layer (Phase 3): a plain-Python core that guides a user through a
joint calculation, with a thin conversational wrapper on top.

    fields    the input schema: questions, reasons, units, defaults
    intake    answers -> validated build_joint arguments
    planner   after a run: is MD worth it, is there a route, what to measure next
    explain   the plain-language answer
    session   the conversation as text-in / text-out (front-end agnostic)

Terminal: python -m openkapitza.agent  (see --help)
"""

from .explain import explain
from .fields import FIELDS, Field, FieldError
from .intake import Intake
from .planner import MD_ROUTES, Plan, plan
from .session import Session

__all__ = ["FIELDS", "Field", "FieldError", "Intake", "MD_ROUTES", "Plan", "Session",
           "explain", "plan"]
