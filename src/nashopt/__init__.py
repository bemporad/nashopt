"""
NashOpt - A Python Library for Solving Generalized Nash Equilibrium (GNE) and Game-Design Problems.

(C) 2025-2026 Alberto Bemporad
"""

from .nonlinear.gnep_base import GNEP
from .nonlinear.gnep_parametric import ParametricGNEP
from .lq.gnep_lq import GNEP_LQ
from .control.gnep_lqr import NashLQR
from .control.gnep_linmpc import NashLinearMPC

__all__ = ["GNEP", "ParametricGNEP", "GNEP_LQ", "NashLQR", "NashLinearMPC"]