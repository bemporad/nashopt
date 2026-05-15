# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Optional dependencies for NashOpt. 
#
# (C) 2025-2026 Alberto Bemporad

import numpy as np

def get_gurobi():
    try:
        import gurobipy as gp
    except ImportError:
        gp = None
    return gp

def get_highspy():
    try:
        import highspy
    except ImportError:
        highspy = None
    return highspy

def add_box_constraints(nvar, A=None, b=None, lb=None, ub=None):
    """Add box constraints to the inequality constraint matrix A and vector b, given lower and upper bounds lb and ub. The box constraints are added in the form of additional rows to A and b, with the appropriate signs to represent the inequalities. The function also removes any rows corresponding to infinite bounds.
    """    
    if lb is None and ub is None:
        return A, b
    
    idx_lb = np.where(np.isfinite(lb))[0]
    n_idx_lb = len(idx_lb)
    idx_ub = np.where(np.isfinite(ub))[0]
    n_idx_ub = len(idx_ub)
    
    if n_idx_lb == 0 and n_idx_ub == 0:
        return A, b
    
    AA = A.copy() if A is not None else np.zeros((0, nvar))
    bb = b.copy() if b is not None else np.zeros(0)
    
    if lb is not None:
        if n_idx_lb > 0:
            rows = np.zeros((n_idx_lb, nvar))
            rows[np.arange(n_idx_lb), idx_lb] = 1.0
            AA = np.vstack((AA, -rows))
            bb = np.concatenate((bb, -lb[idx_lb]))
    if ub is not None:
        if n_idx_ub > 0:
            rows = np.zeros((n_idx_ub, nvar))
            rows[np.arange(n_idx_ub), idx_ub] = 1.0
            AA = np.vstack((AA, rows))
            bb = np.concatenate((bb, ub[idx_ub]))
    return AA, bb
