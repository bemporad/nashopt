# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Optinal dependencies for NashOpt. 
#
# (C) 2025-2026 Alberto Bemporad

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