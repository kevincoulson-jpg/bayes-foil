# Hydrofoil Optimization Problem Configuration

# Geometric parameter bounds
M_BOUNDS = (0.00, 0.06)   # Max camber (m)
P_BOUNDS = (0.10, 0.60)   # Position of max camber (p)
T_BOUNDS = (0.06, 0.20)   # Thickness (t)
ALPHA_BOUNDS = (2.0, 6.0)  # Angle of attack range (degrees)

# Performance constraint
MIN_CL = 0.5              # Minimum required Cl at design alpha

# Design angle of attack (deg)
DESIGN_ALPHA = 5.0

# Objective functions
# 1. Maximize Cl/Cd (lift-to-drag ratio)
# 2. Minimize t (thickness)

def is_feasible(m, p, t, cl=None):
    """
    Check if a candidate design is within geometric bounds and (optionally) meets performance constraints.
    """
    if not (M_BOUNDS[0] <= m <= M_BOUNDS[1]):
        return False
    if not (P_BOUNDS[0] <= p <= P_BOUNDS[1]):
        return False
    if not (T_BOUNDS[0] <= t <= T_BOUNDS[1]):
        return False
    if cl is not None and cl < MIN_CL:
        return False
    return True 