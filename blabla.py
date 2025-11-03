import numpy as np

from orig_solve_bvp import solve_bvp
from solve_bvp_s45 import solve_bvp as solve_bvp45

x = np.linspace(0, 1, 11, dtype=float)
y = np.ones((2, len(x)), dtype=float)

def deriv(x, y):
    dz, u = y
    d_dz = -10 * u
    d_u = dz + u
    
    return np.array([d_dz, d_u], dtype=float)

def bc(ya, yb):
    dza, ua = ya
    dzb, ub = yb
    return np.array([dza - 2, dzb + ub], dtype=float)

np.set_printoptions(
    linewidth=200,   # allow long lines before wrapping
    precision=2,     # show only two decimals
    floatmode='maxprec',  # optional: consistent formatting
    # suppress=True    # avoid scientific notation for small numbers
)

sol = solve_bvp45(deriv, bc, x, y, tol=1e-6, max_nodes=2000, verbose=2)

