"""Boundary value problem solver."""
from warnings import warn

import numpy as np
from numpy.linalg import pinv

from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult


EPS = np.finfo(float).eps


def estimate_fun_jac(fun, x, y, p, f0=None):
    """Estimate derivatives of an ODE system rhs with forward differences.

    Returns
    -------
    df_dy : ndarray, shape (n, n, m)
        Derivatives with respect to y. An element (i, j, q) corresponds to
        d f_i(x_q, y_q) / d (y_q)_j.
    df_dp : ndarray with shape (n, k, m) or None
        Derivatives with respect to p. An element (i, j, q) corresponds to
        d f_i(x_q, y_q, p) / d p_j. If `p` is empty, None is returned.
    """
    n, m = y.shape
    if f0 is None:
        f0 = fun(x, y, p)

    dtype = y.dtype

    df_dy = np.empty((n, n, m), dtype=dtype)
    h = EPS**0.5 * (1 + np.abs(y))
    for i in range(n):
        y_new = y.copy()
        y_new[i] += h[i]
        hi = y_new[i] - y[i]
        f_new = fun(x, y_new, p)
        df_dy[:, i, :] = (f_new - f0) / hi

    k = p.shape[0]
    if k == 0:
        df_dp = None
    else:
        df_dp = np.empty((n, k, m), dtype=dtype)
        h = EPS**0.5 * (1 + np.abs(p))
        for i in range(k):
            p_new = p.copy()
            p_new[i] += h[i]
            hi = p_new[i] - p[i]
            f_new = fun(x, y, p_new)
            df_dp[:, i, :] = (f_new - f0) / hi

    return df_dy, df_dp


def estimate_bc_jac(bc, ya, yb, p, bc0=None):
    """Estimate derivatives of boundary conditions with forward differences.

    Returns
    -------
    dbc_dya : ndarray, shape (n + k, n)
        Derivatives with respect to ya. An element (i, j) corresponds to
        d bc_i / d ya_j.
    dbc_dyb : ndarray, shape (n + k, n)
        Derivatives with respect to yb. An element (i, j) corresponds to
        d bc_i / d ya_j.
    dbc_dp : ndarray with shape (n + k, k) or None
        Derivatives with respect to p. An element (i, j) corresponds to
        d bc_i / d p_j. If `p` is empty, None is returned.
    """
    n = ya.shape[0]
    k = p.shape[0]

    if bc0 is None:
        bc0 = bc(ya, yb, p)

    dtype = ya.dtype

    dbc_dya = np.empty((n, n + k), dtype=dtype)
    h = EPS**0.5 * (1 + np.abs(ya))
    for i in range(n):
        ya_new = ya.copy()
        ya_new[i] += h[i]
        hi = ya_new[i] - ya[i]
        bc_new = bc(ya_new, yb, p)
        dbc_dya[i] = (bc_new - bc0) / hi
    dbc_dya = dbc_dya.T

    h = EPS**0.5 * (1 + np.abs(yb))
    dbc_dyb = np.empty((n, n + k), dtype=dtype)
    for i in range(n):
        yb_new = yb.copy()
        yb_new[i] += h[i]
        hi = yb_new[i] - yb[i]
        bc_new = bc(ya, yb_new, p)
        dbc_dyb[i] = (bc_new - bc0) / hi
    dbc_dyb = dbc_dyb.T

    if k == 0:
        dbc_dp = None
    else:
        h = EPS**0.5 * (1 + np.abs(p))
        dbc_dp = np.empty((k, n + k), dtype=dtype)
        for i in range(k):
            p_new = p.copy()
            p_new[i] += h[i]
            hi = p_new[i] - p[i]
            bc_new = bc(ya, yb, p_new)
            dbc_dp[i] = (bc_new - bc0) / hi
        dbc_dp = dbc_dp.T

    return dbc_dya, dbc_dyb, dbc_dp


def compute_jac_indices(n, m, k):
    """Compute indices for the collocation system Jacobian construction.

    See `construct_global_jac` for the explanation.
    """
    i_col = np.repeat(np.arange((m - 1) * n), n)
    j_col = (np.tile(np.arange(n), n * (m - 1)) +
             np.repeat(np.arange(m - 1) * n, n**2))

    i_bc = np.repeat(np.arange((m - 1) * n, m * n + k), n)
    j_bc = np.tile(np.arange(n), n + k)

    i_p_col = np.repeat(np.arange((m - 1) * n), k)
    j_p_col = np.tile(np.arange(m * n, m * n + k), (m - 1) * n)

    i_p_bc = np.repeat(np.arange((m - 1) * n, m * n + k), k)
    j_p_bc = np.tile(np.arange(m * n, m * n + k), n + k)

    i = np.hstack((i_col, i_col, i_bc, i_bc, i_p_col, i_p_bc))
    j = np.hstack((j_col, j_col + n,
                   j_bc, j_bc + (m - 1) * n,
                   j_p_col, j_p_bc))

    return i, j


def stacked_matmul(a, b):
    """Stacked matrix multiply: out[i,:,:] = np.dot(a[i,:,:], b[i,:,:]).

    Empirical optimization. Use outer Python loop and BLAS for large
    matrices, otherwise use a single einsum call.
    """
    if a.shape[1] > 50:
        out = np.empty((a.shape[0], a.shape[1], b.shape[2]))
        for i in range(a.shape[0]):
            out[i] = np.dot(a[i], b[i])
        return out
    else:
        return np.einsum('...ij,...jk->...ik', a, b)


def construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dp, dbc_dya, dbc_dyb, dbc_dp):
    """Construct the Jacobian of the collocation system.

    There are n * m + k functions: m - 1 collocations residuals, each
    containing n components, followed by n + k boundary condition residuals.

    There are n * m + k variables: m vectors of y, each containing n
    components, followed by k values of vector p.

    For example, let m = 4, n = 2 and k = 1, then the Jacobian will have
    the following sparsity structure:

        1 1 2 2 0 0 0 0  5
        1 1 2 2 0 0 0 0  5
        0 0 1 1 2 2 0 0  5
        0 0 1 1 2 2 0 0  5
        0 0 0 0 1 1 2 2  5
        0 0 0 0 1 1 2 2  5

        3 3 0 0 0 0 4 4  6
        3 3 0 0 0 0 4 4  6
        3 3 0 0 0 0 4 4  6

    Zeros denote identically zero values, other values denote different kinds
    of blocks in the matrix (see below). The blank row indicates the separation
    of collocation residuals from boundary conditions. And the blank column
    indicates the separation of y values from p values.

    Refer to [1]_  (p. 306) for the formula of n x n blocks for derivatives
    of collocation residuals with respect to y.

    Parameters
    ----------
    n : int
        Number of equations in the ODE system.
    m : int
        Number of nodes in the mesh.
    k : int
        Number of the unknown parameters.
    i_jac, j_jac : ndarray
        Row and column indices returned by `compute_jac_indices`. They
        represent different blocks in the Jacobian matrix in the following
        order (see the scheme above):

            * 1: m - 1 diagonal n x n blocks for the collocation residuals.
            * 2: m - 1 off-diagonal n x n blocks for the collocation residuals.
            * 3 : (n + k) x n block for the dependency of the boundary
              conditions on ya.
            * 4: (n + k) x n block for the dependency of the boundary
              conditions on yb.
            * 5: (m - 1) * n x k block for the dependency of the collocation
              residuals on p.
            * 6: (n + k) x k block for the dependency of the boundary
              conditions on p.

    df_dy : ndarray, shape (n, n, m)
        Jacobian of f with respect to y computed at the mesh nodes.
    df_dy_middle : ndarray, shape (n, n, m - 1)
        Jacobian of f with respect to y computed at the middle between the
        mesh nodes.
    df_dp : ndarray with shape (n, k, m) or None
        Jacobian of f with respect to p computed at the mesh nodes.
    df_dp_middle : ndarray with shape (n, k, m - 1) or None
        Jacobian of f with respect to p computed at the middle between the
        mesh nodes.
    dbc_dya, dbc_dyb : ndarray, shape (n, n)
        Jacobian of bc with respect to ya and yb.
    dbc_dp : ndarray with shape (n, k) or None
        Jacobian of bc with respect to p.

    Returns
    -------
    J : csc_matrix, shape (n * m + k, n * m + k)
        Jacobian of the collocation system in a sparse form.

    References
    ----------
    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
       Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
       Number 3, pp. 299-316, 2001.
    """
    df_dy = np.transpose(df_dy, (2, 0, 1)) # move m-axis first, keep others in place

    h = h[:, np.newaxis, np.newaxis]

    dtype = df_dy.dtype

    # Computing diagonal n x n blocks.
    # dPhi_dy_0 = np.empty((m - 1, n, n), dtype=dtype)
    # dPhi_dy_0[:] = -np.identity(n)
    # dPhi_dy_0 -= h / 6 * (df_dy[:-1] + 2 * df_dy_middle)
    # dPhi_dy_0 -= h**2 / 12 * stacked_matmul(df_dy_middle, df_dy[:-1])
    dPhi_dy_0 = np.empty((m-1, n, n), dtype=dtype)
    dPhi_dy_0[:] = -np.identity(n)
    dPhi_dy_0 -= h / 2 * df_dy[:-1]

    # Computing off-diagonal n x n blocks.
    # dPhi_dy_1 = np.empty((m - 1, n, n), dtype=dtype)
    # dPhi_dy_1[:] = np.identity(n)
    # dPhi_dy_1 -= h / 6 * (df_dy[1:] + 2 * df_dy_middle)
    # dPhi_dy_1 += h**2 / 12 * stacked_matmul(df_dy_middle, df_dy[1:])
    dPhi_dy_1 = np.empty((m - 1, n, n), dtype=dtype)
    dPhi_dy_1[:] = np.identity(n)
    dPhi_dy_1 -= h / 2 * df_dy[1:]

    values = np.hstack((dPhi_dy_0.ravel(), dPhi_dy_1.ravel(), dbc_dya.ravel(),
                        dbc_dyb.ravel()))

    J = coo_matrix((values, (i_jac, j_jac)))
    return csc_matrix(J)


def collocation_fun(fun, y, p, x, h):
    """Evaluate collocation residuals.

    This function lies in the core of the method. The solution is sought
    as a cubic C1 continuous spline with derivatives matching the ODE rhs
    at given nodes `x`. Collocation conditions are formed from the equality
    of the spline derivatives and rhs of the ODE system in the middle points
    between nodes.

    Such method is classified to Lobbato IIIA family in ODE literature.
    Refer to [1]_ for the formula and some discussion.

    Returns
    -------
    col_res : ndarray, shape (n, m - 1)
        Collocation residuals at the middle points of the mesh intervals.
    y_middle : ndarray, shape (n, m - 1)
        Values of the cubic spline evaluated at the middle points of the mesh
        intervals.
    f : ndarray, shape (n, m)
        RHS of the ODE system evaluated at the mesh nodes.
    f_middle : ndarray, shape (n, m - 1)
        RHS of the ODE system evaluated at the middle points of the mesh
        intervals (and using `y_middle`).

    """

    f = fun(x, y, p)
    y_ = y[:, :-1] + h / 2 * (f[:, :-1] + f[:, 1:]) 
    col_res = y[:, 1:] - y_

    return col_res, f


def prepare_sys(n, m, k, fun, bc, fun_jac, bc_jac, x, h):
    """Create the function and the Jacobian for the collocation system."""
    # x_middle = x[:-1] + 0.5 * h
    i_jac, j_jac = compute_jac_indices(n, m, k)

    def col_fun(y, p):
        return collocation_fun(fun, y, p, x, h)

    def sys_jac(y, p, f, bc0):
        
        df_dy, df_dp = estimate_fun_jac(fun, x, y, p, f)

        dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(bc, y[:, 0], y[:, -1], p, bc0)

        return construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dp, dbc_dya, dbc_dyb, dbc_dp)

    return col_fun, sys_jac


def solve_newton(n, m, h, col_fun, bc, jac, y, p, B, bvp_tol, bc_tol):
    """Solve the nonlinear collocation system by a Newton method.

    This is a simple Newton method with a backtracking line search. As
    advised in [1]_, an affine-invariant criterion function F = ||J^-1 r||^2
    is used, where J is the Jacobian matrix at the current iteration and r is
    the vector or collocation residuals (values of the system lhs).

    The method alters between full Newton iterations and the fixed-Jacobian
    iterations based

    There are other tricks proposed in [1]_, but they are not used as they
    don't seem to improve anything significantly, and even break the
    convergence on some test problems I tried.

    All important parameters of the algorithm are defined inside the function.

    Parameters
    ----------
    n : int
        Number of equations in the ODE system.
    m : int
        Number of nodes in the mesh.
    h : ndarray, shape (m-1,)
        Mesh intervals.
    col_fun : callable
        Function computing collocation residuals.
    bc : callable
        Function computing boundary condition residuals.
    jac : callable
        Function computing the Jacobian of the whole system (including
        collocation and boundary condition residuals). It is supposed to
        return csc_matrix.
    y : ndarray, shape (n, m)
        Initial guess for the function values at the mesh nodes.
    p : ndarray, shape (k,)
        Initial guess for the unknown parameters.
    B : ndarray with shape (n, n) or None
        Matrix to force the S y(a) = 0 condition for a problems with the
        singular term. If None, the singular term is assumed to be absent.
    bvp_tol : float
        Tolerance to which we want to solve a BVP.
    bc_tol : float
        Tolerance to which we want to satisfy the boundary conditions.

    Returns
    -------
    y : ndarray, shape (n, m)
        Final iterate for the function values at the mesh nodes.
    p : ndarray, shape (k,)
        Final iterate for the unknown parameters.
    singular : bool
        True, if the LU decomposition failed because Jacobian turned out
        to be singular.

    References
    ----------
    .. [1]  U. Ascher, R. Mattheij and R. Russell "Numerical Solution of
       Boundary Value Problems for Ordinary Differential Equations",
       Philidelphia, PA: Society for Industrial and Applied Mathematics,
       1995.
    """
    # We know that the solution residuals at the middle points of the mesh
    # are connected with collocation residuals  r_middle = 1.5 * col_res / h.
    # As our BVP solver tries to decrease relative residuals below a certain
    # tolerance, it seems reasonable to terminated Newton iterations by
    # comparison of r_middle / (1 + np.abs(f_middle)) with a certain threshold,
    # which we choose to be 1.5 orders lower than the BVP tolerance. We rewrite
    # the condition as col_res < tol_r * (1 + np.abs(f_middle)), then tol_r
    # should be computed as follows:
    tol_r = 2/3 * h * 5e-2 * bvp_tol

    # Maximum allowed number of Jacobian evaluation and factorization, in
    # other words, the maximum number of full Newton iterations. A small value
    # is recommended in the literature.
    max_njev = 4

    # Maximum number of iterations, considering that some of them can be
    # performed with the fixed Jacobian. In theory, such iterations are cheap,
    # but it's not that simple in Python.
    max_iter = 8

    # Minimum relative improvement of the criterion function to accept the
    # step (Armijo constant).
    sigma = 0.2

    # Step size decrease factor for backtracking.
    tau = 0.5

    # Maximum number of backtracking steps, the minimum step is then
    # tau ** n_trial.
    n_trial = 4

    col_res, f = col_fun(y, p)
    bc_res = bc(y[:, 0], y[:, -1], p)
    res = np.hstack((col_res.ravel(order='F'), bc_res))

    njev = 0
    singular = False
    recompute_jac = True
    # print("Starting a Newton solver")
    for iteration in range(max_iter):
        # print("Starting a Newton solver iteration")
        if recompute_jac:
            J = jac(y, p, f, bc_res)
            njev += 1
            try:
                LU = splu(J)
            except RuntimeError:
                singular = True
                break

            step = LU.solve(res)
            cost = np.dot(step, step) # norm of the step

        y_step = step[:m * n].reshape((n, m), order='F')
        p_step = step[m * n:]

        alpha = 1
        for trial in range(n_trial + 1):
            y_new = y - alpha * y_step
            if B is not None:
                y_new[:, 0] = np.dot(B, y_new[:, 0])
            p_new = p - alpha * p_step

            col_res, f = col_fun(y_new, p_new)
            bc_res = bc(y_new[:, 0], y_new[:, -1], p_new)
            res = np.hstack((col_res.ravel(order='F'), bc_res))

            step_new = LU.solve(res)
            cost_new = np.dot(step_new, step_new)
            if cost_new < (1 - 2 * alpha * sigma) * cost:
                break

            if trial < n_trial:
                alpha *= tau

        y = y_new
        p = p_new

        if njev == max_njev:
            break

        if np.all(np.abs(bc_res) < bc_tol):
            break

        # If the full step was taken, then we are going to continue with
        # the same Jacobian. This is the approach of BVP_SOLVER.
        if alpha == 1:
            step = step_new
            cost = cost_new
            recompute_jac = False
        else:
            recompute_jac = True

    return y, p, singular


def print_iteration_header():
    print(f"{'Iteration':^15}{'Max residual':^15}{'Max BC residual':^15}"
          f"{'Total nodes':^15}{'Nodes added':^15}")


def print_iteration_progress(iteration, residual, bc_residual, total_nodes, nodes_added):
    print(f"{iteration:^15}{residual:^15.2e}{bc_residual:^15.2e}"
          f"{total_nodes:^15}{nodes_added:^15}")


class BVPResult(OptimizeResult):
    pass


TERMINATION_MESSAGES = {
    0: "The algorithm converged to the desired accuracy.",
    1: "The maximum number of mesh nodes is exceeded.",
    2: "A singular Jacobian encountered when solving the collocation system.",
    3: "The solver was unable to satisfy boundary conditions tolerance on iteration 10."
}


def estimate_rms_residuals(fun, sol, x, h, p):
    """Estimate rms values of collocation residuals using Lobatto quadrature.

    The residuals are defined as the difference between the derivatives of
    our solution and rhs of the ODE system. We use relative residuals, i.e.,
    normalized by 1 + np.abs(f). RMS values are computed as sqrt from the
    normalized integrals of the squared relative residuals over each interval.
    Integrals are estimated using 5-point Lobatto quadrature [1]_, we use the
    fact that residuals at the mesh nodes are identically zero.

    In [2] they don't normalize integrals by interval lengths, which gives
    a higher rate of convergence of the residuals by the factor of h**0.5.
    I chose to do such normalization for an ease of interpretation of return
    values as RMS estimates.

    Returns
    -------
    rms_res : ndarray, shape (m - 1,)
        Estimated rms values of the relative residuals over each interval.

    References
    ----------
    .. [1] http://mathworld.wolfram.com/LobattoQuadrature.html
    .. [2] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
       Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
       Number 3, pp. 299-316, 2001.
    """
    x_middle = x[:-1] + 0.5 * h

    y_middle = sol(x_middle)
    y_middle_prime = sol(x_middle, 1)
    f_middle = fun(x_middle, y_middle, p)
    r_middle = f_middle - y_middle_prime

    s = 0.5 * h * (3/7)**0.5
    x1 = x_middle + s
    x2 = x_middle - s
    y1 = sol(x1)
    y2 = sol(x2)
    y1_prime = sol(x1, 1)
    y2_prime = sol(x2, 1)
    f1 = fun(x1, y1, p)
    f2 = fun(x2, y2, p)
    r1 = y1_prime - f1
    r2 = y2_prime - f2

    r_middle /= 1 + np.abs(f_middle)
    r1 /= 1 + np.abs(f1)
    r2 /= 1 + np.abs(f2)

    r1 = np.sum(np.real(r1 * np.conj(r1)), axis=0)
    r2 = np.sum(np.real(r2 * np.conj(r2)), axis=0)
    r_middle = np.sum(np.real(r_middle * np.conj(r_middle)), axis=0)

    return (0.5 * (32 / 45 * r_middle + 49 / 90 * (r1 + r2))) ** 0.5


def create_spline(y, yp, x, h):
    """Create a cubic spline given values and derivatives.

    Formulas for the coefficients are taken from interpolate.CubicSpline.

    Returns
    -------
    sol : PPoly
        Constructed spline as a PPoly instance.
    """
    from scipy.interpolate import PPoly

    n, m = y.shape
    c = np.empty((4, n, m - 1), dtype=y.dtype)
    slope = (y[:, 1:] - y[:, :-1]) / h
    t = (yp[:, :-1] + yp[:, 1:] - 2 * slope) / h
    c[0] = t / h
    c[1] = (slope - yp[:, :-1]) / h - t
    c[2] = yp[:, :-1]
    c[3] = y[:, :-1]
    c = np.moveaxis(c, 1, 0)

    return PPoly(c, x, extrapolate=True, axis=1)


def modify_mesh(x, insert_1, insert_2):
    # just insert two where rtol > 100 and 1 where rtol > 1
    """Insert nodes into a mesh.

    Nodes removal logic is not established, its impact on the solver is
    presumably negligible. So, only insertion is done in this function.

    Parameters
    ----------
    x : ndarray, shape (m,)
        Mesh nodes.
    insert_1 : ndarray
        Intervals to each insert 1 new node in the middle.
    insert_2 : ndarray
        Intervals to each insert 2 new nodes, such that divide an interval
        into 3 equal parts.

    Returns
    -------
    x_new : ndarray
        New mesh nodes.

    Notes
    -----
    `insert_1` and `insert_2` should not have common values.
    """
    # Because np.insert implementation apparently varies with a version of
    # NumPy, we use a simple and reliable approach with sorting.
    return np.sort(np.hstack((
        x,
        0.5 * (x[insert_1] + x[insert_1 + 1]),
        (2 * x[insert_2] + x[insert_2 + 1]) / 3,
        (x[insert_2] + 2 * x[insert_2 + 1]) / 3
    )))


def wrap_functions(fun, bc, fun_jac, bc_jac, k, a, S, D, dtype):
    """Wrap functions for unified usage in the solver."""
    if fun_jac is None:
        fun_jac_wrapped = None

    if bc_jac is None:
        bc_jac_wrapped = None

    print("k")
    print(k)

    if k == 0:
        def fun_p(x, y, _):
            return np.asarray(fun(x, y), dtype)

        def bc_wrapped(ya, yb, _):
            return np.asarray(bc(ya, yb), dtype)

    if S is None:
        fun_wrapped = fun_p

    return fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped


def solve_bvp(fun, bc, x, y, p=None, S=None, fun_jac=None, bc_jac=None, tol=1e-3, max_nodes=1000, verbose=0, bc_tol=None):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("`x` must be 1 dimensional.")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("`x` must be strictly increasing.")
    a = x[0]

    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.complexfloating):
        dtype = complex
    else:
        dtype = float
    y = y.astype(dtype, copy=False)


    if y.ndim != 2:
        raise ValueError("`y` must be 2 dimensional.")
    if y.shape[1] != x.shape[0]:
        raise ValueError(f"`y` is expected to have {x.shape[0]} columns, but actually "
                         f"has {y.shape[1]}.")

    if p is None:
        p = np.array([])
    else:
        p = np.asarray(p, dtype=dtype)
    if p.ndim != 1:
        raise ValueError("`p` must be 1 dimensional.")

    if tol < 100 * EPS:
        warn(f"`tol` is too low, setting to {100 * EPS:.2e}", stacklevel=2)
        tol = 100 * EPS

    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")

    n = y.shape[0]
    k = p.shape[0]

    if S is not None:
        S = np.asarray(S, dtype=dtype)
        if S.shape != (n, n):
            raise ValueError(f"`S` is expected to have shape {(n, n)}, "
                             f"but actually has {S.shape}")

        # Compute I - S^+ S to impose necessary boundary conditions.
        B = np.identity(n) - np.dot(pinv(S), S)

        y[:, 0] = np.dot(B, y[:, 0])

        # Compute (I - S)^+ to correct derivatives at x=a.
        D = pinv(np.identity(n) - S)
    else:
        B = None
        D = None

    if bc_tol is None:
        bc_tol = tol

    # Maximum number of iterations
    max_iteration = 20

    fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped = wrap_functions(fun, bc, fun_jac, bc_jac, k, a, S, D, dtype)

    f = fun_wrapped(x, y, p)
    if f.shape != y.shape:
        raise ValueError(f"`fun` return is expected to have shape {y.shape}, "
                         f"but actually has {f.shape}.")

    bc_res = bc_wrapped(y[:, 0], y[:, -1], p)
    if bc_res.shape != (n + k,):
        raise ValueError(f"`bc` return is expected to have shape {(n + k,)}, "
                         f"but actually has {bc_res.shape}.")

    status = 0
    iteration = 0
    if verbose == 2:
        print(f"{'Iteration':^15}{'Max residual':^15}{'Max BC residual':^15}{'Total nodes':^15}{'Nodes added':^15}")

    while True:
        m = x.shape[0]

        col_fun, jac_sys = prepare_sys(n, m, k, fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped, x, h)
        y, p, singular = solve_newton(n, m, h, col_fun, bc_wrapped, jac_sys, y, p, B, tol, bc_tol)
        iteration += 1

        col_res, f = collocation_fun(fun_wrapped, y,
                                                         p, x, h)
        bc_res = bc_wrapped(y[:, 0], y[:, -1], p)
        max_bc_res = np.max(abs(bc_res))

        # This relation is not trivial, but can be verified.
        sol = create_spline(y, f, x, h)
        rms_res = estimate_rms_residuals(fun_wrapped, sol, x, h, p)
        max_rms_res = np.max(rms_res)

        if singular:
            status = 2
            break

        insert_1, = np.nonzero((rms_res > tol) & (rms_res < 100 * tol))
        insert_2, = np.nonzero(rms_res >= 100 * tol)
        nodes_added = insert_1.shape[0] + 2 * insert_2.shape[0]

        if m + nodes_added > max_nodes:
            status = 1
            if verbose == 2:
                nodes_added = f"({nodes_added})"
                print_iteration_progress(iteration, max_rms_res, max_bc_res,
                                         m, nodes_added)
            break

        if verbose == 2:
            print_iteration_progress(iteration, max_rms_res, max_bc_res, m,
                                     nodes_added)

        if nodes_added > 0:
            x = modify_mesh(x, insert_1, insert_2)
            h = np.diff(x)
            y = sol(x)
        elif max_bc_res <= bc_tol:
            status = 0
            break
        elif iteration >= max_iteration:
            status = 3
            break

    if verbose > 0:
        if status == 0: # success
            print(f"Solved in {iteration} iterations, number of nodes {x.shape[0]}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")
        elif status == 1: # max nodes exceeded
            print(f"Number of nodes is exceeded after iteration {iteration}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")
        elif status == 2: # singular matrix
            print("Singular Jacobian encountered when solving the collocation "
                  f"system on iteration {iteration}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")
        elif status == 3: # max iterations exceeded
            print("The solver was unable to satisfy boundary conditions "
                  f"tolerance on iteration {iteration}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")

    if p.size == 0:
        p = None

    return BVPResult(sol=sol, p=p, x=x, y=y, yp=f, rms_residuals=rms_res,
                     niter=iteration, status=status,
                     message=TERMINATION_MESSAGES[status], success=status == 0)
