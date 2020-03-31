#-*- coding: utf-8 -*-

# Get numpy
try:
    from .jax_models import np
except ImportError:
    from .models import np

# Other packages
from scipy import optimize
import sys

def find_root(profile, model, method='newton', n_iter=500, tol=1.0e-5, scale=1.0e-2,
              options=None):
    """
    Wrapper for various root-finding optimization algorithms. For any method other than 'newton',
    it calls scipy.optimize.root.

    Parameters
    ----------
    profile: iceutils.sim.Profile
        Profile instance.
    model: iceutils.sim model
        Sliding model instance.
    method: str, optional
        Root finding algorithm. Use any scipy.optimize.root algorithm or 'newton'.
        Default: 'newton'.
    n_iter: int, optional
        Number of interations: Default: 500.
    tol: float, optional
        Tolerence for optimization. Default: 1.0e-5.
    scale: float, optional
        Value for scaling PDE residuals. Default: 1.0e-2.
    options: dict, optional
        Options dict for scipy.optimize.root.

    Returns
    -------
    U: (N,) ndarray
        Ice velocity solution.
    F: (N,) ndarray
        PDE residual values.
    """
    if method == 'newton':
        U, F = _find_root_newton(profile, model, n_iter=n_iter, tol=tol, scale=scale,
                                 **options)
    else:
        result = optimize.root(model.compute_pde_values, profile.u,
                               jac=model.compute_jacobian, method=method,
                               options=options)
        U = result.x
        F = result.fun

    # Done
    return U, F

def _find_root_newton(profile,
                      model,
                      n_iter=500,
                      tol=1.0e-5,
                      scale=1.0e-2,
                      reltol=1.0e-10,
                      delta=0.2,
                      reg_param=1.0e-10,
                      verbose=True):
    """
    Implements Newton's method for finding the roots of a multivariate function. Function
    is provided by model.compute_pde_values().

    Parameters
    ----------
    profile: iceutils.sim.Profile
        Profile instance.
    model: iceutils.sim model
        Sliding model object.
    n_iter: int, optional
        Number of interations: Default: 500.
    tol: float, optional
        Tolerence for optimization. Default: 1.0e-5.
    scale: float, optional
        Value for scaling PDE residuals. Default: 1.0e-2.
    reltol: float, optional
        Relative tolerance (minimum change in residual norm). Default: 1.0e-10.
    delta: float, optional
        Learning rate for Newton iterations. Default: 0.2
    reg_param: float, optional
        Regularization parameter for computing Hessian matrix. Default: 1.0e-10.
    verbose: bool, optional
        Print out iteration progress. Default: False.

    Returns
    -------
    U: (N,) ndarray
        Ice velocity solution.
    F: (N,) ndarray
        PDE residual values.
    """
    # Initial velocity and pde values
    U = profile.u.copy()
    F_prev = model.compute_pde_values(U)

    # Damping matrix
    regmat = reg_param * np.eye(U.size)

    # Begin iterations
    for i in range(n_iter):

        # Compute value of PDE at current point
        F = model.compute_pde_values(U, scale=scale)
        Fmag = np.linalg.norm(F)

        # Diagnostics
        if i % 50 == 0 and verbose:
            print('Iteration %03d error: %8.5e' % (i, Fmag))

        # Check convergence
        F_diff = F - F_prev
        if Fmag < tol:
            break
        elif np.linalg.norm(F_diff) < reltol and i > 5:
            break

        # Compute Jacobian at current point
        J = model.compute_jacobian(U, scale=scale)

        # Compute update vector
        JtJ = np.dot(J.T, J) + regmat
        JtF = np.dot(J.T, F)
        iJtJ = np.linalg.inv(JtJ)
        dU = np.dot(iJtJ, -1.0*JtF)
        
        # Update velocities
        U += delta * dU
        F_prev = F

    return U, F

# end of file
