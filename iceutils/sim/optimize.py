#-*- coding: utf-8 -*-

# Get numpy
try:
    from .jax_models import np
except ImportError:
    from .models import np

# Other packages
from scipy import optimize
import scipy.linalg as sla
import sys

def find_root(profile, model, method='newton', n_iter=500, tol=1.0e-5, options=None):
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
        U, F = _find_root_newton(profile, model, n_iter=n_iter, tol=tol, **options)
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
                      reltol=1.0e-10,
                      delta=0.2,
                      reg_param=1.0e-10,
                      disp=50):
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
    reltol: float, optional
        Relative tolerance (minimum change in residual norm). Default: 1.0e-10.
    delta: float, optional
        Learning rate for Newton iterations. Default: 0.2
    reg_param: float, optional
        Regularization parameter for computing Hessian matrix. Default: 1.0e-10.
    disp: int, optional
        Skip number of lines to print diagnostics. If None, nothing is printed to screen.

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
    #regmat = reg_param * np.eye(U.size)
    #import matplotlib.pyplot as plt

    # Begin iterations
    for i in range(n_iter):

        # Compute value of PDE at current point
        F = model.compute_pde_values(U)
        Fmag = np.linalg.norm(F)

        # Diagnostics
        if disp is not None and i % disp == 0:
            print('Iteration %03d error: %8.5e' % (i, Fmag))

        # Check convergence
        F_diff = F - F_prev
        if Fmag < tol:
            break
        elif np.linalg.norm(F_diff) < reltol and i > 5:
            break

        # Compute Jacobian at current point
        J = model.compute_jacobian(U)

        # Compute update vector
        JtJ = np.dot(J.T, J) #+ regmat
        JtF = np.dot(J.T, F)

        iJtJ = np.linalg.inv(JtJ)
        dU = np.dot(iJtJ, -1.0*JtF)

        #Am = 0.5 * (JtJ + JtJ.T)
        #c, low = sla.cho_factor(Am)
        #dU = sla.cho_solver((c, low), -1.0*JtF)

        #dU = sla.lstsq(JtJ, -1.0*JtF)[0]

        # Update velocities
        U += delta * dU
        F_prev = F

    return U, F

# end of file
