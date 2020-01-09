#-*- coding: utf-8 -*-

import numpy as np
from scipy import optimize
import sys

def find_root(profile, model, method='newton', n_iter=500, tol=1.0e-5, scale=1.0e-2,
              options=None):
    """
    Wrapper for various root-finding optimization algorithms. For any method other than 'newton',
    it calls scipy.optimize.root.
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

def _find_root_newton(profile, model, n_iter=500, tol=1.0e-5, scale=1.0e-2,
                      reltol=1.0e-10, delta=0.2, rcond=1.0e-10):
    """
    Implements Newton's method for finding the roots of a multivariate function. Function
    is provided by model.compute_pde_values().
    """
    # Initial velocity and pde values
    U = profile.u.copy()
    F_prev = model.compute_pde_values(U)

    # Begin iterations
    for i in range(n_iter):

        # Compute value of PDE at current point
        F = model.compute_pde_values(U, scale=scale)
        Fmag = np.linalg.norm(F)

        # Diagnostics
        if i % 50 == 0:
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
        dU = np.linalg.lstsq(J, -F, rcond=rcond)[0]

        # Update velocities
        U += delta * dU
        F_prev = F

    return U, F