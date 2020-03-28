#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
import sys


class IceStream:
    """
    1-D flowline model for basal sliding, e.g.

    membrane_stress + basal_drag = driving_stress

    Parameters
    ----------
    profile: iceutils.sim.Profile
        Profile instance.
    calving_force: iceutils.sim.CalvingForce
        CalvingForce instance.
    A: float
        Glen's Flow Law parameter in {a^-1} {Pa^-3}.
    cb: float, optional
        Sliding law prefactor. Default: 6.0e2.
    n: int, optional
        Glen's Flow Law exponent. Default: 3.
    m: int, optional
        Sliding law exponent. Default: 3.
    bv_scale: float, optional
        Scale factor for boundary conditions. Default: 500.0.
    """
    
    def __init__(self, profile, calving_force, A, cb=6.0e2, n=3, m=3, bv_scale=500.0):
        """
        Initialize IceStream class.
        """
        # Save the profile object
        self.profile = profile

        # Physical parameters
        self.A = A
        self.g = 9.80665
        self.cb = cb
        self.n = n
        self.m = m
        self.rho_ice = profile.rho_ice
        self.rho_water = profile.rho_water

        # Numerical parameters
        self.boundary_value_scale = bv_scale

        # Epsilon value when computing the effective viscosity
        # When grid cell size gets smaller, this should also be smaller to ensure stability
        self.nu_eps = 1.0e-6

        # The forc at the calving front
        self.fs = calving_force

        # Constant array to multiply membrane stress Jacobian
        self.K = 2.0 * profile.D * A**(-1 / n)

        # Pre-allocate arrays for storing PDE array and Jacobian
        self.F = np.zeros(self.profile.N + 2)
        self.J = np.zeros((self.profile.N + 2, self.profile.N))

    def compute_pde_values(self, u, scale=1.0e-2, return_components=False):
        """
        Compute vector of PDE residuals for a given velocity profile.

        Parameters
        ----------
        u: (N,) ndarray
            Array of ice velocity in m/yr.
        scale: float, optional
            Value for scaling PDE residuals. Default: 1.0e-2.
        return_components: bool, optional
            If True, return individual stress components in a dictionary.

        Returns
        -------
        F: (N+2,) ndarray
            Array of PDE residuals. Returned if return_components = False.
        stress_dict: dict
            Dictionary of individual stress components. Returned if return_components = True.
        """
        # Cache some parameters to use here
        g, n, m, A = [getattr(self, attr) for attr in ('g', 'n', 'm', 'A')]

        # Cache some variables from the profile
        D, h, alpha = [getattr(self.profile, attr) for attr in ('D', 'h', 'alpha')]

        # Compute gradient of velocity profile
        Du = np.dot(D, u)

        # Dynamic viscosity
        nu = A**(-1 / n) / (np.abs(Du)**((n - 1) / n) + self.nu_eps)

        # Membrane stresses
        membrane = scale * 2.0 * np.dot(D, h * nu * Du)

        # Basal drag
        absu = np.abs(u)
        usign = u / absu
        drag = scale * usign * self.cb * absu**(1.0 / m)

        # Driving stress
        Td = scale * self.rho_ice * g * h * alpha

        if return_components:
            return {'membrane': membrane,
                    'drag': drag,
                    'driving': Td}

        # Fill out the PDE vector, including boundary values
        self.F[:-2] = membrane - drag + Td
        self.F[-2] = self.boundary_value_scale * Du[0]
        self.F[-1] = self.boundary_value_scale * (Du[-1] - self.fs)

        # Return a copy (numdifftools may manipulate self.F during Jacobian computation)
        return self.F.copy()

    def compute_jacobian(self, u, scale=1.0e-2):
        """
        Compute Jacobian of residual array with respect to a given velocity array.
        
        Parameters
        ----------
        u: (N,) ndarray
            Array of ice velocity in m/yr.
        scale: float, optional
            Value for scaling PDE residuals. Default: 1.0e-2.

        Returns
        -------
        J: (N+2, N+2) ndarray
            2D Jacobian array.
        """
        # Cache some parameters to use here
        g, n, m = [getattr(self, attr) for attr in ('g', 'n', 'm')]

        # Cache some variables from the profile
        D, h, alpha = [getattr(self.profile, attr) for attr in ('D', 'h', 'alpha')]

        # Compute gradient of velocity profile
        Du = np.dot(D, u)

        # Normalized nu (nu / A)
        nu = 1.0 / (np.abs(Du)**((n - 1) / n) + self.nu_eps)

        # Factor that looks like nu, but used in Jacobian
        nu_hat = (n - 1) / n * Du / (np.abs(Du)**((n + 1) / n) + self.nu_eps)

        # Jacobian related to | du/dx | ** ((1 -n) / n)
        Gp = gradient_product(-1.0 * (nu**2) * nu_hat, D)

        # Composite Jacobian related to membrane stresses
        J1 = gradient_product(h * Du, Gp)
        J2 = gradient_product(h * nu, D)
        J_membrane = scale * np.dot(self.K, J1 + J2)

        # Jacobian for sliding drag (diagonal)
        absu = np.abs(u)
        usign = np.copysign(np.ones_like(u), u)
        J_drag = scale * self.cb * usign**2 / m * (absu + self.drag_eps)**((1 - m) / m)

        # Subtract drag Jacobian from diagonal of membrane stress Jacobian
        N = u.size
        J_membrane[range(N), range(N)] -= J_drag

        # Fill out full Jacobian
        self.J[:-2,:] = J_membrane
        self.J[-2,:] = self.boundary_value_scale * D[0,:]
        self.J[-1,:] = self.boundary_value_scale * D[-1,:]

        return self.J

    def compute_numerical_jacobian(self, u, step=1.0e-7):
        """
        Computes finite difference approximation to the Jacobian. Not intended to be used
        for simulation runs since it's quite slow, but it's useful for debugging.

        Parameters
        ----------
        u: (N,) ndarray
            Array of ice velocity in m/yr.
        step: float, optional
            Step size for finite differences. Default: 1.0e-7.

        Returns
        -------
        J: (N+2, N+2) ndarray
            2D Jacobian array.
        """
        import numdifftools as nd
        jacfun = nd.Jacobian(self.compute_pde_values, step=step)
        return jacfun(u)


class LateralIceStream:
    """
    1-D flowline model for basal sliding, e.g.

    membrane_stress + basal_drag = driving_stress

    Parameters
    ----------
    profile: iceutils.sim.Profile
        Profile instance.
    calving_force: iceutils.sim.CalvingForce
        CalvingForce instance.
    A: float
        Glen's Flow Law parameter {a^-1} {Pa^-3}.
    W: float, optional
        Width of glacier in meters. Default: 3000.0.
    mu: float, optional
        Frictional prefactor for sliding law. Default: 100.0.
    n: int, optional
        Glen's Flow Law exponent. Default: 3.
    m: int, optional
        Sliding law exponent. Default: 3.
    bv_scale: float, optional
        Scale factor for boundary conditions. Default: 500.0.
    """
    
    def __init__(self, profile, calving_force, A, W=3000.0, mu=100.0, n=3, m=3, bv_scale=500.0):
        """
        Initialize LateralIceStream class.
        """
        # Save the profile object
        self.profile = profile

        # Physical parameters
        self.W = W                  # FULL width of glacier
        self.A = A
        self.g = 9.80665
        self.mu = mu
        self.n = n
        self.m = m
        self.rho_ice = profile.rho_ice
        self.rho_water = profile.rho_water

        # Numerical parameters
        self.boundary_value_scale = bv_scale

        # Epsilon value when computing the effective viscosity
        # When grid cell size gets smaller, this should also be smaller to ensure stability
        self.nu_eps = 1.0e-8
        self.drag_eps = 1.0e-3

        # The force at the calving front
        self.fs = calving_force

        # Effective flotation height (any height below flotation is set to a small number)
        self.Hf = profile.h - self.rho_water / self.rho_ice * profile.depth
        self.Hf[self.Hf < 0.01] = 0.01

        # Constant array to multiply membrane stress Jacobian
        self.K = 2.0 * profile.D * A**(-1 / n)

        # Pre-allocate arrays for storing PDE array and Jacobian
        self.F = np.zeros(self.profile.N + 2)
        self.J = np.zeros((self.profile.N + 2, self.profile.N))

    def compute_pde_values(self, u, scale=1.0e-2, return_components=False):
        """
        Parameters
        ----------
        u: (N,) ndarray
            Array of ice velocity in m/yr.
        scale: float, optional
            Value for scaling PDE residuals. Default: 1.0e-2.
        return_components: bool, optional
            If True, return individual stress components in a dictionary.

        Returns
        -------
        F: (N+2,) ndarray
            Array of PDE residuals. Returned if return_components = False.
        stress_dict: dict
            Dictionary of individual stress components. Returned if return_components = True.
        """

        # Cache some parameters to use here
        g, n, m, W, A, mu = [
            getattr(self, attr) for attr in 
            ('g', 'n', 'm', 'W', 'A', 'mu')
        ]

        # Cache some variables from the profile
        D, h, depth, alpha, rho_ice, rho_water = [
            getattr(self.profile, attr) for attr in 
            ('D', 'h', 'depth', 'alpha', 'rho_ice', 'rho_water')
        ]

        # Compute gradient of velocity profile
        Du = np.dot(D, u)

        # Dynamic viscosity
        nu = A**(-1 / n) / (np.abs(Du)**((n - 1) / n) + self.nu_eps)

        # Membrane stresses
        membrane = scale * 2.0 * np.dot(D, h * nu * Du)

        # Basal drag
        absu = np.abs(u)
        usign = np.copysign(np.ones_like(u), u)
        basal_drag = scale * usign * mu * (self.Hf * absu)**(1 / m)

        # Lateral drag
        lateral_drag = scale * 2 * usign * h / W * (5 * absu / (A * W))**(1 / n)

        # Driving stress
        Td = scale * self.rho_ice * g * h * alpha

        # At this point, return individual components if requested
        if return_components:
            cdict = {'membrane': membrane, 'basal': basal_drag,
                     'lateral': lateral_drag, 'driving': Td}
            return cdict

        # Combine resistive stresses
        membrane -= (basal_drag + lateral_drag)

        # Fill out the PDE vector, including boundary values
        self.F[:-2] = membrane + Td
        self.F[-2] = self.boundary_value_scale * Du[0]
        self.F[-1] = self.boundary_value_scale * (Du[-1] - self.fs)

        # Return a copy (numdifftools may manipulate self.F during Jacobian computation)
        return self.F.copy()

    def compute_jacobian(self, u, scale=1.0e-2):
        """
        Compute Jacobian of residual array with respect to a given velocity array.
        
        Parameters
        ----------
        u: (N,) ndarray
            Array of ice velocity in m/yr.
        scale: float, optional
            Value for scaling PDE residuals. Default: 1.0e-2.

        Returns
        -------
        J: (N+2, N+2) ndarray
            2D Jacobian array.
        """

        # Cache some parameters to use here
        g, n, m, W, A, mu = [
            getattr(self, attr) for attr in 
            ('g', 'n', 'm', 'W', 'A', 'mu')
        ]

        # Cache some variables from the profile
        D, h, depth, alpha, rho_ice, rho_water = [
            getattr(self.profile, attr) for attr in 
            ('D', 'h', 'depth', 'alpha', 'rho_ice', 'rho_water')
        ]

        # Compute gradient of velocity profile
        Du = np.dot(D, u)

        # Normalized nu (nu / A)
        nu = 1.0 / (np.abs(Du)**((n - 1) / n) + self.nu_eps)

        # Factor that looks like nu, but used in Jacobian
        nu_hat = (n - 1) / n * Du / (np.abs(Du)**((n + 1) / n) + self.nu_eps)

        # Jacobian related to | du/dx | ** ((1 -n) / n)
        Gp = gradient_product(-1.0 * (nu**2) * nu_hat, D)

        # Composite Jacobian related to membrane stresses
        J1 = gradient_product(h * Du, Gp)
        J2 = gradient_product(h * nu, D)
        Jpde = scale * np.dot(self.K, J1 + J2)

        # Jacobian for sliding drag (diagonal)
        absu = np.abs(u)
        usign = np.copysign(np.ones_like(u), u)
        J_basal = (scale * mu * usign * (self.Hf * usign) / 
                   m * (self.Hf * absu)**((1 - m) / m))
        
        # Jacobian for lateral drag (diagonal)
        J_lateral = (scale * usign * 10 * h * usign /
                    (n * A * W**2) * (5 * absu / (A * W))**((1 - n) / n))

        # Subtract drag terms from diagonal of membrane stress Jacobian
        N = u.size
        Jpde[range(N), range(N)] -= J_basal
        Jpde[range(N), range(N)] -= J_lateral

        # Fill out full Jacobian
        self.J[:-2,:] = Jpde
        self.J[-2,:] = self.boundary_value_scale * D[0,:]
        self.J[-1,:] = self.boundary_value_scale * D[-1,:]

        return self.J

    def compute_numerical_jacobian(self, u, step=1.0e-7):
        """
        Computes finite difference approximation to the Jacobian. Not intended to be used
        for simulation runs since it's quite slow, but it's useful for debugging.

        Parameters
        ----------
        u: (N,) ndarray
            Array of ice velocity in m/yr.
        step: float, optional
            Step size for finite differences. Default: 1.0e-7.

        Returns
        -------
        J: (N+2, N+2) ndarray
            2D Jacobian array.
        """
        import numdifftools as nd
        jacfun = nd.Jacobian(self.compute_pde_values, step=step)
        return jacfun(u)

    

def gradient_product(self, b, A):
    """
    Helper function for returning the equivalent of the product:

    (I .* outer(b, 1s)) * A

    but using the faster einsum operation.
    """
    return np.einsum('ij,i->ij', A, b)

# end of file
