#-*- coding: utf-8 -*-

import jax.numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
import jax
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
        self.nu_eps = 1.0e-8

        # The force at the calving front
        self.fs = calving_force

        # Effective flotation height (any height below flotation is set to a small number)
        Hf = profile.h - self.rho_water / self.rho_ice * profile.depth
        mask = (Hf < 0.0).nonzero()[0]
        self.Hf = jax.ops.index_update(Hf, mask, 0.001)

        # Initialize jacobian function
        self.fjac = jax.jacfwd(self.compute_pde_values, 0)

    def update_nu(self, u):
        """
        Computes effective viscosity for a given velocity profile.
        """
        n, A = self.n, self.A
        D = self.profile.D
        Du = np.dot(D, u)
        self.nu = A**(-1 / n) / (np.abs(Du)**((n - 1) / n) + self.nu_eps)

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
        g, n, m, A, nu = [getattr(self, attr) for attr in ('g', 'n', 'm', 'A', 'nu')]

        # Cache some variables from the profile
        D, h, alpha, N = [getattr(self.profile, attr) for attr in ('D', 'h', 'alpha', 'N')]

        # Allocate array for vector result
        F = np.zeros(N + 2)

        # Compute gradient of velocity profile
        Du = np.dot(D, u)

        # Membrane stresses
        membrane = scale * 2.0 * np.dot(D, h * nu * Du)

        # Basal drag
        absu = np.abs(u)
        usign = u / absu
        basal = scale * usign * self.cb * absu**(1.0 / m)

        # Driving stress
        Td = scale * self.rho_ice * g * h * alpha

        # Optionally return individual stress components
        if return_components:
            return {'membrane': membrane,
                    'basal': basal,
                    'driving': Td}

        # Compute boundary conditions
        b1 = self.boundary_value_scale * Du[0]
        b2 = self.boundary_value_scale * (Du[-1] - self.fs)

        # Fill out PDE residual array
        stress = membrane - basal + Td
        F1 = jax.ops.index_update(F, slice(0, N), stress)
        F2 = jax.ops.index_update(F1, slice(N, N + 2), [b1, b2])

        # Done
        return F2

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
        # Pass arguments to jacobian function
        J = self.fjac(u, scale=scale)
        # Done
        return J


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
    q: float, optional
        Exponent for effective pressure. Default: 3.5.
    bv_scale: float, optional
        Scale factor for boundary conditions. Default: 500.0.
    """
    
    def __init__(self, profile, calving_force, A, W=3000.0, mu=1.0, n=3, m=3, q=0.333,
                 bv_scale=500.0):
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
        self.q = q
        self.rho_ice = profile.rho_ice
        self.rho_water = profile.rho_water

        # Numerical parameters
        self.boundary_value_scale = bv_scale

        # Epsilon value when computing the effective viscosity
        # When grid cell size gets smaller, this should also be smaller to ensure stability
        self.nu_eps = 1.0e-8

        # The force at the calving front
        self.fs = calving_force

        # Effective flotation height (any height below flotation is set to a small number)
        Hf = profile.h - self.rho_water / self.rho_ice * profile.depth
        mask = (Hf < 0.01).nonzero()[0]
        self.Hf = jax.ops.index_update(Hf, mask, 0.01)

        # Initialize jacobian function
        self.fjac = jax.jacfwd(self.compute_pde_values, 0)

    def update_nu(self, u):
        """
        Computes effective viscosity for a given velocity profile.
        """
        n, A = self.n, self.A
        D = self.profile.D
        Du = np.dot(D, u)
        self.nu = A**(-1 / n) / (np.abs(Du)**((n - 1) / n) + self.nu_eps)

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
        g, n, m, q, W, A, mu, nu = [
            getattr(self, attr) for attr in 
            ('g', 'n', 'm', 'q', 'W', 'A', 'mu', 'nu')
        ]

        # Cache some variables from the profile
        D, h, depth, alpha, rho_ice, rho_water, N = [
            getattr(self.profile, attr) for attr in 
            ('D', 'h', 'depth', 'alpha', 'rho_ice', 'rho_water', 'N')
        ]
    
        # Allocate array for vector result
        F = np.zeros(N + 2)

        # Compute gradient of velocity profile
        Du = np.dot(D, u)

        # Membrane stresses
        membrane = scale * 2.0 * np.dot(D, h * nu * Du)

        # Basal drag scaled by effective pressure
        absu = np.abs(u)
        usign = u / absu
        basal_drag = scale * usign * mu * (self.Hf**q) * (absu**(1 / m))

        # Lateral drag
        lateral_drag = scale * 2 * usign * h / W * (5 * absu / (A * W))**(1 / n)

        # Driving stress
        Td = scale * self.rho_ice * g * h * alpha

        # At this point, return individual components if requested
        if return_components:
            cdict = {'membrane': membrane, 'basal': basal_drag,
                     'lateral': lateral_drag, 'driving': Td}
            return cdict

        # Compute boundary conditions
        b1 = self.boundary_value_scale * Du[0]
        b2 = self.boundary_value_scale * (Du[-1] - self.fs)

        # Fill out PDE residual array
        stress = membrane - basal_drag - lateral_drag + Td
        F1 = jax.ops.index_update(F, slice(0, N), stress)
        F2 = jax.ops.index_update(F1, slice(N, N + 2), [b1, b2])

        # Done
        return F2
        
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
        # Pass arguments to jacobian function
        J = self.fjac(u, scale=scale)
        # Done
        return J


# end of file
