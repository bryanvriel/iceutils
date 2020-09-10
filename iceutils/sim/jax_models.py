#-*- coding: utf-8 -*-

import jax.numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from functools import partial
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
    scale: float, optional
        Scale factor for computing PDE residuals. Default: 0.01.
    """
    
    def __init__(self, profile, calving_force, A, cb=6.0e2, n=3, m=3, bv_scale=500.0,
                 scale=1.0e-2):
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
        self.scale = scale
        
        # The force at the calving front
        self.fs = calving_force

        # Effective flotation height (any height below flotation is set to a small number)
        Hf = profile.h - self.rho_water / self.rho_ice * profile.depth
        mask = (Hf < 0.0).nonzero()[0]
        self.Hf = jax.ops.index_update(Hf, mask, 0.001)

        # Initialize jacobian function
        self.fjac = jax.jacfwd(self.compute_pde_values, 0)

    @partial(jax.jit, static_argnums=(0,))
    def compute_pde_values(self, u):
        """
        Compute vector of PDE residuals for a given velocity profile.
        
        Parameters
        ----------
        u: (N,) ndarray
            Array of ice velocity in m/yr.

        Returns
        ------- 
        F: (N+2,) ndarray
            Array of PDE residuals. Returned if return_components = False.
        """

        # Cache some parameters to use here
        g, n, m, A, scale = [getattr(self, attr) for attr in ('g', 'n', 'm', 'A', 'scale')]

        # Cache some variables from the profile
        D, h, alpha, N = [getattr(self.profile, attr) for attr in ('D', 'h', 'alpha', 'N')]

        # Allocate array for vector result
        F = np.zeros(N + 2)

        # Compute gradient of velocity profile
        Du = np.dot(D, u) + 1.0e-13 # epsilon to avoid divide-by-zero

        # Dynamic viscosity
        nu = A**(-1 / n) / (np.abs(Du)**((n - 1) / n))

        # Membrane stresses
        membrane = scale * 2.0 * np.dot(D, h * nu * Du)

        # Basal drag
        absu = np.abs(u) + 1.0e-13 # epsilon to avoid divide-by-zero
        usign = u / absu
        basal = scale * usign * self.cb * absu**(1.0 / m)
        #basal = scale * usign * self.cb

        # Driving stress
        Td = scale * self.rho_ice * g * h * alpha

        # Compute boundary conditions
        b1 = self.boundary_value_scale * Du[0]
        b2 = self.boundary_value_scale * (Du[-1] - self.fs)

        # Fill out PDE residual array
        stress = membrane - basal + Td
        F1 = jax.ops.index_update(F, slice(0, N), stress)
        F2 = jax.ops.index_update(F1, slice(N, N + 2), [b1, b2])

        # Done
        return F2

    @partial(jax.jit, static_argnums=(0,))
    def compute_jacobian(self, u):
        """ 
        Compute Jacobian of residual array with respect to a given velocity array.
                    
        Parameters
        ----------
        u: (N,) ndarray
            Array of ice velocity in m/yr.
        
        Returns
        -------
        J: (N+2, N+2) ndarray
            2D Jacobian array.
        """
        # Pass arguments to jacobian function
        J = self.fjac(u)
        # Done
        return J

    def compute_stress_components(self, u):
        """
        Compute stress components and return in dict.
        
        Parameters
        ----------
        u: (N,) ndarray
            Array of ice velocity in m/yr.
        
        Returns
        ------- 
        stress_dict: dict
            Dictionary of individual stress components.
        """

        # Cache some parameters to use here
        g, n, m, A = [getattr(self, attr) for attr in ('g', 'n', 'm', 'A')]

        # Cache some variables from the profile
        D, h, alpha, N = [getattr(self.profile, attr) for attr in ('D', 'h', 'alpha', 'N')]

        # Compute gradient of velocity profile
        Du = np.dot(D, u) + 1.0e-13

        # Dynamic viscosity
        nu = A**(-1 / n) / (np.abs(Du)**((n - 1) / n))

        # Membrane stresses
        membrane = 2.0 * np.dot(D, h * nu * Du)

        # Basal drag
        absu = np.abs(u) + 1.0e-13
        usign = u / absu
        basal = usign * self.cb * absu**(1.0 / m)

        # Driving stress
        Td = self.rho_ice * g * h * alpha

        # Return individual stress components
        return {'membrane': membrane,
                'basal': basal,
                'driving': Td}



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
    scale: float, optional
        Scale factor for computing PDE residuals. Default: 0.01.
    """
    
    def __init__(self, profile, calving_force, A, W=3000.0, mu=1.0, n=3, m=3, bv_scale=500.0,
                 scale=1.0e-2):
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
        self.scale = scale
        
        # The force at the calving front
        self.fs = calving_force

        # Effective flotation height (any height below flotation is set to a small number)
        Hf = profile.h - self.rho_water / self.rho_ice * profile.depth
        mask = (Hf < 0.01).nonzero()[0]
        self.Hf = jax.ops.index_update(Hf, mask, 0.01)

        # Initialize jacobian function
        self.fjac = jax.jacfwd(self.compute_pde_values, 0)

    @partial(jax.jit, static_argnums=(0,))
    def compute_pde_values(self, u):
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
        g, n, m, W, A, mu, scale = [
            getattr(self, attr) for attr in 
            ('g', 'n', 'm', 'W', 'A', 'mu', 'scale')
        ]

        # Cache some variables from the profile
        D, h, depth, alpha, rho_ice, rho_water, N = [
            getattr(self.profile, attr) for attr in 
            ('D', 'h', 'depth', 'alpha', 'rho_ice', 'rho_water', 'N')
        ]
    
        # Allocate array for vector result
        F = np.zeros(N + 2)

        # Compute gradient of velocity profile
        Du = np.dot(D, u) + 1.0e-13 # epsilon to avoid divide-by-zero

        # Dynamic viscosity
        nu = A**(-1 / n) / (np.abs(Du)**((n - 1) / n))

        # Membrane stresses
        membrane = scale * 2.0 * np.dot(D, h * nu * Du)

        # Basal drag
        absu = np.abs(u) + 1.0e-13 # epsilon to avoid divide-by-zero
        usign = u / absu
        #basal_drag = scale * usign * mu * (self.Hf * absu)**(1 / m)
        basal_drag = scale * usign * mu * absu**(1.0 / m)

        # Lateral drag
        lateral_drag = scale * 2 * usign * h / W * (5 * absu / (A * W))**(1 / n)

        # Driving stress
        Td = scale * self.rho_ice * g * h * alpha
        
        # Compute boundary conditions
        b1 = self.boundary_value_scale * Du[0]
        b2 = self.boundary_value_scale * (Du[-1] - self.fs)

        # Fill out PDE residual array
        stress = membrane - basal_drag - lateral_drag + Td
        F1 = jax.ops.index_update(F, slice(0, N), stress)
        F2 = jax.ops.index_update(F1, slice(N, N + 2), [b1, b2])

        # Done
        return F2
       
    @partial(jax.jit, static_argnums=(0,)) 
    def compute_jacobian(self, u):
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
        J = self.fjac(u)
        # Done
        return J

    def compute_stress_components(self, u):
        """
        Compute stress components and return in dict.
        
        Parameters
        ----------
        u: (N,) ndarray
            Array of ice velocity in m/yr.
        
        Returns
        ------- 
        stress_dict: dict
            Dictionary of individual stress components.
        """
        # Cache some parameters to use here
        g, n, m, W, A, mu, scale = [
            getattr(self, attr) for attr in 
            ('g', 'n', 'm', 'W', 'A', 'mu', 'scale')
        ]

        # Cache some variables from the profile
        D, h, depth, alpha, rho_ice, rho_water, N = [
            getattr(self.profile, attr) for attr in 
            ('D', 'h', 'depth', 'alpha', 'rho_ice', 'rho_water', 'N')
        ]
    
        # Compute gradient of velocity profile
        Du = np.dot(D, u) + 1.0e-13 # epsilon to avoid divide-by-zero

        # Dynamic viscosity
        nu = A**(-1 / n) / (np.abs(Du)**((n - 1) / n))

        # Membrane stresses
        membrane = scale * 2.0 * np.dot(D, h * nu * Du)

        # Basal drag
        absu = np.abs(u) + 1.0e-13 # epsilon to avoid divide-by-zero
        usign = u / absu
        #basal_drag = scale * usign * mu * (self.Hf * absu)**(1 / m)
        basal_drag = scale * usign * mu * absu**(1.0 / m)

        # Lateral drag
        lateral_drag = scale * 2 * usign * h / W * (5 * absu / (A * W))**(1 / n)

        # Driving stress
        Td = scale * self.rho_ice * g * h * alpha

        # Return individual components in dictionary
        cdict = {'membrane': membrane, 'basal': basal_drag,
                 'lateral': lateral_drag, 'driving': Td}
        return cdict


# end of file
