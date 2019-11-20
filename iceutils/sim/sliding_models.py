#-*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
import sys

from .utilities import *

class Basal:

    def __init__(self, n=3, m=3, rho_ice=917.0, rho_water=1024.0):
        """
        Store fixed parameters of model.
        """
        self.n = n
        self.m = m
        self.g = 9.80665
        self.rho_ice = rho_ice
        self.rho_water = rho_water

    def compute_velocities(self, params, profile):
        """
        Compute advective and diffusion velocity components for given tuning parameters.
        """
        # Unpack the tuning parameters
        k = params[0]

        # Cache other parameters
        n, m, g = self.n, self.m, self.g
        rho_water, rho_ice = self.rho_water, self.rho_ice

        # The sliding velocity
        u = k * (profile.alpha**n * profile.h**(n - m)) / (1 - profile.PwPi)**m

        # Advective speed
        C = u * (1 + ((n - m) - n * profile.PwPi) / (1 - profile.PwPi)**m)

        # Diffusion
        D = (k * n * profile.alpha**(n - 1) * profile.h**(n - m + 1)) / (1 - profile.PwPi)**m

        #return VelocityComponents(profile.s, u, C, D, spline=True, smoothing=2.0)
        return VelocityComponents(profile.x, u, C, D)


class LateralShear:

    def __init__(self, half_width, n=3, rho_ice=917.0, rho_water=1024.0):
        """
        Store fixed parameters of model.
        """
        # Store half-width
        self.W = half_width

        # Store other parameters
        self.n = n
        self.g = 9.80665
        self.rho_ice = rho_ice
        self.rho_water = rho_water

    def compute_velocities(self, params, profile):
        """
        Compute advective and diffusion velocity components for given tuning parameters.
        """
        # Unpack the tuning parameters
        T = params[0]

        # Cache other parameters
        n, g = self.n, self.g
        W = self.W

        # Compute Glen's flow law parameter (units of 1 / (yr * Pa^3))
        A = AGlen_vs_temp(T)

        # Compute constant (no variation in space)
        K = 2*A / (n + 1) * (W**(n + 1)) * ((self.rho_ice * g)**n)

        # Advective speed
        C = K * profile.alpha**n
        D = K * profile.h * n * profile.alpha**(n - 1)

        return VelocityComponents(profile.x, C, C, D)


class LateralBasal:

    def __init__(self, half_width, n=3, rho_ice=917.0, rho_water=1024.0):
        """
        Store fixed parameters of model.
        """
        # Store half-width
        self.W = half_width

        # Store other parameters
        self.n = n
        self.g = 9.80665
        self.rho_ice = rho_ice
        self.rho_water = rho_water

    def compute_velocities(self, params, profile):
        """
        Compute advective and diffusion velocity components for given tuning parameters.
        """
        # Unpack the tuning parameters
        C, T = params

        # Cache other parameters
        n, g = self.n, self.g
        W = self.W
        PwPi = profile.PwPi

        # Compute Glen's flow law parameter (units of 1 / (yr * Pa^3))
        A = AGlen_vs_temp(T)

        # Compute constant (varies in space)
        K = 2*A / (n + 1) * (W**(n + 1)) * (self.rho_ice * g * (1 - (1 - PwPi)**C))**n
       
        # Velocity components
        C = K * profile.alpha**n
        D = K * profile.h * n * profile.alpha**(n - 1)
        Cp = np.gradient(C, profile.ds, edge_order=2)
        Dp = np.gradient(D, profile.ds, edge_order=2)

        return VelocityComponents(C, C, D, Cp, Dp)


class VelocityComponents:

    def __init__(self, s, U, C, D, spline=False, smoothing=1.0):

        # Store the data
        self.s = s
        self.U = U
        self.C = C
        self.D = D
        self.ds = abs(s[1] - s[0])

        # Check if using a spline representation for C and D components
        self.spline = spline
        if spline:
            self._cspline = UnivariateSpline(self.s, self.C, k=5, s=smoothing)
            self._dspline = UnivariateSpline(self.s, self.D, k=5, s=smoothing)

    @property
    def Vk(self):
        """
        Kinematic wave speed.
        """
        return self.C - self.Dp

    @property
    def Cp(self):
        """
        The spatial derivative of C component.
        """
        if self.spline:
            return self._cspline.derivative(n=1)(self.s)
        else:
            return np.gradient(self.C, self.ds, edge_order=2)

    @Cp.setter
    def Cp(self, value):
        raise ValueError('Cannot set Cp explicitly')

    @property
    def Dp(self):
        """
        The spatial derivative of D component.
        """
        if self.spline:
            return self._dspline.derivative(n=1)(self.s)
        else:
            return np.gradient(self.D, self.ds, edge_order=2)

    @Dp.setter
    def Dp(self, value):
        raise ValueError('Cannot set Dp explicitly')

# ------------------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------------------

def compute_Pe(V, profile):
    """
    Computes Peclet number for velocity result and profile.
    """
    return (V.C - V.Dp) / V.D * (profile.s.max() - profile.s)

# end of file
