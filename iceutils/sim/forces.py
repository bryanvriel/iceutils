#-*- coding: utf-8 -*-

import numpy as np
import sys

class CalvingForce:
    """
    Encapsulates a time-varying forcing function at calving front.
    """

    def __init__(self, A, n=3, fs=None, mode='constant', cval=1.0, period=1.0, amplitude=0.5,
                 rho_ice=917.0, rho_water=1024.0, g=9.80665):
        """
        Store parameters.
        """
        self.A = A
        self.n = n
        self.rho_ice = rho_ice
        self.rho_ratio = rho_ice / rho_water
        self.g = g

        # The time history of the forcing factor
        if fs is not None:
            self.fs = fs
        elif mode == 'constant':
            self.fs = lambda t: cval
        elif mode == 'periodic':
            self.fs = lambda t: amplitude * np.sin(2.0 * np.pi / period * t) + 1.0
        elif mode == 'step':
            self.fs = lambda t: 1.5 if t < 0.5 else 1.0
        else:
            raise NotImplementedError('Forcing mode not yet implemented.')

    def __call__(self, t, h):
        """
        Evaluate force at a given time and height of calving front.
        """
        # Evaluate the force value
        fs = self.fs(t)

        # Compute force
        return fs * self.A * (0.25 * self.rho_ice * self.g * h * (1 - self.rho_ratio))**self.n


# end of file
