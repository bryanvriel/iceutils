#-*- coding: utf-8 -*-

# Get numpy
try:
    from .jax_models import np
except ImportError:
    from .models import np

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d, InterpolatedUnivariateSpline
import sys


class Profile:

    def __init__(self, x, h, b, u, rho_ice=917.0, rho_water=1024.0, t=0.0):
        """
        Stores profiles of DOWNSTREAM distance:

            s = 0 -> glacier start
            s = L -> glacier terminus

        and ice thickness and bed elevation below sea level.
        """
        # Store the data
        self.x = x
        self.h = h
        self.b = b
        self.u = u
        self.rho_water, self.rho_ice = rho_water, rho_ice
        self.depth = -1.0 * self.b
        self.N = len(x)
        self.dx = x[1] - x[0]
        self.t = t

        # Upstream distance
        self.l = x.max() - x

        # Create finite difference matrix
        self.D = construct_finite_diff_matrix(self.x, edge_order=2)

        # Initialize other thickness-dependent quantities
        self._init_derived_quantities()

    def _init_derived_quantities(self):
        """
        Private function for computing quantities depending on the thickness profile.
        """
        # Ice surface
        self.s = self.b + self.h
        # Surface slope
        self.alpha = -1.0 * np.dot(self.D, self.s)
        # Depth ratio
        self.depth_ratio = self.depth / self.h
        # Water pressure ratio
        self.PwPi = self.rho_water / self.rho_ice * self.depth_ratio
        # Height above flotation
        self.HAF = self.b + self.h * self.rho_ice / self.rho_water

    def set_profile_ydata(self, h=None, b=None, u=None):
        """
        Update thickness profiles for height, bed, and velocity.
        """
        # Height
        if h is not None:
            self.h = h
            self._init_derived_quantities()

        # Bed
        if b is not None:
            self.b = b
            self._init_derived_quantities()

        # Velocity
        if u is not None:
            self.u = u

    def update_coordinates(self, x, interp_kind='cubic', extrapolate=False):
        """
        Creates a NEW Profile object with a new set of coordinates.
        """
        # Handle extrapolation arguments
        if extrapolate:
            fill_value = 'extrapolate'
            bounds_error = False
        else:
            fill_value = np.nan
            bounds_error = True

        # Create new object with interpolated profiles
        return Profile(
            x,
            interp1d(self.x, self.h, kind=interp_kind, fill_value=fill_value,
                     bounds_error=bounds_error)(x),
            interp1d(self.x, self.b, kind=interp_kind, fill_value=fill_value,
                     bounds_error=bounds_error)(x),
            interp1d(self.x, self.u, kind=interp_kind, fill_value=fill_value,
                     bounds_error=bounds_error)(x),
        )

    def update_terminus(self, pad=5, verbose=False):
        """
        Updates terminus position to satisfy flotation criterion.
        """
        # Compute flotation criterion
        f = self.flotation_criterion()

        # Check for retreat
        if f[-1] < 0.0:
            new_profile = retreat_terminus(self, verbose=verbose)
        
        # Check for advance
        else:
            new_profile = advance_terminus(self, pad=pad, verbose=verbose)
            # Take another corrective retreat
            if new_profile.HAF[-1] < 0.0:
                new_profile = retreat_terminus(new_profile, verbose=verbose)

        # Return new profile
        return new_profile

    def flotation_criterion(self):
        """
        Computes flotation criterion (height above flotation).
        """
        flotation = self.rho_ice * self.h - self.rho_water * self.depth
        return flotation

    def subset_grounded(self, float_eps=0.0, return_mask=False):
        """
        Return a new Profile object subset to where ice thicknesses are above
        a flotation criterion.
        """
        # Compute the flotation matrix
        flotation = self.flotation_criterion()

        # Mask
        mask = flotation > float_eps

        # Subset and make new object
        profile = Profile(self.x[mask], self.h[mask], self.b[mask], self.u[mask],
                          rho_ice=self.rho_ice, rho_water=self.rho_water, t=self.t)

        # Override the grounded slope with masked full slope 
        # (To minimize edge effects at grounding line)
        profile.alpha = self.alpha[mask]
   
        if return_mask:
            return profile, mask
        else: 
            return profile

    def plot(self, axes=None, items=['geo', 'vel']):
        if axes is None:
            fig, axes = plt.subplots(nrows=len(items), figsize=(10,6))
        else:
            assert len(axes) == len(items), 'Incompatible number of subplots'
        for count, item in enumerate(items):
            ax = axes[count]
            if item == 'geo':
                ax.plot(self.x, self.b)
                ax.plot(self.x, self.s)
                ax.plot(self.x, self.HAF)
            elif item == 'vel':
                ax.plot(self.x, self.u)
            elif item == 'flux':
                flux = self.h * self.u
                dflux = np.gradient(flux, self.x)
                ax.plot(self.x, flux)
                ax_twin = ax.twinx()
                ax_twin.plot(self.x, dflux, 'r')

        for ax in axes:
            ax.grid(True, linestyle=':')
        plt.tight_layout()
        plt.show()

def construct_finite_diff_matrix(x, edge_order=2):
    """     
    Construct finite difference matrix operator using central differences.
    """
    # Need standard numpy
    import numpy
      
    # Non-uniform grid spacing
    N = x.size
    dx = numpy.diff(x)
    D = numpy.zeros((N, N))
                
    # Off-diagonals for central difference
    dx1 = dx[:-1]
    dx2 = dx[1:]
    a = - (dx2) / (dx1 * (dx1 + dx2))
    b = (dx2 - dx1) / (dx1 * dx2)
    c = dx1 / (dx2 * (dx1 + dx2))
    D[range(1, N-1), range(0, N-2)] = a
    D[range(1, N-1), range(1, N-1)] = b
    D[range(1, N-1), range(2, N)] = c

    # First-order edges
    if edge_order == 1:

        # Left
        a = -1.0 / dx[0]
        b = 1.0 / dx[0]
        D[0,:2] = [a, b]

        # Right
        a = -1.0 / dx[-1]
        b = 1.0 / dx[-1]
        D[-1,-2:] = [a, b]

    # Second-order edges
    elif edge_order == 2:

        # Left
        dx1 = dx[0]
        dx2 = dx[1]
        a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
        b = (dx1 + dx2) / (dx1 * dx2)
        c = - dx1 / (dx2 * (dx1 + dx2))
        D[0,:3] = [a, b, c]

        # Right
        dx1 = dx[-2]
        dx2 = dx[-1]
        a = (dx2) / (dx1 * (dx1 + dx2))
        b = - (dx2 + dx1) / (dx1 * dx2)
        c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
        D[-1,-3:] = [a, b, c]

    else:
        raise ValueError('Invalid order number.')
    
    # Done
    return D

def refine_grid(x, U, thresh=5.0, dx_min=10.0, num_iter=10, verbose=False):
    """
    Refine a grid by the gradient of the velocity field.
    """
    # Compute initial diff
    dU = np.diff(U)

    # Interpolation function
    interp_fun = interp1d(x, U, kind='cubic')

    # Iterate
    N_prev = x.size
    if verbose: print('Original domain size:', N_prev)
    for iternum in range(num_iter):

        # Loop over points
        x_new = [x[0]]
        for i in range(x.size - 1):

            # Get corresponding change in velocities to next grid cell
            dU_next = dU[i]

            # Check criterion
            if (x[i+1] - x[i]) <= (2*dx_min):
                x_new.append(x[i+1])
            elif dU_next > thresh:
                x_new.extend([0.5 * (x[i+1] + x[i]), x[i+1]])
            else:
                x_new.append(x[i+1])

        # Update x coordinates
        x = np.array(x_new)
        N_new = x.size
        if verbose:
            print('Refinement iteration %02d domain size: %d' % (iternum, N_new))

        # Break if no refinement was done
        if N_new == N_prev:
            break
        else:
            N_prev = N_new

        # Re-sample velocity field
        U = interp_fun(x)
        
        # Update differences
        dU = np.diff(U)

    return x
    
def create_sinusoidal_bed_profile(x):
   
    xt = 75.0e3 
    b = make_line(x, x[0], 150.0, xt, -200.0)

    mask = x > xt
    x_periodic = x[mask]
    period = 105000.0
    b_periodic = 300.0 * np.cos(2.0*np.pi/period*(x_periodic - x_periodic[0]))
    bias = b_periodic[0] - (-200.0)
    b_periodic -= bias
    b[mask] = b_periodic

    # Smoothe the line
    b = smoothe_line(x, b, win_size=80)

    return b

def create_sloping_bed_profile(x):

    xt = 120.0e3
    b = make_line(x, x[0], 300.0, xt, -500.0)

    b[x >= xt] = -500.0

    return b

def create_flat_bed(x, b0=-500.0):
    b = b0 * np.ones_like(x)
    return b

def create_surface_profile(x, x_pin=50.0e3, s0=1000.0, amp=750.0, smoothing_win=40):

    # Flat profile leading into logarithmic decay
    s = s0 * np.ones_like(x)
    sub_mask = x > x_pin
    l = x.max() - x
    ssub = amp * np.log(1.0 + l[sub_mask] / 40000.0)
    ssub += s0 - ssub[0]
    s[sub_mask] = ssub

    # Smoothe the line
    s = smoothe_line(x, s, win_size=smoothing_win)

    return s

def smoothe_line(x, y, win_size=20):

    import pandas as pd

    # First create spline representation
    spline = UnivariateSpline(x, y, s=0.001, ext='extrapolate')

    # Extend coordinates to account for smoothing window
    dx = x[1] - x[0]
    x_pad_begin = x[0] - dx - dx * np.arange(win_size)
    x_pad_end = x[-1] + dx + dx * np.arange(win_size)
    x = np.hstack((x_pad_begin[::-1], x, x_pad_end))

    # Extrapolate segment
    y = spline(x)

    # Use pandas to do a rolling window smoothing to smoothe corners
    df = pd.DataFrame({'B': y})
    df_filt = df.rolling(win_size, win_type='triang', center=True).sum()
    y = df_filt.values.squeeze() / (0.5 * win_size)

    return y[slice(win_size, -win_size)]

def make_line(x, x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    C = y1 - slope * x1
    line = slope * x + C
    return line

def advance_terminus(profile, pad=5, verbose=False):
    """
    Advances terminus position to satisfy height above flotation criterion.
    """

    # Extract last <pad> points of profile
    x_origin = profile.x[-1]
    x_term = profile.x[-pad:] - x_origin
    h_term = profile.HAF[-pad:]

    # Fit quadratic to these points
    phi = np.polyfit(x_term, h_term, 2)

    # Compute the roots
    roots = quadratic_roots(phi)
    # Keep the one closest to zero
    argmin = np.argmin(np.abs(roots))
    x_zero = roots[argmin] + x_origin

    # Compute rough number of extra grid points to add
    n_add = int(np.round((x_zero - x_origin) / profile.dx))
    if verbose:
        print('New terminus at', x_zero)
        print('Advancing %d cells' % n_add)

    # Create new grid of x points
    x_new = np.linspace(profile.x[0], x_zero, profile.N + n_add)

    # Check if we need to add an extra point
    dx_thresh = profile.dx + 0.05 * profile.dx
    if (x_new[1] - x_new[0]) > dx_thresh:
        x_new = np.linspace(profile.x[0], x_zero, profile.N + n_add + 1)

    # Create new profile
    new_profile = profile.update_coordinates(x_new, extrapolate=True)
    new_profile.t = profile.t
    if verbose:
        print('New spacing:', new_profile.dx)

    return new_profile

def retreat_terminus(profile, verbose=False):
    """
    Retreats terminus position to satisfy height above flotation criterion.
    """
    # Create spline representation of HAF
    spline = InterpolatedUnivariateSpline(profile.x, profile.HAF)

    # Find roots
    roots = spline.roots()
    if len(roots) > 1:
        raise ValueError('Found multiple zero crossings of HAF')
    x_zero = roots[0]

    # Compute rough number of grid points to remove
    n_remove = int(np.round((profile.x[-1] - x_zero) / profile.dx))
    if verbose:
        print('Retreating %d cells' % n_remove)

    # Create new grid of x points
    x_new = np.linspace(profile.x[0], x_zero, profile.N - n_remove)

    # Check if we need to remove an extra point
    dx_thresh = profile.dx - 0.05 * profile.dx
    if (x_new[1] - x_new[0]) < dx_thresh:
        x_new = np.linspace(profile.x[0], x_zero, profile.N - n_remove - 1)

    # Create new profile
    new_profile = profile.update_coordinates(x_new, extrapolate=False)
    new_profile.t = profile.t
    if verbose:
        print('New spacing:', new_profile.dx)

    return new_profile

def quadratic_roots(phi):
    """
    Computes roots of a quadratic polynomial.
    """
    a, b, c = phi
    r1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    r2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    return r1, r2

# end of file
