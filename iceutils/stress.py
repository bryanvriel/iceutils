#-*- coding: utf-8 -*-

import numpy as np
import sys

def compute_stress_strain(vx, vy, dx=100, dy=-100, grad_method='numpy', window_size=3,
                          h=None, b=None, AGlen=None, rho_ice=917.0, g=9.80665, rotate=False, n=3):
    """
    Compute stress and strain fields and return in dictionaries.
    """
    # Cache image shape
    Ny, Nx = vx.shape

    # Compute velocity gradients
    L11 = gradient(vx, dx, axis=1, window_size=window_size, method=grad_method)
    L12 = gradient(vx, dy, axis=0, window_size=window_size, method=grad_method)
    L21 = gradient(vy, dx, axis=1, window_size=window_size, method=grad_method)
    L22 = gradient(vy, dy, axis=0, window_size=window_size, method=grad_method)

    # Compute components of strain-rate tensor
    D = np.empty((2, 2, vx.size))
    D[0, 0, :] = 0.5 * (L11 + L11).ravel()
    D[0, 1, :] = 0.5 * (L12 + L21).ravel()
    D[1, 0, :] = 0.5 * (L21 + L12).ravel()
    D[1, 1, :] = 0.5 * (L22 + L22).ravel()

    # Compute pixel-dependent rotation tensor if requested
    if rotate:
        R = np.empty((2, 2, vx.size))
        theta = np.arctan2(vy, vx).ravel()
        R[0, 0, :] = np.cos(theta)
        R[0, 1, :] = np.sin(theta)
        R[1, 0, :] = -np.sin(theta)
        R[1, 1, :] = np.cos(theta)

        # Apply rotation tensor
        D = np.einsum('ijm,kjm->ikm', D, R)
        D = np.einsum('ijm,jkm->ikm', R, D)

    # Cache elements of strain-rate tensor for easier viewing
    D11 = D[0, 0, :]
    D12 = D[0, 1, :]
    D21 = D[1, 0, :]
    D22 = D[1, 1, :]

    # Normal strain rates
    exx = D11.reshape(Ny, Nx)
    eyy = D22.reshape(Ny, Nx)

    # Shear- same result as: e_xy_max = np.sqrt(0.25 * (e_x - e_y)**2 + e_xy**2)
    trace = D11 + D22
    det = D11 * D22 - D12 * D21
    shear = np.sqrt(0.25 * trace**2 - det).reshape(Ny, Nx)

    # Compute scalar quantities from stress tensors
    dilatation = (L11 + L22).reshape(Ny, Nx)
    effective_strain = np.sqrt(0.5 * (D11**2 + D12**2 + D21**2 + D22**2)).reshape(Ny, Nx)

    # Store strain components in dictionary
    strain_dict = {'e_xx': exx,
                   'e_yy': eyy,
                   'e_xy': shear,
                   'dilatation': dilatation,
                   'effective': effective_strain}

    # Compute AGlen if not provided
    if AGlen is None:
        from .sim import AGlen_vs_temp
        AGlen = AGlen_vs_temp(-10.0)

    # Effective viscosity
    strain = np.sqrt(L11**2 + L22**2 + 0.25 * (L12 + L21)**2 + L11 * L22)
    scale_factor = 0.5 * AGlen**(-1 / n)
    eta = scale_factor * strain**((1.0 - n) / n)

    # Compute PHYSICAL stress tensor comonents
    txx = 2.0 * eta * D11.reshape(Ny, Nx)
    tyy = 2.0 * eta * D22.reshape(Ny, Nx)
    txy = 2.0 * eta * D12.reshape(Ny, Nx)
    tyx = 2.0 * eta * D21.reshape(Ny, Nx)

    # Create stress dictionary
    stress_dict = {'eta': eta,
                   'txx': txx,
                   'txy': 0.5 * (txy + tyx),
                   'tyy': tyy}

    # Compute SSA stress components if thickness and bed provided
    if h is not None:

        # Compute thickness gradients
        h_x = gradient(h, dx, axis=1, window_size=window_size, method=grad_method)
        h_y = gradient(h, dy, axis=0, window_size=window_size, method=grad_method)

        # For surface, add bed if it exists to compute driving stress
        if b is not None:
            s = h + b
            s_x = gradient(s, dx, axis=1, window_size=window_size, method=grad_method)
            s_y = gradient(s, dy, axis=0, window_size=window_size, method=grad_method)
        else:
            s_x = h_x
            s_y = h_y

        # Membrane stresses
        tmxx = gradient(h * (2 * txx + tyy), dx, axis=1,
                        window_size=window_size, method=grad_method)
        tmxy = gradient(h * 0.5 * (txy + tyx), dy, axis=0,
                        window_size=window_size, method=grad_method)
        tmyy = gradient(h * (2 * tyy + txx), dy, axis=0,
                        window_size=window_size, method=grad_method)
        tmyx = gradient(h * 0.5 * (txy + tyx), dx, axis=1,
                        window_size=window_size, method=grad_method)

        # Driving stresses
        tdx = -1.0 * rho_ice * g * h * s_x
        tdy = -1.0 * rho_ice * g * h * s_y

        # Optional rotation of driving stress
        if rotate:

            # Compute unit vectors
            vmag = np.sqrt(vx**2 + vy**2)
            uhat = vx / vmag
            vhat = vy / vmag

            # Rotate to along-flow, across-flow
            tdx2 = tdx * uhat + tdy * vhat
            tdy2 = tdx * (-vhat) + tdy * uhat
            tdx, tdy = tdx2, tdy2

        # Pack extra stress components
        stress_dict['tmxx'] = tmxx
        stress_dict['tmxy'] = tmxy
        stress_dict['tmyy'] = tmyy
        stress_dict['tmyx'] = tmyx
        stress_dict['tdx'] = tdx
        stress_dict['tdy'] = tdy

    # Return strain and stress dictionaries
    return strain_dict, stress_dict


def gradient(z, spacing, axis=0, window_size=3, method='numpy'):
    """
    Calls either Numpy or Savitzky-Golay gradient computation routines.

    Parameters
    ----------
    z: array_like
        2-dimensional array containing samples of a scalar function.
    spacing: float or tuple of floats, optional
        Spacing between f values along specified axes. If tuple provided, spacing is
        specified as (dy, dx). Default is unitary spacing.
    window_size: int, optional
        Window size for Savitzky-Golay in units of specified spacing. Default: 3.
    method: str, optional
        Method specifier in ('numpy', 'sgolay'). Default: 'numpy'.

    Returns
    -------
    s: array_like
        Gradient of z along specified axis.
    """
    if method == 'numpy':
        s = np.gradient(z, spacing, axis=axis, edge_order=2)
    else:
        s = sgolay_gradient(z, spacing=spacing, axis=axis, window_size=window_size)
    return s


def sgolay_gradient(z, spacing=1.0, axis=0, window_size=3, order=4):
    """
    Wrapper around Savitzky-Golay code to compute window size in pixels and call _sgolay2d
    with correct arguments.

    Parameters
    ----------
    z: array_like
        2-dimensional array containing samples of a scalar function.
    spacing: float or tuple of floats, optional
        Spacing between f values along specified axes. If tuple provided, spacing is
        specified as (dy, dx). Default is unitary spacing.
    axis: int or str, optional
        Axis along which to compute gradients. If axis is 'both', gradient computed
        along both dimensions. Default: 0.
    window_size: scalar, optional
        Window size in units of specified spacing. Default: 3.
    order: int, optional
        Polynomial order. Default: 4.

    Returns
    -------
    vargs: array_like
        Array or tuple of array corresponding to gradients. If both axes directions
        are specified, returns (dz/dy, dz/dx).
    """
    # Compute derivatives in both directions
    if isinstance(spacing, (tuple, list)):

        # Unpack spacing
        assert len(spacing) == 2, 'Spacing must be 2-element tuple.'
        dy, dx = spacing

        # Compute window sizes
        if isintance(window_size, (tuple, list)):
            assert len(window_size) == 2, 'Window size must be 2-element tuple.'
            wy, wx = window_size
        else:
            wy = wx = window_size
        wy, wx = int(np.ceil(abs(wy / dy))), int(np.ceil(abs(wx / dx)))

        # Ensure odd windows
        wy = _make_odd(wy)
        wx = _make_odd(wx)

        # Call Savitzky-Golay twice in order to use different window sizes
        sy = _sgolay2d(z, wy, order=order, derivative='col')
        sx = _sgolay2d(z, wx, order=order, derivative='row')

        # Scale by spacing and return
        return sy / dy, sx / dx

    # Or derivative in a single direction
    else:

        # Compute window size
        w = _make_odd(int(np.ceil(abs(window_size / spacing))))

        # Call Savitzky-Golay
        if axis == 0:
            s = _sgolay2d(z, w, order=order, derivative='col')
        elif axis == 1:
            s = _sgolay2d(z, w, order=order, derivative='row')
        else:
            raise ValueError('Axis must be 0 or 1.')

        # Scale by spacing and return
        return s / spacing


def _make_odd(w):
    """
    Convenience function to ensure a numbed is odd.
    """
    if w % 2 == 0:
        w += 1
    return w


def _sgolay2d(z, window_size, order, derivative=None):
    """
    Original lower-level code from Scipy cookbook.
    """
    from scipy.signal import fftconvolve

    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(Z, -r, mode='valid'), fftconvolve(Z, -c, mode='valid')


# end of file
