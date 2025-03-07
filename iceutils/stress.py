#-*- coding: utf-8 -*-

import numpy as np
from itertools import product
from functools import partial
from multiprocessing import Pool
import sys

from .raster import Raster, inpaint as _inpaint

def compute_stress_strain(
    vx,
    vy,
    dx=100,
    dy=-100,
    grad_method='numpy',
    inpaint=True,
    rotate=False,
    h=None,
    b=None,
    AGlen=None,
    BGlen=None,
    rho_ice=917.0,
    g=9.80665,
    n=3,
    eps_min=1.0e-5,
    **kwargs
):
    """
    Compute stress and strain fields and return in dictionaries.

    Parameters
    ----------
    vx: (M, N) ndarray
        Input array for velocities in X-direction.
    vy: (M, N) ndarray
        Input array for velocities in Y-direction.
    dx: float, optional
        Spacing in X-direction.
    dy: float, optional
        Spacing in Y-direction.
    grad_method: str, optional
        Gradient method specifier in ('numpy', 'sgolay', 'robust_l2', 'robust_lp').
        Default: 'numpy'.
    inpaint: bool, optional
        Inpaint arrays prior to gradient computation. Default: True.
    rotate: bool, optional
        Rotate velocity gradients to along-flow direction. Default: False.
    h: (M, N) ndarray, optional
        Input array of ice thickness for computing SSA stresses. Default: None.
    b: (M, N) ndarray, optional
        Input array of bed elevation for computing SSA stresses. Default: None.
    AGlen: float, optional
        Glen's flow law rate parameter in units of {a^-1} {Pa^-3}. Default: None.
    rho_ice: float, optional
        Density of ice in kg/m^3. Default: 917.0
    g: float, optional
        Gravitational constant. Default: 9.80665
    n: int, optional
        Glen's flow law exponent. Default: 3.
    eps_min: float, optional
        Minimum effective strain rate value for numerical stability. Default: 1e-5.
    **kwargs:
        Extra keyword arguments to pass to gradient computation. 

    Returns
    -------
    strain_dict: dict
        Dictionary of (M, N) arrays containing strain components 'e_xx', 'e_yy', 'e_xy',
        'dilatation', 'effective'.
    stress_dict: dict
        Dictionary of (M, N) arrays containing stress components 't_xx', 't_yy', 't_xy'.
        Also contains effective dynamic viscosity 'eta'. If thickness and bed are
        provided, also contains SSA stresses 'tmxx', 'tmxy', 'tmyx', 'tdx', 'tdy', where
        the first three are membrane stresses and the last two are driving stresses.
    
    """
    # Cache image shape
    Ny, Nx = vx.shape

    # Compute velocity gradients
    L12, L11 = gradient(
        vx, spacing=(dy, dx), method=grad_method, inpaint=inpaint, **kwargs
    )
    L22, L21 = gradient(
        vy, spacing=(dy, dx), method=grad_method, inpaint=inpaint, **kwargs
    )

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
    exy = D12.reshape(Ny, Nx)

    # Shear- same result as
    # -> e_xy_max = |e_max - e_min|
    # -> e_xy_max = 2 * np.sqrt(0.25 * (e_x - e_y)**2 + e_xy**2)
    trace = (D11 + D22).reshape(Ny, Nx)
    det = (D11 * D22 - D12 * D21).reshape(Ny, Nx)
    shear = np.sqrt(0.25 * trace**2 - det)

    # Compute scalar quantities from stress tensors
    dilatation = (L11 + L22).reshape(Ny, Nx)
    effective_strain = np.sqrt(
        L11**2 + L22**2 + 0.25 * (L12 + L21)**2 + L11 * L22 + eps_min**2
    ).reshape(Ny, Nx)
    
    # Principal strain rates. Equivalent to
    # e_max/min = 0.5 * (e_x + e_y) +/- sqrt(0.25 * (e_x - e_y)**2 + e_xy**2)
    e_max = 0.5 * trace + 0.5 * shear
    e_min = 0.5 * trace - 0.5 * shear

    # Store strain components in dictionary
    strain_dict = {'e_xx': exx,
                   'e_yy': eyy,
                   'e_xy': exy,
                   'shear_magnitude': shear,
                   'dilatation': dilatation,
                   'effective': effective_strain,
                   'e_max': e_max,
                   'e_min': e_min,}

    # Compute AGlen if not provided
    if BGlen is None:
        if AGlen is None:
            from .sim import AGlen_vs_temp
            AGlen = AGlen_vs_temp(-10.0)
        BGlen = AGlen ** (-1 / n)

    # Effective viscosity
    scale_factor = 0.5 * BGlen
    eta = scale_factor * effective_strain**((1.0 - n) / n)

    # Compute PHYSICAL stress tensor comonents
    txx = 2.0 * eta * D11.reshape(Ny, Nx)
    tyy = 2.0 * eta * D22.reshape(Ny, Nx)
    txy = 2.0 * eta * D12.reshape(Ny, Nx)
    tyx = 2.0 * eta * D21.reshape(Ny, Nx)

    # Create stress dictionary
    stress_dict = {'eta': eta,
                   't_xx': txx,
                   't_xy': 0.5 * (txy + tyx),
                   't_yy': tyy}

    # Compute SSA stress components if thickness and bed provided
    if h is not None:

        # For surface, add bed if it exists to compute driving stress
        if b is not None:
            s = h + b
            s_x = gradient(s, dx, axis=1, method=grad_method, inpaint=inpaint,
                           remask=False, **kwargs)
            s_y = gradient(s, dy, axis=0, method=grad_method, inpaint=inpaint,
                           remask=False, **kwargs)

        # Otherwise, compute thickness gradients as surface gradients
        else:
            s_x = gradient(h, dx, axis=1, method=grad_method, inpaint=inpaint,
                           remask=False, **kwargs)
            s_y = gradient(h, dy, axis=0, method=grad_method, inpaint=inpaint,
                           remask=False, **kwargs)

        # Membrane stresses
        tmxx = gradient(h * (2 * txx + tyy), dx, axis=1, method=grad_method,
                        inpaint=inpaint, **kwargs)
        tmxy = gradient(h * 0.5 * (txy + tyx), dy, axis=0, method=grad_method,
                        inpaint=inpaint, **kwargs)
        tmyy = gradient(h * (2 * tyy + txx), dy, axis=0, method=grad_method,
                        inpaint=inpaint, **kwargs)
        tmyx = gradient(h * 0.5 * (txy + tyx), dx, axis=1, method=grad_method,
                        inpaint=inpaint, **kwargs)

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


def gradient(z, spacing=1.0, axis=None, remask=True, method='numpy',
             inpaint=False, **kwargs):
    """
    Calls either Numpy or Savitzky-Golay gradient computation routines.

    Parameters
    ----------
    z: Raster or array_like
        2-dimensional raster or array containing samples of a scalar function.
    spacing: float or tuple of floats, optional
        Spacing between f values along specified axes. If tuple provided, spacing corresponds
        to axes directions specified by axis. Default: 1.0.
    axis: None or int or tuple of ints, optional
        Axis or axes to compute gradient. If None, derivative computed along all
        dimensions. Default: 0.
    remask: bool, optional
        Apply NaN mask on gradients. Default: True.
    method: str, optional
        Method specifier in ('numpy', 'sgolay', 'robust'). Default: 'numpy'.
    inpaint: bool, optional
        Inpaint image prior to gradient computation (recommended for
        'sgolay' method). Default: False.
    **kwargs:
        Extra keyword arguments to pass to specific gradient computation.

    Returns
    -------
    s: ndarray or list of ndarray
        Set of ndarrays (or single ndarry for only one axis) with same shape as z
        corresponding to the derivatives of z with respect to each axis.
    """
    # Check if a raster has been passed
    if isinstance(z, Raster):
        spacing = z.hdr.spacing
        z = z.data.copy()

    # Mask mask of NaNs
    nan_mask = np.isnan(z)
    have_nan = np.any(nan_mask)

    # For sgolay method, we need to inpaint if NaNs detected
    if inpaint and have_nan:
        z_inp = _inpaint(z, mask=nan_mask)
    else:
        z_inp = z

    # Compute gradient with numpy
    if method == 'numpy':
        if isinstance(spacing, (tuple, list)):
            s = np.gradient(z_inp, spacing[0], spacing[1], axis=(0, 1), edge_order=2)
        else:
            s = np.gradient(z_inp, spacing, axis=axis, edge_order=2)

    # With Savtizky-Golay
    elif method == 'sgolay':
        s = sgolay_gradient(z_inp, spacing=spacing, axis=axis, **kwargs)

    # With robust polynomial
    elif method in ('robust_l2', 'robust_lp'):
        zs, z_dy, z_dx = robust_gradient(z_inp, spacing=spacing, lsq_method=method, **kwargs)
        s = (z_dy, z_dx)
        if axis is not None and isinstance(axis, int):
            s = s[axis]

    else:
        raise ValueError('Unsupported gradient method.')

    # Re-apply mask
    if remask and have_nan:
        if isinstance(s, (tuple, list)):
            for arr in s:
                arr[nan_mask] = np.nan
        else:
            s[nan_mask] = np.nan

    return s


def sgolay_gradient(z, spacing=1.0, axis=None, window_size=3, order=4, pad='reflect'):
    """
    Wrapper around Savitzky-Golay code to compute window size in pixels and call _sgolay2d
    with correct arguments.

    Parameters
    ----------
    z: array_like
        2-dimensional array containing samples of a scalar function.
    spacing: float or tuple of floats, optional
        Spacing between f values along specified axes. If tuple provided, spacing is
        specified as (dy, dx) and derivative computed along both dimensions.
        Default is unitary spacing.
    axis: None or int, optional
        Axis along which to compute gradients. If None, gradient computed
        along both dimensions. Default: None.
    window_size: scalar or tuple of scalars, optional
        Window size in units of specified spacing. If tuple provided, window size is
        specified as (win_y, win_x). Default: 3.
    order: int, optional
        Polynomial order. Default: 4.
    pad: str, optional
        Option for np.pad kwarg 'mode'. Default: 'reflect'.

    Returns
    -------
    gradient: a
        Array or tuple of array corresponding to gradients. If both axes directions
        are specified, returns (dz/dy, dz/dx).
    """
    # Compute derivatives in both directions
    if axis is None or isinstance(spacing, (tuple, list)):

        # Compute window sizes
        wy, wx = compute_windows(window_size, spacing)

        # Unpack spacing
        dy, dx = spacing
        
        # Call Savitzky-Golay twice in order to use different window sizes
        sy = _sgolay2d(z, wy, order=order, derivative='col', pad=pad)
        sx = _sgolay2d(z, wx, order=order, derivative='row', pad=pad)

        # Scale by spacing and return
        return sy / dy, sx / dx

    # Or derivative in a single direction
    else:

        assert axis is not None, 'Must specify axis direction.'

        # Compute window size
        w = int(np.ceil(abs(window_size / spacing)))
        if w % 2 == 0:
            w += 1

        # Call Savitzky-Golay
        if axis == 0:
            s = _sgolay2d(z, w, order=order, derivative='col', pad=pad)
        elif axis == 1:
            s = _sgolay2d(z, w, order=order, derivative='row', pad=pad)
        else:
            raise ValueError('Axis must be 0 or 1.')

        # Scale by spacing and return
        return s / spacing


def robust_gradient(z, spacing=1.0, window_size=3.0, order=2,
                    ftol=1e-5, std_thresh=3, maxiter=9, lsq_method='robust_l2', njobs=None):
    """ A robust filter that minimizes the misfit (Gx - d) using either iterative
    least squares (outlier removal) or iterative re-weighted least squares for Lp-norm
    approximation.

    The function is multiprocessed and is robust to Laplace or Gaussian 
    noise. NaN values are ignored when computing the IRLS optimization.

    Parameters
    ----------
    z : array_like
        2-dimensional array containing samples of a scalar function.
    spacing: float or tuple of floats, optional
        Spacing between pixels along axis. If tuple, each element specifies
        spacing along different axes. Default: 1.0.
    window_size: scalar or array_like, optional
        Window size in units of specified spacing. If array, must be same shape as z
        and specifies window size at each pixel. Default: 3.0.
    order : int or array_like, optional
        Polynomial order for the robust fitting. If array, must be same shape as z
        and specifies order at each pixel. Default: 2.
    std_thresh : float, optional
        Number of standard deviations above which defines outliers (for lsq_method='robust_l2').
        Default: 3.0.
    ftol : float, optional
        Accepted mean residual stopping point (for lsq_method='robust_lp').
    maxiter : int, optional
        Maximum number of iterations of IRLS when approximating the L1-norm.
        Setting maxiter to 0 will return the coefficient solution to minimizing
        the L2-norm.
    lsq_method : str, optional
        Least squares method ('robust_l2', 'robust_lp'). Default: 'robust_l2'.
    njobs : int or None, optional
        Number of processes to use. If None, os.cpu_count() is used. 

    Returns
    -------
    z_smooth : array_like
        2-dimensional array of smoothed data.
    z_dy : array_like
        2-dimensional array with the column gradient.
    z_dx : array_like
        2-dimensional array with the row gradient.
    """
    nrow, ncol = np.shape(z)

    # Unpack spacing
    if isinstance(spacing, (tuple, list)):
        dy, dx = spacing
    else:
        dy = dx = spacing
    
    # Build polynomial matrix using maximum order and window size
    if isinstance(order, np.ndarray):
        max_order = np.max(order)
    else:
        max_order = order
        order = np.full(z.shape, order, dtype=np.int16)
    if isinstance(window_size, np.ndarray):
        max_win_size = np.max(window_size)
    else:
        max_win_size = window_size
        window_size = np.full(z.shape, window_size, dtype=z.dtype)

    # Get maximum window sizes in both directions
    max_win_size_y, max_win_size_x = compute_windows(max_win_size, spacing)
    max_half_size_y = max_win_size_y // 2
    max_half_size_x = max_win_size_x // 2

    # The center of the window is at (0, 0)
    window_x = np.arange(-max_half_size_x, max_half_size_x+1)
    window_y = np.arange(-max_half_size_y, max_half_size_y+1)

    # Build design matrix using max order and max window size
    exps = _compute_exps(max_order)
    ncoef = len(exps)
    G = np.empty((ncoef, max_win_size_y, max_win_size_x))
    for i, exp in enumerate(exps):
        G[i,:,:] = np.outer(window_y**exp[0], window_x**exp[1])
    
    # Get coordinates for each data point in padded array
    Zy = np.arange(max_half_size_y, nrow + max_half_size_y)
    Zx = np.arange(max_half_size_x, ncol + max_half_size_x)
    coords = product(Zy, Zx)

    # Now compute pixel-dependent window sizes
    win_size_y, win_size_x = compute_windows(window_size, spacing)
    half_size_y = win_size_y // 2
    half_size_x = win_size_x // 2

    # Pad all data by maximum half sizes (such that they share same coordinates)
    pad = (max_half_size_y, max_half_size_x)
    Z, order, half_size_y, half_size_x = [np.pad(arr, pad, mode='edge') for arr in 
                                          (z, order, half_size_y, half_size_x)]

    # Cache least squares method
    if lsq_method == 'robust_l2':
        func = partial(_run_ILS, Z, G, order, half_size_y, half_size_x,
                       max_half_size_y, max_half_size_x, std_thresh, maxiter)
    elif lsq_method == 'robust_lp':
        func = partial(_run_IRLS, Z, G, order, half_size_y, half_size_x,
                       max_half_size_y, max_half_size_x, ftol, maxiter)
    else:
        raise ValueError('Unsupported least squares method.')

    # Apply regression to each pixel in data
    with Pool(processes=njobs) as p:
        x = list(p.map(func, coords))

    # Get zero- and first-order coefficients
    x = np.reshape(x, (nrow, ncol, 3))
    z_smooth = x[:, :, 0]
    z_dy = x[:, :, 1] / dy
    z_dx = x[:, :, 2] / dx

    return z_smooth, z_dy, z_dx


def _run_IRLS(Z, G, order_map, half_size_y_map, half_size_x_map, max_half_size_y, max_half_size_x,
              ftol, maxiter, coord):
    """ Iterative Reweighted Least Squares to minimize the L1 norm 
    for a window surrounding a pixel. --> ||Gx-d||_1

    Paramters
    ---------
    Z : array_like
        2-dimensional array of data with padding
    G : array_like
        3-dimensional design matrix of shape (ncoef, win_size_y, win_size_x).
    order_map: array_like
        2-dimensional integer array of polynomial order with padding.
    half_size_y_map: array_like
        2-dimensional integer array of half-window length in y-direction with padding.
    half_size_x_map: array_like
        2-dimensional integer array of half-window length in x-direction with padding.
    max_half_size_y: int
        Maximum half-window length in y-direction.
    max_half_size_x: int
        Maximum half-window length in x-direction.
    ftol : float
        accepted mean residual stopping point
    maxiter : int
        maximum number of iterations of IRLS when approximating the L1 norm.
        Setting maxiter to 0 will return the coefficient solution to minimizing
        the L2-norm.
    coord : tuple
        the (x, y) coordinate in Z representing the data point around which
        to create the window
    
    Returns
    -------
    x : array_like
        1-dimensional array of coefficients of the polynomial fit that
        minimizes the L1-norm ||Gx-d||_1 where (0, 0) represents the relative
        position of the coordinate at the center of the window.

        In the window's relative coordinate system, (0, 0) represents the
        center point of interest. This is neat because it allows for easy 
        evaluation of the smoothed value as well as higher x and y 
        derivatives of z.

        Ex. coefs = [a0, a1, a2]
        z = a0 + a1*y + a2*x
        smooth  @ (0, 0) = a0
        dz/dy   @ (0, 0) = a1
        dz/dx   @ (0, 0) = a2
        etc...
    """
    # For all cases, we only return first three coefficients associated
    # with zero- and first-order polynomials
    N_par_return = 3

    # Don't fit if the center coordinate is nan
    if np.isnan(Z[coord]):
        return np.full(N_par_return, np.nan)

    # Get local order and window size
    row, col = coord
    order = order_map[row, col]
    half_size_y = half_size_y_map[row, col]
    half_size_x = half_size_x_map[row, col]

    # Calculate window bounds
    rstart, rend = row - half_size_y, row + half_size_y + 1
    cstart, cend = col - half_size_x, col + half_size_x + 1

    # Get sub-window of data surrounding coordinate
    d = Z[rstart:rend, cstart:cend]

    # Get correct subset of polynomial design matrix (accounting for order and window size)
    ncoef = len(_compute_exps(order))
    rstart = max_half_size_y - half_size_y
    rend = rstart + d.shape[0]
    cstart = max_half_size_x - half_size_x
    cend = cstart + d.shape[1]
    G = G[:ncoef, rstart:rend, cstart:cend].reshape(-1, d.size)

    # Flatten data and transpose design matrix
    d = d.ravel()
    G = G.T

    # Ignore any nan values
    nan_mask = np.isnan(d).nonzero()[0]
    if (len(nan_mask) > 0):
        G = np.delete(G, nan_mask, axis=0)
        d = np.delete(d, nan_mask)

    # x is solution coefficient matrix
    x = np.matmul(np.linalg.pinv(G), d)

    # Approximate the L1 norm using IRLS
    for _ in range(maxiter):
        # Calculate residual
        res = np.matmul(G,x) - d

        if np.mean(res) < ftol:
            break # Good enough solution achieved

        # weights vector for L1 approximation
        w = abs(res)**(-0.5)
        W = np.diag(w/sum(w))
        WG = np.matmul(W, G)

        try:
            # New approximate coefficient solution
            x, _, _, _ = np.linalg.lstsq(np.matmul(WG.T, WG),
                np.matmul(np.matmul(WG.T, W), d))
        except:
            # Break if solution can't be found (ie residuals become too small)
            break

    return x[:N_par_return]


def _run_ILS(Z, G, order_map, half_size_y_map, half_size_x_map, max_half_size_y, max_half_size_x,
             std_thresh, maxiter, coord):
    """ Iterative Least Squares where outliers are removed in successive iterations.

    Paramters
    ---------
    Z : array_like
        2-dimensional array of data with padding
    G : array_like
        3-dimensional design matrix of shape (ncoef, max_win_size_y, max_win_size_x). 
    order_map: array_like
        2-dimensional integer array of polynomial order with padding.
    half_size_y_map: array_like
        2-dimensional integer array of half-window length in y-direction with padding.
    half_size_x_map: array_like
        2-dimensional integer array of half-window length in x-direction with padding.
    max_half_size_y: int
        Maximum half-window length in y-direction.
    max_half_size_x: int
        Maximum half-window length in x-direction.
    std_thresh: float
        number of standard deviations above which defines outliers.
    maxiter : int
        maximum number of iterations of outlier detection and removal.
    coord : tuple
        the (x, y) coordinate in Z representing the data point around which
        to create the window
    
    Returns
    -------
    x : array_like
        1-dimensional array of coefficients of the polynomial fit that
        minimizes the L2-norm ||Gx-d||_2 where (0, 0) represents the relative
        position of the coordinate at the center of the window.

        In the window's relative coordinate system, (0, 0) represents the
        center point of interest. This is neat because it allows for easy 
        evaluation of the smoothed value as well as higher x and y 
        derivatives of z.

        Ex. coefs = [a0, a1, a2]
        z = a0 + a1*y + a2*x
        smooth  @ (0, 0) = a0
        dz/dy   @ (0, 0) = a1
        dz/dx   @ (0, 0) = a2
        etc...
    """
    # For all cases, we only return first three coefficients associated
    # with zero- and first-order polynomials
    N_par_return = 3

    # Don't fit if the value at center coordinate is nan
    if np.isnan(Z[coord]):
        return np.full(N_par_return, np.nan)

    # Get local order and window size
    row, col = coord
    order = order_map[row, col]
    half_size_y = half_size_y_map[row, col]
    half_size_x = half_size_x_map[row, col]

    # Calculate window bounds
    rstart, rend = row - half_size_y, row + half_size_y + 1
    cstart, cend = col - half_size_x, col + half_size_x + 1

    # Get sub-window of data surrounding coordinate
    d = Z[rstart:rend, cstart:cend]

    # Get correct subset of polynomial design matrix (accounting for order and window size)
    ncoef = len(_compute_exps(order))
    rstart = max_half_size_y - half_size_y
    rend = rstart + d.shape[0]
    cstart = max_half_size_x - half_size_x
    cend = cstart + d.shape[1]
    G = G[:ncoef, rstart:rend, cstart:cend].reshape(-1, d.size)

    # Flatten data and transpose design matrix
    d = d.ravel()
    G = G.T

    # Keep only finite values
    fmask = np.isfinite(d)
    Gf = G[fmask, :]
    df = d[fmask]
    if df.size < ncoef:
        return np.full(N_par_return, np.nan)

    # Iterative least squares
    for _ in range(maxiter):
    
        # Perform least squares
        x = np.linalg.lstsq(Gf, df, rcond=None)[0]

        # Calculate residual
        res = np.dot(Gf, x) - df

        # Detect outliers
        std = np.std(res)
        outliers = np.abs(res) > (std_thresh * std)
        # Break if no outliers detected
        n_outliers = np.sum(outliers)
        if n_outliers == 0:
            break

        # Subset inliers
        inliers = np.invert(outliers)
        Gf = Gf[inliers, :]
        df = df[inliers]
        if df.size < ncoef:
            return np.full(N_par_return, np.nan)
        
    return x[:N_par_return]


def _sgolay2d(z, window_size, order, derivative=None, pad='reflect'):
    """
    Max Filter, January 2021.

    Original lower-level code from Scipy cookbook, with modifications to
    padding.
    """
    from scipy.signal import fftconvolve

    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

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
    Z = np.pad(z, half_size, mode=pad)

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        Zf = fftconvolve(Z, m, mode='valid')
        return Zf
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        Zc = fftconvolve(Z, -c, mode='valid')
        return Zc
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        Zr = fftconvolve(Z, -r, mode='valid')
        return Zr
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        Zc = fftconvolve(Z, -c, mode='valid')
        Zr = fftconvolve(Z, -r, mode='valid')
        return Zr, Zc


def compute_windows(window_size, spacing):
    """
    Convenience function to compute window sizes in pixels given spacing of
    pixels in physical coordinates.

    Parameters
    ----------
    spacing: scalar or tuple of scalars
        Spacing between pixels along axis. If tuple, each element specifies
        spacing along different axes.
    window_size: scalar or tuple of scalars
        Window size in units of specified spacing. If tuple provided, window size is
        specified as (win_y, win_x).

    Returns
    -------
    w: tuple of scalars
        Odd-number window sizes in both axes directions in number of pixels.
    """
    # Unpack spacing
    if isinstance(spacing, (tuple, list)):
        assert len(spacing) == 2, 'Spacing must be 2-element tuple.'
        dy, dx = spacing
    else:
        dy = dx = spacing

    # Compute window sizes
    if isinstance(window_size, (tuple, list)):
        assert len(window_size) == 2, 'Window size must be 2-element tuple.'
        wy, wx = window_size
    else:
        wy = wx = window_size

    # Array of windows
    if isinstance(wy, np.ndarray):
        wy = np.ceil(np.abs(wy / dy)).astype(int)
        wx = np.ceil(np.abs(wx / dx)).astype(int)
        wy[(wy % 2) == 0] += 1
        wx[(wx % 2) == 0] += 1

    # Or scalar windows
    else:
        wy, wx = int(np.ceil(abs(wy / dy))), int(np.ceil(abs(wx / dx)))
        wy = _make_odd(wy)
        wx = _make_odd(wx)

    return wy, wx


def _make_odd(w):
    """
    Convenience function to ensure a number is odd.
    """
    if w % 2 == 0:
        w += 1
    return w

# Lambda for getting order-dependent polynomial exponents
_compute_exps = lambda order: [(k-n, n) for k in range(order + 1) for n in range(k + 1)]


# end of file
