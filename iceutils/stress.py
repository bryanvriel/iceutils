#-*- coding: utf-8 -*-

import numpy as np
import sys

def compute_stress_strain(vx, vy, dx=100, dy=-100, h=None, b=None, AGlen=None,
                          rho_ice=917.0, g=9.80665, rotate=False, n=3):
    """
    Compute stress and strain fields and return in dictionaries.
    """
    # Cache image shape
    Ny, Nx = vx.shape
 
    # Compute velocity gradients
    L11 = np.gradient(vx, dx, axis=1)
    L12 = np.gradient(vx, dy, axis=0)
    L21 = np.gradient(vy, dx, axis=1)
    L22 = np.gradient(vy, dy, axis=0)

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

    # Compute stress components if thickness and bed provided
    if h is not None:

        # Compute thickness gradients
        h_x = np.gradient(h, dx, axis=1)
        h_y = np.gradient(h, dy, axis=0)

        # For surface, add bed if it exists to compute driving stress
        if b is not None:
            s = h + b
            s_x = np.gradient(s, dx, axis=1)
            s_y = np.gradient(s, dy, axis=0)
        else:
            s_x = h_x
            s_y = h_y

        # Compute AGlen if not provided
        if AGlen is None:
            AGlen = ice.sim.AGlen_vs_temp(-10.0)

        # Effective viscosity
        strain = np.sqrt(L11**2 + L22**2 + 0.25 * (L12 + L21)**2 + L11 * L22)
        scale_factor = 0.5 * AGlen**(-1 / n)
        eta = scale_factor * strain**((1.0 - n) / n)

        # Compute PHYSICAL stress tensor comonents
        txx = 2.0 * eta * D11.reshape(Ny, Nx)
        tyy = 2.0 * eta * D22.reshape(Ny, Nx)
        txy = 2.0 * eta * D12.reshape(Ny, Nx)
        tyx = 2.0 * eta * D21.reshape(Ny, Nx)

        # Membrane stresses
        tmxx = np.gradient(h * (2 * txx + tyy), dx, axis=1)
        tmxy = np.gradient(h * 0.5 * (txy + tyx), dy, axis=0)
        tmyy = np.gradient(h * (2 * tyy + txx), dy, axis=0)
        tmyx = np.gradient(h * 0.5 * (txy + tyx), dx, axis=1)

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

        # Pack stress
        stress_dict = {'txx': txx,
                       'txy': 0.5 * (txy + tyx),
                       'tyy': tyy,
                       'tmxx': tmxx,
                       'tmxy': tmxy,
                       'tmyy': tmyy,
                       'tmyx': tmyx,
                       'tdx': tdx,
                       'tdy': tdy}

        return strain_dict, stress_dict

    else:
        return strain_dict


def sgolay2d ( z, window_size, order, derivative=None):
    """
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
