#-*- coding: utf-8 -*-

import numpy as np
import sys

def compute_stress_scalars(vx, vy, dem, dx=100, dy=-100):
    """
    Compute stress fields.
    """
    # Magnitude
    vmag = np.sqrt(vx**2 + vy**2)

    # Compute gradients
    L11 = np.gradient(1.0e3*vx, dx, axis=1)
    L12 = np.gradient(1.0e3*vx, dy, axis=0)
    L21 = np.gradient(1.0e3*vy, dx, axis=1)
    L22 = np.gradient(1.0e3*vy, dy, axis=0)
    
    # Compute components of strain-rate tensor
    D11 = 0.5 * (L11 + L11)
    D12 = 0.5 * (L12 + L21)
    D21 = 0.5 * (L21 + L12)
    D22 = 0.5 * (L22 + L22)

    # Compute  components of rotation-rate tensor
    #W12 = 0.5 * (L12 - L21)
    #W21 = 0.5 * (L21 - L12)
    #rotation = np.sqrt(0.5 * (W12**2 + W21**2))

    # Gradients for computing elevation change (want in units of meters per year)
    hL11 = np.gradient(dem * (1.0e3 * vx), dx, axis=1)
    hL22 = np.gradient(dem * (1.0e3 * vy), dy, axis=0)

    # Compute elevation change (in meters per year)
    dhdt = hL11 + hL22

    # Compute scalar quantities from stress tensors
    dilatation = L11 + L22
    effective_strain = np.sqrt(0.5 * (D11**2 + D12**2 + D21**2 + D22**2))

    # Use convention that shear strain is half the difference of eigenvalues of strain-rate tensor
    # For 2x2 matrix, the eigenvalues are:
    #   1/2 * Tr(D) +/- sqrt(Tr(D)**2 / 4 - det(D))
    det = D11 * D22 - D12 * D21
    shear = 2.0 * np.sqrt(0.25 * dilatation**2 - det)

    # Done
    return vmag, dhdt, shear, effective_strain

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
