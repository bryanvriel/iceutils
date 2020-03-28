#-*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
import scipy.ndimage as ndimage
import cv2 as cv
import pymp
import sys
import os

from .raster import Raster, RasterInfo

def correlate(master_raster, slave_raster, win_x=64, win_y=64, search=20, margin=50,
              skip_x=32, skip_y=32, coarse=False, coarse_over=False, n_proc=1):

    # Get shape information from master image
    hdr = RasterInfo(rasterfile=master_raster)
    ny, nx = hdr.ny, hdr.nx

    # Load master image memory map
    mmap = np.memmap(master_raster, dtype=np.float32, mode='r').reshape(ny, nx)

    # Load slave image memory map
    smap = np.memmap(slave_raster, dtype=np.float32, mode='r').reshape(ny, nx)

    # Make center indices for image chips
    i_chip = np.arange(win_y // 2 + margin, ny - win_y // 2 - margin, skip_y)
    j_chip = np.arange(win_x // 2 + margin, nx - win_x // 2 - margin, skip_x)
    chip_rows = len(i_chip)
    chip_cols = len(j_chip)
    n_chips = chip_rows * chip_cols

    # Convert indices to flattened meshgrids
    j_grid, i_grid = [a.ravel() for a in np.meshgrid(j_chip, i_chip)]

    # Create output header
    X, Y = hdr.meshgrid()
    X_out = X[i_chip, :][:, j_chip]
    Y_out = Y[i_chip, :][:, j_chip]
    out_hdr = RasterInfo(X=X_out, Y=Y_out)

    # Allocate arrays for results
    aoff = pymp.shared.array((chip_rows, chip_cols), dtype='f')
    roff = pymp.shared.array((chip_rows, chip_cols), dtype='f')
    snr = pymp.shared.array((chip_rows, chip_cols), dtype='f')
    print('Output grid shape:', aoff.shape)

    # Create correlator objects
    template_matcher = TemplateMatcher(win_y, win_x, search=search)
    if not coarse and not coarse_over:
        phase_matcher = PhaseRampCorrelator(win_y, win_x)

    # Process chips in parallel
    with pymp.Parallel(n_proc) as manager:
        for k in manager.range(n_chips):

            # Global index for chip center
            i = i_grid[k]
            j = j_grid[k]

            # Go ahead and compute output grid coordinates
            out_row, out_col = np.unravel_index(k, (chip_rows, chip_cols))

            # Chip for master (include search window)
            rows = slice(i - (win_y // 2) - search, i + (win_y // 2) + search)
            cols = slice(j - (win_x // 2) - search, j + (win_x // 2) + search)
            master = mmap[rows, cols]

            # Check if values are large enough
            if np.median(master) < 0.2:
                aoff[out_row, out_col] = np.nan
                roff[out_row, out_col] = np.nan
                snr[out_row, out_col] = np.nan
                continue

            # Chip for slave
            rows = slice(i - (win_y // 2), i + (win_y // 2))
            cols = slice(j - (win_x // 2), j + (win_x // 2))
            slave = smap[rows, cols]

            # Run template maching to get coarse offset
            dx0, dy0, snr_value = template_matcher.correlate(master, slave, oversample=coarse_over)

            # Continue loop if only performing template matching
            if coarse or coarse_over:
                aoff[out_row, out_col] = dy0
                roff[out_row, out_col] = dx0
                snr[out_row, out_col] = snr_value
                continue

            # Keep only interior portion of master chip
            master = master[search:-search, search:-search]

            # Compute phase ramp correlation
            dx, dy, var_reduction = phase_matcher.correlate(master, slave, [-dx0, -dy0])

            # Store result
            aoff[out_row, out_col] = -dy
            roff[out_row, out_col] = -dx
            snr[out_row, out_col] = var_reduction

    # Create output rasters
    a_raster = Raster(data=aoff, hdr=out_hdr)
    r_raster = Raster(data=roff, hdr=out_hdr)
    s_raster = Raster(data=snr, hdr=out_hdr)

    return a_raster, r_raster, s_raster
    

class Correlator:
    """
    Base class for correlation methods. Responsible for measuring offset between
    two image chips.
    """

    def __init__(self, chip_ny, chip_nx, search=0):

        # Cache parameters
        self.chip_ny, self.chip_nx = chip_ny, chip_nx
        self.search = search

    def correlate(self, master, slave, **kwargs):
        raise NotImplementedError('Child classes must implement correlate method.')

class TemplateMatcher(Correlator):
    """
    Compares two image chips using sliding correlation to estimate offset between chips.

    Parameters
    ----------
    chip_ny: int
        Chip dimension in y-direction.
    chip_nx: int
        Chip dimension in x-direction.
    search: int, optional
        Search window. Default: 20.
    zoom: int, optional
        Zoom factor for oversampling. Default: 32.
    zoom_window: int, optional
        Window around peak for oversampling. Default: 8.
    """

    def __init__(self, chip_ny, chip_nx, search=20, zoom=32, zoom_window=8):
        """
        Initialize TemplateMatcher class.
        """
        # Initialize parent class
        super().__init__(chip_ny, chip_nx, search=search)

        # Pre-construct arrays for oversampling correlation surface
        self.zoom = zoom
        self.zoom_window = zoom_window
        x = np.linspace(-zoom_window // 2, zoom_window // 2 - 1, zoom * zoom_window)
        self.x_over, self.y_over = np.meshgrid(x, x)

        return

    def correlate(self, master, slave, oversample=False, method=cv.TM_CCOEFF, order=2):
        """
        Use template matching to construct correlation surface.

        Parameters
        ----------
        master: (N, N) ndarray
            Array for master chip.
        slave: (N, N) ndarray
            Array for slave chip.
        oversample: bool, optional
            Perform oversampling of correlation surface. Default: False.
        method: int, optional
            Enumerated value for OpenCV template matching method. Default: cv.TM_CCOEFF.
        order: int, optional
            Order of spline interpolation. Default: 2.

        Returns
        -------
        dx: float
            Offset in X-direction.
        dy: float
            Offset in Y-direction.
        snr: float
            SNR value.
        """
        # Run template matching to get coarse offset
        corr = cv.matchTemplate(master, slave, method)

        # Oversample correlation surface
        if oversample:
            dx, dy, max_value = self.zoom_peak(corr, order=order)

        # Or just get peak of coarse correlation surface
        else:
            ind = np.argmax(corr)
            imax, jmax = np.unravel_index(ind, corr.shape)
            dx = jmax - self.search
            dy = imax - self.search
            max_value = corr[imax, jmax]

        # Compute SNR value
        snr_value = np.abs(max_value) / np.median(np.abs(corr))

        # Return results
        return dx, dy, snr_value

    def zoom_peak(self, corr, order=2):
        """
        Oversample correlation surface.

        Parameters
        ----------
        corr: (N, N) ndarray
            Coarse correlation surface.
        order: int, optional
            Order of spline interpolation. Default: 2.

        Returns
        -------
        dx: float
            Offset in X-direction.
        dy: float
            Offset in Y-direction.
        """
        # Find peak
        ind = np.argmax(corr)
        imax, jmax = np.unravel_index(ind, corr.shape)

        # Check if peak is within valid area
        if (imax < self.zoom_window or imax > (corr.shape[0] - self.zoom_window) or
            jmax < self.zoom_window or jmax > (corr.shape[1] - self.zoom_window)):
            return np.nan, np.nan, np.nan

        # Get the maximum correlation value
        max_val = corr[imax, jmax]

        # Extract sub-window around peak
        islice = slice(imax - self.zoom_window // 2, imax + self.zoom_window // 2)
        jslice = slice(jmax - self.zoom_window // 2, jmax + self.zoom_window // 2)
        sub = corr[islice, jslice]

        # Oversample
        sub_over = ndimage.zoom(sub, zoom=self.zoom, prefilter=False, order=order)

        # Get peak of oversampled
        ind = np.argmax(sub_over)
        imax_over, jmax_over = np.unravel_index(ind, sub_over.shape)
        dx = self.x_over[imax_over, jmax_over] + jmax - self.search
        dy = self.y_over[imax_over, jmax_over] + imax - self.search

        # Return results
        return dx, dy, max_val


class PhaseRampCorrelator(Correlator):
    """
    Compares two image chips using sliding correlation to estimate offset between chips.

    Parameters
    ----------
    chip_ny: int
        Chip dimension in y-direction.
    chip_nx: int
        Chip dimension in x-direction.
    """

    def __init__(self, chip_ny, chip_nx):

        # Assert chip sizes are equal
        assert chip_ny == chip_nx, 'Chip dimensions must be equal'

        # Initialize parent class with zero search window
        super().__init__(chip_ny, chip_nx, search=0)

        # Construct FFT frequencies
        wx = 2.0 * np.pi * np.fft.fftfreq(chip_nx)
        wx = np.fft.fftshift(wx)
        self.Wx, self.Wy = np.meshgrid(wx, wx)

        # Construct separable 2D hamming filter in frequency domain
        w_filt = np.hamming(chip_nx)
        w_filt[w_filt < 0.01] = 0.01
        self.W_filt = np.outer(w_filt, w_filt)

    def correlate(self, master, slave, m0):
        """
        Phase ramp correlation.

        Parameters
        ----------
        master: (N, N) ndarray
            Array for master chip.
        slave: (N, N) ndarray
            Array for slave chip.
        m0: (2,) array_like
            Initial guess for offset parameters.

        Returns
        -------
        dx: float
            Offset in X-direction.
        dy: float
            Offset in Y-direction.
        var_reduction: float
            Phase ramp variance reduction.
        """
        # Perform 2D FFT on chips
        M = np.fft.fft2(master)
        S = np.fft.fft2(slave)
        M = np.fft.fftshift(M)
        S = np.fft.fftshift(S)

        # Weight by raised cosine to suppress high frequencies
        M *= self.W_filt
        S *= self.W_filt

        # Compute normalized cross-spectrum phase
        prod = M * np.conj(S)
        C = prod / np.abs(prod)
        phase = np.angle(C)

        # Weighting mask
        LS = np.log10(np.abs(prod))
        NLS = LS - np.max(LS)
        weight_mask = NLS > (0.99 * np.median(NLS))
        weight_sum = np.sum(weight_mask)

        # Perform optimization
        res = minimize(self.cost_func, m0, method='Nelder-Mead',
                       args=(C, weight_mask, weight_sum))
        dx, dy = res.x

        # Compute variance reduction as a measure of SNR
        theta = self.Wx * dx + self.Wy * dy
        pred_phase = np.arctan2(np.sin(theta), np.cos(theta))
        misfit = phase - pred_phase
        var_reduction = 1.0 - np.var(misfit) / np.var(phase)

        # Done
        return dx, dy, var_reduction

    def cost_func(self, m, C, weight_mask, weight_sum):
        """
        Computes scalar misfit between complex cross-spectrum array and prediction
        by phase ramp.

        Parameters
        ----------
        m: (2,) array_like
            Array of parameters.
        C: (N, N) ndarray, complex
            Normalized cross-spectrum array.
        weight_mask: (N, N) ndarray
            Weighting array.
        weight_sum: float
            Sum of weighting array.

        Returns
        -------
        cost: float
            Scalar misfit.
        """
        # Unpack parameters
        dx, dy = m

        # Compute prediction
        pred_phase = self.Wx * dx + self.Wy * dy
        pred = np.exp(1j * pred_phase)

        # Residual matrix
        R = np.abs(C - pred)**2 * weight_mask

        # Return Frobenius norm
        cost = np.sum(R) / weight_sum
        return cost


# end of file