#-*- coding: utf-8 -*-

import numpy as np
import h5py

from ..timeutils import tdec2datestr, datestr2tdec
from . import timefn

class Model:
    """
    Class for handling time series predictions.
    """

    def __init__(self, t, collection=None, t0=None, tf=None, rank=0):
        """
        Initialize the Model class with a TimeRepresentation object.

        Parameters
        ----------
        t: array type
            Array of observation times as datetime objects
        collection: {giant.utilities.timefn.TimefnCollection, None}, optional
            GIAnT TimefnCollection instance. If None, constructs a polynomial collection.
        t0: datetime or None
            Starting date for estimating parameters.
        tf: datetime or None
            Ending date for estimating parameters.
        rank: int, optional
            MPI rank. Default: 1.
        """

        if isinstance(collection, timefn.TimefnCollection):
            self.collection = collection
        else:
            self.collection = timefn.TimefnCollection()
            self.collection.append(timefn.fnmap['poly'](t[0], order=1, units='years'))

        # Evaluate the collection
        self.G = self.collection(t)

        # Get indices for the functional partitions
        fnParts = timefn.getFunctionTypes(self.collection)
        for key in ('secular', 'seasonal', 'transient', 'step'):
            setattr(self, 'i%s' % key, fnParts[key])
        self.npar = self.G.shape[1]
        self.ifull = np.arange(self.npar, dtype=int)
        self._updatePartitionSizes()

        # Make mask for time window for estimating parameters
        t0 = datetime.strptime(t0, '%Y-%m-%d') if t0 is not None else t[0]
        tf = datetime.strptime(tf, '%Y-%m-%d') if tf is not None else t[-1]
        self.time_mask = (t >= t0) * (t <= tf)

        # Save the regularization indices
        self.reg_indices = fnParts['reg']
    
        # Save MPI rank
        self.rank = rank

        return


    def _updatePartitionSizes(self):
        """
        Update the sizes of the list of indices for the functional partitions.
        """
        for attr in ('secular', 'seasonal', 'transient', 'step', 'full'):
            ind_list = getattr(self, 'i%s' % attr)
            setattr(self, 'n%s' % attr, len(ind_list))
        return


    def invert(self, solver, d, wgt=None):
        """
        Perform least squares inversion using a solver.
        """

        # Do the inversion
        ind = np.isfinite(d) * self.time_mask
        if wgt is None:
            m, Cm = solver.invert(self.G[ind,:], d[ind])
        else:
            m, Cm = solver.invert(self.G[ind,:], d[ind], wgt=wgt[ind])

        # Save the partitions
        self.Cm = Cm
        self.coeff = {'secular': m[self.isecular], 'seasonal': m[self.iseasonal],
            'transient': m[self.itransient], 'step': m[self.istep]}
        return m


    def initializeOutput(self):
        """
        Makes an empty dictionary for holding partitional results.
        """
        N = self.G.shape[0]
        zero = np.zeros((N,))
        self.fit_dict = {}
        for attr in ('step', 'seasonal', 'secular', 'transient', 'full', 'sigma'):
            self.fit_dict[attr] = zero.copy()
        return


    def updateOutput(self, new_dict):
        """
        Update functional fits using a given dictionary.
        """
        for key,arr in new_dict.items():
            self.fit_dict[key] += arr
        return


    def computeSeasonalAmpPhase(self):
        """
        Try to compute the seasonal amplitude and phase.
        """
        try:
            m1, m2 = self.coeff['seasonal'][-2:]
            phs = np.arctan2(m2, m1) * 182.5/np.pi
            amp = np.sqrt(m1**2 + m2**2)
            if phs < 0.0:
                phs += 365.0
        except ValueError:
            phs, amp = None, None
        return amp, phs

    
    def getSecular(self, mvec):
        """
        Return the polynomial component with the highest power.
        """
        msec = mvec[self.isecular]
        variance_secular = np.diag(self.Cm)[self.isecular]
        if len(msec) != 2:
            return 0.0, 0.0
        else:
            return msec[-1], np.sqrt(variance_secular[-1])


    def getStep(self, mvec):
        """
        Return step coefficients.
        """
        mstep = mvec[self.istep]
        variance_step = np.diag(self.Cm)[self.isecular]
        if len(msec) < 1:
            return 0.0, 0.0
        else:
            return mstep[-1], variance_step[-1]


    def predict(self, mvec, out=None, sigma=True):
        """
        Predict time series with a functional decomposition specified by data.

        Parameters
        ----------
        mvec: np.ndarray
            Array of parameters.
        """

        # Compute different components
        secular = np.dot(self.G[:,self.isecular], mvec[self.isecular])
        seasonal = np.dot(self.G[:,self.iseasonal], mvec[self.iseasonal])
        transient = np.dot(self.G[:,self.itransient], mvec[self.itransient])
        step = np.dot(self.G[:,self.istep], mvec[self.istep])

        # Compute the functional partitions
        results = {'secular': secular, 'seasonal': seasonal, 'transient': transient,
                   'step': step, 'full': secular + seasonal + transient + step}

        # Add uncertainty if applicable
        if hasattr(self, 'Cm') and sigma:
            sigma = np.sqrt(np.diag(np.dot(self.G, np.dot(self.Cm, self.G.T))))
            results['sigma'] = sigma

        return results


    def detrend(self, d, recon, parts_to_remove):
        """
        Detrend the data and update model fit.
        """
        # Determine which functional partition to keep
        parts_to_keep = np.setdiff1d(['seasonal', 'secular', 'transient', 'step'], 
            parts_to_remove)

        # Compute signal to remove
        signal_remove = np.zeros_like(d)
        for key in parts_to_remove:
            signal_remove += recon[key]

        # And signal to keep
        signal_keep = np.zeros_like(d)
        for key in parts_to_keep:
            signal_keep += recon[key]
        
        # Detrend
        d -= signal_remove
        return signal_keep


def predict(stack_list, time_index, name='recon', islice=None, jslice=None):
    """
    Predict a time snapshot velocity field.
    """

    # Make sure input stacks are in a list
    if not isinstance(stack_list, (list, tuple)):
        stack_list = [stack_list]

    # Construct row slices
    Ny, Nx = stack_list[0].Ny, stack_list[0].Nx
    if islice is None:
        islice = slice(0, Ny)
    
    # Construct column slices
    if jslice is None:
        jslice = slice(0, Nx)

    # Compute reconstruction
    fit = 0.0
    for stack in stack_list:
        fit += stack[name][time_index, islice, jslice]

    return fit


def build_temporal_model(t, userfile, cov=False):

    from importlib.machinery import SourceFileLoader

    # If decimal times passed in, convert to dates
    if isinstance(t[0], (float, np.float32)):
        dates = tdec2datestr(t, returndate=True)
    elif isinstance(t[0], (datetime.datetime, datetime.date)):
        dates = t
    else:
        raise ValueError('Incompatible array of times.')
        
    # Load collection
    collfun = SourceFileLoader('build', userfile).load_module()
    collection = collfun.build(dates)

    # Create a model for handling the time function
    model = Model(dates, collection=collection)

    # Also try to build a prior covariance matrix
    if cov:
        collfun = SourceFileLoader('computeCm', userfile).load_module()
        Cm = collfun.computeCm(collection)
        return model, Cm
    else:
        return model


def build_temporal_design_matrix(t, userfile):

    # Create a model for handling the time function
    model = build_temporal_model(dates, userfile)
    return model.G


def build_seasonal_matrix(t):

    from . import timefn

    # If decimal times passed in, convert to dates
    if isinstance(t[0], (float, np.float32)):
        dates = tdec2datestr(t, returndate=True)
    elif isinstance(t[0], (datetime.datetime, datetime.date)):
        dates = t
    else:
        raise ValueError('Incompatible array of times.')

    # Get time bounds
    tstart, tend = dates[0], dates[-1]
    tdec_start = datestr2tdec(pydtime=tstart)
    tdec_end = datestr2tdec(pydtime=tend)

    # Initalize a collection and relevant basis functions
    collection = timefn.TimefnCollection()
    periodic = timefn.fnmap['periodic']
    poly = timefn.fnmap['poly']

    # Polynomial first
    collection.append(poly(tref=tstart, order=0, units='years'))

    # Seasonal terms
    collection.append(periodic(tref=tstart, units='years', period=0.5,
        tmin=tstart, tmax=tend))
    collection.append(periodic(tref=tstart, units='years', period=1.0,
        tmin=tstart, tmax=tend))

    # Evaluate collection
    G = collection(dates)

    return G


def get_model_component_indices(collection):

    from . import timefn

    fnParts = timefn.getFunctionTypes(collection)
    index_dict = {}
    for key in ('seasonal', 'transient' ,'secular'):
        index_dict[key] = len(fnParts[key])

    return index_dict

# end of file
