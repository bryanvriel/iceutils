#-*- coding: utf-8 -*-

import numpy as np
import datetime
import h5py

from ..constants import FAIL
from ..timeutils import tdec2datestr, datestr2tdec
from . import timefn
from .timefn.observations import _normalize_times, _normalize_observations
from .LinearRegression import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    OrthogonalMatchingPursuit,
)


def _check_identifiability(solver, G, wgt):
    """Reject null directions left unconstrained by a built-in solver."""
    if type(solver) not in (
        LinearRegression,
        RidgeRegression,
        LassoRegression,
        OrthogonalMatchingPursuit,
    ):
        return
    if len(G) < solver.n_min:
        return

    # Build rows with the same null space as the weighted data and penalties.
    rows = G if wgt is None else G * wgt[:, None]
    npar = G.shape[1]
    regMat = getattr(solver, "regMat", None)
    if regMat is not None:
        regMat = np.asarray(regMat, dtype=float)
        if (
            regMat.shape != (npar, npar)
            or not np.all(np.isfinite(regMat))
            or not np.allclose(regMat, regMat.T)
        ):
            raise ValueError(
                "regMat must be a finite symmetric parameter precision matrix."
            )
        values, vectors = np.linalg.eigh(regMat)
        tolerance = npar * np.finfo(float).eps * np.max(np.abs(values))
        if np.any(values < -tolerance):
            raise ValueError("regMat must be positive semidefinite.")
        positive = values > tolerance
        penalty_rows = np.sqrt(values[positive])[:, None] * vectors[:, positive].T
        rows = np.vstack((rows, penalty_rows))

    # Sparse penalties constrain null directions without identifying a unique
    # allocation among overlapping penalized columns. OMP selects all columns.
    if type(solver) is OrthogonalMatchingPursuit:
        return
    if type(solver) is LassoRegression or (
        type(solver) is RidgeRegression and regMat is None
    ):
        indices = np.asarray(solver.reg_indices, dtype=int)
        penalty = np.broadcast_to(solver.penalty, indices.shape)
        if np.any(~np.isfinite(penalty)) or np.any(penalty < 0):
            raise ValueError("Penalties must be finite and nonnegative.")
        penalty_rows = np.eye(npar)[indices] * np.sqrt(penalty)[:, None]
        rows = np.vstack((rows, penalty_rows))

    # Normalize columns so rank assessment is not set by coefficient units.
    norms = np.linalg.norm(rows, axis=0)
    scaled = rows / np.where(norms > 0, norms, 1)
    if np.linalg.matrix_rank(scaled) < npar:
        raise ValueError(
            "Observation matrix and penalties leave unconstrained null directions. "
            "Omit unobservable/duplicate terms or provide a suitable penalty. "
            "For displacement differences, omit the constant with min_poly=1."
        )

class Model:
    """
    Class for handling time series predictions.
    """

    def __init__(
        self,
        t,
        collection=None,
        t0=None,
        tf=None,
        rank=0,
        *,
        observation="sample",
        t_end=None,
        time_units="years"
    ):
        """
        Initialize the Model class with a TimeRepresentation object.

        Parameters
        ----------
        t: array_like
            Sample epochs or interval starts: naive dates/datetimes, NumPy
            datetimes, or decimal years. Scalars produce one observation.
        collection: {giant.utilities.timefn.TimefnCollection, None}, optional
            GIAnT TimefnCollection instance. If None, constructs a polynomial collection.
        t0: datetime or None
            Starting date for estimating parameters.
        tf: datetime or None
            Ending date for estimating parameters.
        rank: int, optional
            MPI rank. Default: 0.
        observation: {'sample', 'integral', 'average', 'difference'}, optional
            Observation operator on the modeled quantity. Default: 'sample'.
        t_end: array_like, optional
            Interval ends matching t, required for non-sample observations.
        time_units: str, optional
            Elapsed-time units of integrals and durations. Default: 'years'
            (365.25 days). Basis units and coefficient ordering remain fixed.
        """

        t, t_end, durations = _normalize_observations(t, observation, t_end, time_units)

        if isinstance(collection, timefn.TimefnCollection):
            self.collection = collection
        else:
            self.collection = timefn.TimefnCollection()
            tref = t[0] if observation == "sample" else min(t)
            self.collection.append(
                timefn.fnmap["poly"](tref=tref, order=1, units="years")
            )

        # Save the time array
        self.t = t
        self.observation = observation
        self.t_end = t_end
        self.durations = durations
        self.time_units = time_units
        self.tdec = np.array([datestr2tdec(dateobj=date) for date in t])

        # Evaluate the collection
        self.G = self.collection.design_matrix(t, observation, t_end, time_units)

        # Get indices for the functional partitions
        fnParts = timefn.getFunctionTypes(self.collection)
        for key in ('secular', 'seasonal', 'transient', 'step'):
            setattr(self, 'i%s' % key, fnParts[key])
        self.npar = self.G.shape[1]
        self.ifull = np.arange(self.npar, dtype=int)
        self._updatePartitionSizes()

        # Make mask for time window for estimating parameters
        if isinstance(t0, str):
            t0 = datetime.datetime.fromisoformat(t0)
        if isinstance(tf, str):
            tf = datetime.datetime.fromisoformat(tf)
        default_start = t[0] if observation == "sample" else min(t)
        default_end = t[-1] if observation == "sample" else max(t_end)
        self.t0 = default_start if t0 is None else _normalize_times(t0)[0]
        self.tf = default_end if tf is None else _normalize_times(tf)[0]
        window_end = t if observation == "sample" else t_end
        self.time_mask = (t >= self.t0) & (window_end <= self.tf)

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
        Perform least squares inversion using a solver. Raises RuntimeError if the
        solver could not estimate a coefficient vector.
        """

        d = np.asarray(d)
        if d.shape != (self.G.shape[0],):
            raise ValueError("d must have one value per observation.")
        if wgt is not None:
            wgt = np.asarray(wgt)
            if wgt.shape != d.shape:
                raise ValueError(
                    "wgt must have one inverse standard deviation per observation."
                )

        # Do the inversion over the finite data within the estimation window.
        ind = np.isfinite(d) * self.time_mask
        if self.observation != "sample":
            if wgt is not None:
                if np.any(wgt[ind] < 0):
                    raise ValueError("Inverse standard deviations must be nonnegative.")
                ind &= np.isfinite(wgt) & (wgt > 0)
            weights = None if wgt is None else wgt[ind]
            _check_identifiability(solver, self.G[ind, :], weights)
        if wgt is None:
            status, m, Cm = solver.invert(self.G[ind,:], d[ind])
        else:
            status, m, Cm = solver.invert(self.G[ind,:], d[ind], wgt=wgt[ind])

        # A failed solver returns no coefficients, so stop before unpacking them
        if status == FAIL:
            raise RuntimeError('Time series inversion failed.')

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


    def design_matrix(self, t=None, *, observation=None, t_end=None, time_units=None):
        """Evaluate this model's fixed basis on new observations.

        Parameters
        ----------
        t : array_like, optional
            New epochs or interval starts. With no arguments, return self.G.
        observation : {'sample', 'integral', 'average', 'difference'}, optional
            Defaults to this model's operator. Pass 'sample' explicitly to
            predict instantaneous values from an interval-fitted model.
        t_end : array_like, optional
            Matching interval ends. New interval starts require new ends.
        time_units : str, optional
            Integration units, defaulting to this model's time_units.

        Returns
        -------
        G : (N, P) numpy.ndarray
            Design matrix with the original basis and coefficient ordering.
        """
        if t is None and observation is None and t_end is None and time_units is None:
            return self.G
        if observation is None:
            observation = self.observation
        if time_units is None:
            time_units = self.time_units
        if t is None:
            t = self.t
            if t_end is None and observation != "sample":
                t_end = self.t_end
        return self.collection.design_matrix(t, observation, t_end, time_units)

    def metadata(self):
        """Return a JSON-compatible snapshot of basis and observation definitions.

        Returns
        -------
        metadata : dict
            ISO datetime endpoints, elapsed durations and units, fitting window,
            basis class names and actual parameters, and column partitions.
            Save the solver configuration and coefficients separately. Custom
            basis classes must provide their own serialization.
        """
        basis_definitions = []
        for basis in self.collection:
            if type(basis) not in timefn.fnmap.values():
                raise NotImplementedError(
                    "Metadata is not supported for custom basis classes."
                )
            parameters = {}
            for name in (
                "tmin",
                "tmax",
                "tref",
                "units",
                "norm",
                "tau",
                "exponent",
                "period",
                "scale",
                "order",
                "offset",
                "radfactor",
                "efactor",
            ):
                if hasattr(basis, name):
                    value = getattr(basis, name)
                    if isinstance(value, datetime.datetime):
                        value = value.isoformat()
                    elif isinstance(value, np.generic):
                        value = value.item()
                    parameters[name] = value
            basis_definitions.append({"type": basis.fnname, "parameters": parameters})

        partitions = {}
        for part in ("secular", "seasonal", "transient", "step"):
            partitions[part] = list(getattr(self, "i" + part))
        return {
            "observation": self.observation,
            "t": [date.isoformat() for date in self.t],
            "t_end": (
                None
                if self.t_end is None
                else [date.isoformat() for date in self.t_end]
            ),
            "durations": None if self.durations is None else self.durations.tolist(),
            "time_units": self.time_units,
            "t0": self.t0.isoformat(),
            "tf": self.tf.isoformat(),
            "basis": basis_definitions,
            "partitions": partitions,
            "reg_indices": list(self.reg_indices),
        }

    def predict(
        self,
        mvec,
        out=None,
        sigma=True,
        *,
        t=None,
        observation=None,
        t_end=None,
        time_units=None
    ):
        """
        Predict time series with a functional decomposition specified by data.

        Parameters
        ----------
        mvec: (P,) np.ndarray
            Coefficients of the original basis, independent of the operator.
        out: optional
            Retained for positional compatibility; unused.
        sigma: bool, optional
            Include propagated uncertainty when self.Cm exists. Default: True.
        t, observation, t_end, time_units: optional
            Observation arguments as in Model.design_matrix(). Defaults predict
            the stored fitting observations without rebuilding the basis.

        Returns
        -------
        results: dict
            Full and partitioned predictions, each with shape (N,), and sigma
            when requested and a coefficient covariance is available.
        """

        G = self.design_matrix(
            t, observation=observation, t_end=t_end, time_units=time_units
        )

        # Compute different components
        secular = np.dot(G[:, self.isecular], mvec[self.isecular])
        seasonal = np.dot(G[:, self.iseasonal], mvec[self.iseasonal])
        transient = np.dot(G[:, self.itransient], mvec[self.itransient])
        step = np.dot(G[:, self.istep], mvec[self.istep])

        # Compute the functional partitions
        results = {'secular': secular, 'seasonal': seasonal, 'transient': transient,
                   'step': step, 'full': secular + seasonal + transient + step}

        # Some legacy basis types have no named partition, but still contribute
        # to the modeled quantity and must be retained in the full prediction.
        partitioned = self.nsecular + self.nseasonal + self.ntransient + self.nstep
        if partitioned != self.npar:
            results["full"] = np.dot(G, mvec)

        # Add uncertainty if applicable
        if hasattr(self, 'Cm') and sigma:
            sigma = np.sqrt(np.diag(np.dot(G, np.dot(self.Cm, G.T))))
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
        fit += stack[name].isel(time=time_index, y=islice, x=jslice).values

    return fit


def build_temporal_model(
    t,
    poly=1,
    min_poly=0,
    periods=[0.5, 1.0],
    steps=[],
    isplines=[32, 16, 8, 4],
    bsplines=None,
    seasonal_bspline_sep=None,
    userfile=None,
    *,
    observation="sample",
    t_end=None,
    time_units="years"
):
    """
    Convenience function to build a temporal model from commonly-used pieces. Can alternatively
    build the model by reading in relevant code from an external file.

    Parameters
    ----------
    t: (N,) array
        Array of datetime or decimal years corresponding to time epochs.
    poly: int, optional
        Maximum order of polynomial to include. Default: 1.
    min_poly: int, optional
        Minimum order of polynomial to include. Default: 0.
    periods: list, optional
        Periods (in years) of sinusoidal components to include. Default: [0.5, 1.0].
    steps: list, optional
        Time epochs to include step functions. Default: [].
    isplines: list, optional
        I-splines to include. Default: [32, 16, 8, 4].
    bsplines: list, optional
        B-splines to include. Default: None.
    seasonal_bspline_sep: float, optional
        Specification of spacing (in years) for B-splines for repeating signals.
    userfile: str, optional
        External file to read model code from. Default: None.
    observation: {'sample', 'integral', 'average', 'difference'}, optional
        Temporal observation operator. Default: 'sample'. For intervals, basis
        definitions span all endpoints, independent of observation row order.
    t_end: array_like, optional
        Interval ends matching t. Required for non-sample observations.
    time_units: str, optional
        Integration and duration units. Default: 'years' (365.25 days).
        Periods and automatic basis definitions still use years.

    Returns
    -------
    model: ice.tseries.Model
        Temporal model.
    """
    from . import timefn

    # If external file provided, go ahead and call that function
    if userfile is not None:
        model = build_temporal_model_fromfile(
            t,
            userfile,
            cov=False,
            observation=observation,
            t_end=t_end,
            time_units=time_units,
        )
        return model

    dates, ends, durations = _normalize_observations(t, observation, t_end, time_units)

    # Get time bounds
    tstart, tend = dates[0], dates[-1]
    if ends is not None:
        tstart, tend = min(dates), max(ends)

    # Initalize a collection and relevant basis functions
    collection = timefn.TimefnCollection()
    periodic = timefn.fnmap['periodic']
    ispline = timefn.fnmap['isplineset']
    bspline = timefn.fnmap['bsplineset']
    single_bspl = timefn.fnmap['bspline']
    polyfn = timefn.fnmap['poly']
    stepfn = timefn.fnmap['constant']

    # Polynomial first
    min_poly = min(min_poly, poly)
    collection.append(polyfn(tref=tstart, order=poly, minorder=min_poly, units='years'))

    # Seasonal terms
    periods = periods or []
    for period in periods:
        collection.append(periodic(tref=tstart, units='years', period=period,
                                   tmin=tstart, tmax=tend))

    # Steps
    for step in steps:
        collection.append(stepfn(tmin=step))

    # B-splines
    bsplines = bsplines or []
    for nspl in bsplines:
        collection.append(bspline(order=3, num=nspl, units='years',
                                  tmin=tstart, tmax=tend))

    # Seasonal B-splines
    if seasonal_bspline_sep is not None:
        Δtdec = seasonal_bspline_sep
        Δt = datetime.timedelta(days=int(Δtdec*365))
        t_current = tstart
        while t_current <= tend:
            collection.append(single_bspl(order=3, scale=Δtdec, units='years', tref=t_current))
            t_current += Δt

    # Integrated B-splines
    isplines = isplines or []
    for nspl in isplines:
        collection.append(ispline(order=3, num=nspl, units='years',
                                  tmin=tstart, tmax=tend))

    # Build temporal model
    model = Model(
        dates,
        collection=collection,
        observation=observation,
        t_end=ends,
        time_units=time_units,
    )

    return model


def build_temporal_model_fromfile(
    t,
    userfile,
    cov=False,
    *,
    observation="sample",
    t_end=None,
    time_units="years",
    **kwargs
):
    """Build a fixed temporal basis from a user-supplied Python module.

    Parameters
    ----------
    t : array_like
        Sample epochs or interval starts, as dates/datetimes or decimal years.
    userfile : str
        Module defining build(dates, **kwargs) returning a TimefnCollection.
        For intervals, dates is the sorted unique union of all endpoints.
    cov : bool, optional
        Also return computeCm(collection, **kwargs). Default: False.
    observation : {'sample', 'integral', 'average', 'difference'}, optional
        Observation operator. Default: 'sample'.
    t_end : array_like, optional
        Matching interval ends; required for non-sample observations.
    time_units : str, optional
        Integration and duration units. Default: 'years' (365.25 days).
    **kwargs
        Passed to the userfile's build and computeCm functions.

    Returns
    -------
    model : Model
        Model with the userfile basis and requested observation matrix.
    Cm : numpy.ndarray, optional
        Prior coefficient covariance, returned only when cov is true.
    """

    from importlib.machinery import SourceFileLoader

    dates, ends, durations = _normalize_observations(t, observation, t_end, time_units)
    basis_dates = dates
    if ends is not None:
        basis_dates = np.unique(np.concatenate((dates, ends)))
        
    # Load collection
    collfun = SourceFileLoader('build', userfile).load_module()
    collection = collfun.build(basis_dates, **kwargs)

    # Create a model for handling the time function
    model = Model(
        dates,
        collection=collection,
        observation=observation,
        t_end=ends,
        time_units=time_units,
    )

    # Also try to build a prior covariance matrix
    if cov:
        Cm = collfun.computeCm(collection, **kwargs)
        return model, Cm
    else:
        return model


def build_temporal_design_matrix(t, userfile):

    # Create a model for handling the time function
    model = build_temporal_model(dates, userfile)
    return model.G


def get_model_component_indices(collection):

    from . import timefn

    fnParts = timefn.getFunctionTypes(collection)
    index_dict = {}
    for key in ('seasonal', 'transient' ,'secular'):
        index_dict[key] = len(fnParts[key])

    return index_dict

# end of file
