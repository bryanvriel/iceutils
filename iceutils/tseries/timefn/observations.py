"""Time normalization shared by temporal observation interfaces."""

import datetime
import numpy as np
from pint.errors import DimensionalityError, UndefinedUnitError

from ...timeutils import tdec2datestr
from .CenteredBasisFn import ureg


def _normalize_times(t):
    """Return a nonempty vector of naive Python datetimes."""
    values = np.atleast_1d(np.asarray(t))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Times must be a nonempty one-dimensional array.")

    dates = []
    for value in values:
        if isinstance(value, np.datetime64):
            if np.isnat(value):
                raise ValueError("Times must be finite; NaT is not supported.")
            value = value.astype("datetime64[us]").astype(datetime.datetime)
        elif isinstance(value, (float, int, np.floating, np.integer)):
            if not np.isfinite(value):
                raise ValueError("Times must be finite.")
            value = tdec2datestr(float(value), returndate=True)

        if isinstance(value, datetime.datetime):
            if value.utcoffset() is not None:
                raise ValueError("Timezone-aware times are not supported.")
        elif isinstance(value, datetime.date):
            value = datetime.datetime.combine(value, datetime.time())
        else:
            raise ValueError("Times must be dates, datetimes, or decimal years.")
        dates.append(value)

    return np.array(dates, dtype=object)


def _seconds_per_unit(time_units):
    """Convert a positive, fixed elapsed-time unit to seconds."""
    try:
        seconds = float(ureg.Quantity(1, time_units).to("seconds").magnitude)
    except (DimensionalityError, UndefinedUnitError, TypeError, ValueError) as error:
        raise ValueError("time_units must specify an elapsed-time unit.") from error
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("time_units must specify a positive finite time unit.")
    return seconds


def _normalize_observations(t, observation, t_end, time_units):
    """Validate observation geometry and return dates, endpoints, and durations."""
    if observation not in ("sample", "integral", "average", "difference"):
        raise ValueError("Unknown observation kind: %s" % observation)
    dates = _normalize_times(t)
    seconds = _seconds_per_unit(time_units)
    if observation == "sample":
        if t_end is not None:
            raise ValueError("Sample observations do not accept t_end.")
        return dates, None, None

    if t_end is None:
        raise ValueError("Interval observations require t_end.")
    ends = _normalize_times(t_end)
    if dates.shape != ends.shape:
        raise ValueError("t and t_end must have matching shapes; no broadcasting.")
    durations = (
        np.array([(end - start).total_seconds() for start, end in zip(dates, ends)])
        / seconds
    )
    if np.any(durations <= 0):
        raise ValueError("Interval observations require strictly positive durations.")
    return dates, ends, durations
