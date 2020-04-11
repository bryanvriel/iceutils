#-*- coding: utf-8 -*-

def AGlen_vs_temp(T, KPa=False):
    """
    Copied from Hilmar's Ua code, but can return in units of {a^-1} {Pa^-3}

    Parameters
    ----------
    T: float
        Temperature in Celsus.
    KPa: bool, optional
        Return in units of {a^-1} {KPa^-3}. Default: False.

    Returns
    -------
    AGlen: float
        Glen's Flow Law parameter.
    """
    import numpy

    T = T + 273.15
    a0 = 5.3e-15 * 365.25 * 24. * 60. * 60. # a-1 kPa-3
    fa = 1.2478766e-39 * numpy.exp(0.32769 * T) + 1.9463011e-10 * numpy.exp(0.07205 * T)
    AGlen = a0 * fa

    if KPa:
        return AGlen
    else:
        return AGlen * 1.0e-9

# end of file
