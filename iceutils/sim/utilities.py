#-*- coding: utf-8 -*-

import numpy as np

def load_profile_from_h5(h5file):
    """
    Creates a Profile object using data from an HDF5 output run
    """
    import h5py
    from .geometry import Profile
    with h5py.File(h5file, 'r') as fid:
        u, h, b, x = [fid[key][()] for key in ('u', 'h', 'b', 'x')]
        profile = Profile(x, h, b, u)
        if 't' in fid.keys():
            profile.t = fid['t'][()]
        else:
            profile.t = 0.0
    return profile

def save_profile_to_h5(profile, h5file, aux_data={}):
    """
    Saves a Profile object to HDF5.
    """
    import h5py
    with h5py.File(h5file, 'w') as fid:

        # Save standard profile data
        for key in ('u', 'h', 'b', 'x'):
            fid[key] = getattr(profile, key)
        if hasattr(profile, 't'):
            fid['t'] = profile.t

        # If any extra data has been provided, save those as Datasets
        for key, value in aux_data.items():
            fid[key] = value

    return

def AGlen_vs_temp(T, KPa=False):
    """
    Copied from Hilmar's Ua code, but cat return in units of {a^-1} {Pa^-3}
    """
    T = T + 273.15
    a0 = 5.3e-15 * 365.25 * 24. * 60. * 60. # a-1 kPa-3
    fa = 1.2478766e-39 * np.exp(0.32769 * T) + 1.9463011e-10 * np.exp(0.07205 * T)
    AGlen = a0 * fa

    if KPa:
        return AGlen
    else:
        return AGlen * 1.0e-9

# end of file
