#-*- coding: utf-8 -*-

import datetime
import numpy as np
import copy
import math
import sys

def mjd2ymd(mjd):
    """
    Convert modified julian date to a tuple of (year,month,day). Adapted from CGS network
    Javascript.
    """

    # Julian day
    jd = mjd + 2400000.5
    jdi = jd.astype(int)

    # Fractional part of day
    jdf = jd - jdi + 0.5

    # Check if the next calendar day
    ind = jdf >= 1.0
    jdf[ind] -= 1.0
    jdi[ind] += 1

    l = jdi + 68569.0
    n = np.floor(4.0 * l / 146097.0)
    l = np.floor(l) - np.floor((146097.0 * n + 3.0) / 4.0)
    year = np.floor(4000.0 * (l + 1.0) / 1461001.0)

    l = l - np.floor(1461.0 * 0.25 * year) + 31.0
    month = np.floor(80.0 * l / 2447.0)

    day = l - np.floor(2447.0 * month / 80.0) + jdf

    l = np.floor(month / 11.0)

    month = np.floor(month + 2.0 - 12.0 * l)
    year = np.floor(100.0 * (n - 49.0) + year + l)

    return year, month, day


def mjd2doy(mjd):
    """
    Convert modified julian date to a decimal year.
    """

    # Julian day
    jd = mjd + 2400000.5
    jdi = jd.astype(int)

    # Fractional part of day
    jdf = jd - jdi + 0.5

    # Check if the next calendar day
    ind = jdf >= 1.0
    jdf[ind] -= 1.0
    jdi[ind] += 1

    # Compute the year
    l = jdi + 68569.0
    n = np.floor(4.0 * l / 146097.0)
    l = np.floor(l) - np.floor((146097.0 * n + 3.0) / 4.0)
    year = np.floor(4000.0 * (l + 1.0) / 1461001.0)

    # Compute the decimal day
    day = l - np.floor(1461.0 * 0.25 * year) + 31.0

    
    return year, day


def datestr2tdec(yy=0, mm=0, dd=0, hour=0, minute=0, sec=0, microsec=0,
                 datestr=None, pydtime=None):
    """
    Convert year, month, day, hours, minutes, seconds to decimal year.
    """
    if datestr is not None:
        yy, mm, dd = [int(val) for val in datestr.split('-')]
        hour, minute, sec = [0, 0, 0]

    if pydtime is not None:
        attrs = ['year', 'month', 'day', 'hour', 'minute', 'second']
        yy, mm, dd, hour, minute, sec = [getattr(pydtime, attr) for attr in attrs]


    # Make datetime object for start of year
    yearStart = datetime.datetime(yy, 1, 1, 0, 0, 0)
    # Make datetime object for input time
    current = datetime.datetime(yy, mm, dd, hour, minute, sec, microsec)
    # Compute number of days elapsed since start of year
    tdelta = current - yearStart
    # Convert to decimal year and account for leap year
    if yy % 4 == 0:
        return float(yy) + tdelta.total_seconds() / (366.0 * 86400)
    else:
        return float(yy) + tdelta.total_seconds() / (365.0 * 86400)


def tdec2datestr(tdec_in, returndate=False):
    """
    Convert a decimaly year to an iso date string.
    """
    if isinstance(tdec_in, (list, np.ndarray)):
        tdec_list = copy.deepcopy(tdec_in)
    else:
        tdec_list = [tdec_in]
    current_list = []
    for tdec in tdec_list:
        year = int(tdec)
        yearStart = datetime.datetime(year, 1, 1)
        if year % 4 == 0:
            ndays_in_year = 366.0
        else:
            ndays_in_year = 365.0
        days = (tdec - year) * ndays_in_year
        seconds = (days - int(days)) * 86400
        tdelta = datetime.timedelta(days=int(days), seconds=int(seconds))
        current = yearStart + tdelta
        if not returndate:
            current = current.isoformat(' ').split()[0]
        current_list.append(current)

    if len(current_list) == 1:
        return current_list[0]
    else:
        return np.array(current_list)


def generateRegularTimeArray(tmin, tmax):
    """
    Make a regularly spaced time array with the correct days per year.
    """
    year_start = int(tmin)
    year_end = int(tmax)
    first = True
    for year in range(year_start, year_end+1):
        # Check for leap year
        if year % 4 == 0:
            ndays_in_year = 366.0
        else:
            ndays_in_year = 365.0
        # Get day array
        if year == year_start:
            ndays = round((tmin - float(year_start)) * ndays_in_year, 0)
            days = np.arange(ndays, ndays_in_year, dtype=float) / ndays_in_year
        elif year == year_end:
            ndays = round((tmax - float(year_end)) * ndays_in_year, 0)
            days = np.arange(ndays+1, dtype=float) / ndays_in_year
        else:
            days = np.arange(ndays_in_year, dtype=float) / ndays_in_year
        # Add array
        if first:
            tdec = float(year) + days
            first = False
        else:
            tdec = np.hstack((tdec, float(year)+days))

    return tdec


def generateDateList(start, end, delta):
    """
    Makes a list of datetime objects specified by start, end, and timedelta.
    """
    curr = start
    result = [start]
    while curr < end:
        curr += delta
        result.append(curr)

    return result


# end of file 
