#!/usr/bin/env python3

# externals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider
from matplotlib.ticker import FormatStrFormatter
import argparse
import sys
import os

import iceutils as ice

plt.rc('font', size=12)

def parse():
    parser = argparse.ArgumentParser(description="""
        Explore time series tack""")
    parser.add_argument('stackfile', type=str,
        help='Input stack file to display.')
    parser.add_argument('-key', action='store', type=str, default='data',
        help='Dataset to view. Default: data.')
    parser.add_argument('-mfile', action='store', type=str, default=None,
        help='Model stack file. Default: None.')
    parser.add_argument('-mkey', action='store', type=str, default='data',
        help='Model dataset view. Default: data.')
    parser.add_argument('-mtdec', action='store', type=str, default='tdec',
        help='Model time vector. Default: tdec.')
    parser.add_argument('-ref', action='store', type=str, default=None,
        help='Reference SAR raster for background. Default: None.')
    parser.add_argument('-alpha', action='store', type=float, default=1.0,
        help='Image alpha for displaying raster. Default: 1.0.')
    parser.add_argument('-cmap', action='store', type=str, default='turbo',
        help='Matplotlib cmap to use for displaying raster. Default: turbo.')
    parser.add_argument('-clim', action='store', type=float, nargs=2, default=[None, None],
        help='Color limits for display.')
    parser.add_argument('-sigma', action='store_true',
        help='Plot errobars using weights dataset.')
    parser.add_argument('-frame', action='store', type=str, default='initial',
        help="""
            Type of stat to use for displaying map (mean, initial, final, std).
            Alternatively, may supply an integer index as index_{index} or key to 2D dataset.
            Default: initial.""")
    return parser.parse_args()

def main(args):

    # Load the stack
    stack = ice.Stack(args.stackfile)

    # Get frame
    if args.frame == 'initial':
        mean = stack.slice(0, key=args.key)
    elif args.frame == 'final':
        mean = stack.slice(stack.Nt - 1, key=args.key)
    elif args.frame == 'mean':
        mean = stack.mean(key=args.key)
    elif args.frame == 'std':
        mean = stack.std(key=args.key)
    elif args.frame.startswith('index_'):
        ind = int(args.frame.split('_')[1])
        mean = stack.slice(ind, key=args.key)
    else:
        try:
            mean = stack[args.frame][()]
        except KeyError:
            raise ValueError('Unsupported frame type.')

    # Check time array shape
    ds_shape = stack[args.key].shape
    if stack.fmt == 'NHW':
        Nt = ds_shape[0]
    elif stack.fmt == 'HWN':
        Nt = ds_shape[2]
    if Nt != stack.tdec.size:
        tdec = np.arange(Nt)
        tlabel = 'Index'
    else:
        tdec = stack.tdec
        tlabel = 'Year'
    
    # If model directory is given, load model stack (full fit)
    mstack = None
    if args.mfile is not None:
        mstack = ice.Stack(args.mfile)
        # Load correct time array
        if args.mtdec != 'tdec':
            mtdec = mstack[args.mtdec][()]
        else:
            mtdec = mstack.tdec

    # Load reference SAR image
    if args.ref is not None:
        sar = ice.Raster(rasterfile=args.ref)
        if sar.hdr != stack.hdr:
            sar.resample(stack.hdr)
        db = 10.0 * np.log10(sar.data.astype(np.float32))
        low = np.percentile(db.ravel(), 5)
        high = np.percentile(db.ravel(), 99.9)
    else:
        db = None

    # Initialize image plot
    fig, ax = plt.subplots()
    vmin, vmax = args.clim
    cmap = ice.get_cmap(args.cmap)
    if db is not None:
        ref = ax.imshow(db, aspect='auto', cmap='gray', vmin=low, vmax=high,
                        extent=stack.hdr.extent)
    im = ax.imshow(mean, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap,
                   extent=stack.hdr.extent, alpha=args.alpha)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(args.key)

    # Initialize plot for time series for a given pixel
    pts, axts = plt.subplots(figsize=(10,6))

    # Define action for clicking on deformation map
    def printcoords(event):

        if event.inaxes != ax:
            return

        # Get cursor coordinates
        y, x = event.ydata, event.xdata

        # Print out pixel locaton
        i, j = stack.hdr.xy_to_imagecoord(x, y)
        print('Row: %d Col: %d' % (i, j))

        # Get time series for cursor location
        d = stack.timeseries(xy=(x, y), key=args.key)
        
        # Plot data and fit
        axts.clear()
        if args.sigma:
            w = stack.timeseries(xy=(x, y), key='weights')
            sigma = 1.0 / w
            axts.errorbar(tdec, d, yerr=sigma, fmt='o')
        else:
            axts.plot(tdec, d, 'o', label=args.key)

        if mstack is not None:
            fit = mstack.timeseries(xy=(x, y), key=args.mkey)
            axts.plot(mtdec, fit, 'o', label='model ' + args.mkey)

        axts.set_xlabel(tlabel)
        axts.set_ylabel(args.key)
        axts.legend(loc='best')

        pts.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', printcoords)
    plt.show()
    fig.canvas.mpl_disconnect(cid)


if __name__ == '__main__':
    args = parse()
    main(args)

# end of file
