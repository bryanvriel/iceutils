#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import cm
import iceutils as ice
import argparse
import sys

line = None

def parse():
    parser = argparse.ArgumentParser(description="""
        Draw points on a map""")
    parser.add_argument('rasterfile', type=str,
        help='Input raster file to display.')
    parser.add_argument('output', type=str,
        help='Output points file.')
    parser.add_argument('-clim', action='store', type=float, nargs=2, default=[None, None],
        help='Color limits for display.')
    return parser.parse_args()

def main(args):

    # Load the raster
    raster = ice.Raster(rasterfile=args.rasterfile)

    path = '/Users/briel/data/jakobshavn/gimp_imagery/S1_2016-02-01_2016-02-12_northwest_v02.dat'
    sar = ice.Raster(rasterfile=path)
    db = 10.0 * np.log10(sar.data)
    low = np.percentile(db.ravel(), 5)
    high = np.percentile(db.ravel(), 99.9)

    fig, ax = plt.subplots()
    vmin, vmax = args.clim
    ref = ax.imshow(db, aspect='auto', cmap='gray', vmin=low, vmax=high, extent=raster.hdr.extent)
    im = ax.imshow(raster.data, aspect='auto', vmin=vmin, vmax=vmax, cmap=cm.GMT_haxby,
                   extent=raster.hdr.extent, alpha=0.2)

    # Define action for clicking on deformation map
    xpts = []; ypts = []
    def printcoords(event):

        global line

        if event.inaxes != ax:
            return

        # Get cursor coordinates if left-click
        if event.button == 1:
            y, x = event.ydata, event.xdata
            xpts.append(x)
            ypts.append(y)

            if line is None:
                line, = ax.plot(xpts, ypts, '-o')
            else:
                line.set_xdata(xpts)
                line.set_ydata(ypts)

        # Right-click clear the points
        elif event.button == 3:
            xpts.clear()
            ypts.clear()
            line.set_xdata(xpts)
            line.set_ydata(ypts)
           
        fig.canvas.draw() 

    cid = fig.canvas.mpl_connect('button_press_event', printcoords)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    # Save the points
    np.savetxt(args.output, np.column_stack((xpts, ypts)))


if __name__ == '__main__':
    args = parse()
    main(args)

# end of file
