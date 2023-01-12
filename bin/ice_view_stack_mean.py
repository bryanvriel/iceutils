#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import iceutils as ice
import argparse
import sys

line = None

def parse():
    parser = argparse.ArgumentParser(description="""
        Draw points on a map""")
    parser.add_argument('stackfile', type=str,
        help='Input stack file to display.')
    parser.add_argument('-key', action='store', type=str, default='data',
        help='Dataset to compute mean of. Default: data.')
    parser.add_argument('-ref', action='store', type=str, default=None,
        help='Reference SAR raster for background. Default: None.')
    parser.add_argument('-alpha', action='store', type=float, default=1.0,
        help='Image alpha for displaying raster. Default: 1.0.')
    parser.add_argument('-cmap', action='store', type=str, default='turbo',
        help='Matplotlib cmap to use for displaying raster. Default: turbo.')
    parser.add_argument('-clim', action='store', type=float, nargs=2, default=[None, None],
        help='Color limits for display.')
    parser.add_argument('-xscale', action='store', type=float, default=1.0,
        help='Scale factor for raster coordinates. Default: 1.0.')
    parser.add_argument('-save', action='store', type=str, default=None,
        help='Save mean image to raster. Default: None.')
    parser.add_argument('-save_epsg', action='store', type=int, default=None,
        help='Override EPSG for saving mean raster. Default: None.')
    return parser.parse_args()

def main(args):

    # Load the stack
    stack = ice.Stack(args.stackfile)

    # Check if requested dataset is 2D. If so, view it directly
    if stack[args.key].ndim == 2:
        mean = stack[args.key][()]
    # Otherwise, compute mean
    else:
        mean = stack.mean(key=args.key)

    # Load reference SAR image
    if args.ref is not None:
        sar = ice.Raster(rasterfile=args.ref)
        if sar.hdr != stack.hdr:
            sar.resample(stack.hdr)
        db = 10.0 * np.log10(sar.data)
        low = np.percentile(db.ravel(), 5)
        high = np.percentile(db.ravel(), 99.9)
    else:
        db = None

    fig, ax = plt.subplots()
    vmin, vmax = args.clim
    cmap = ice.get_cmap(args.cmap)
    if db is not None:
        ref = ax.imshow(db, aspect='auto', cmap='gray', vmin=low, vmax=high,
                        extent=stack.hdr.extent)
    im = ax.imshow(mean, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap,
                   extent=args.xscale*stack.hdr.extent, alpha=args.alpha)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(args.key)

    plt.show()

    if args.save is not None:
        out = ice.Raster(data=mean, hdr=stack.hdr)
        out.write_gdal(args.save, epsg=args.save_epsg)


if __name__ == '__main__':
    args = parse()
    main(args)

# end of file
