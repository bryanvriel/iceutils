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
    parser.add_argument('-cmap', action='store', type=str, default='GMT_haxby',
        help='Matplotlib cmap to use for displaying raster. Default: GMT_haxby.')
    parser.add_argument('-clim', action='store', type=float, nargs=2, default=[None, None],
        help='Color limits for display.')
    parser.add_argument('-save', action='store', type=str, default=None,
        help='Save mean image to raster. Default: None.')
    return parser.parse_args()

def main(args):

    # Load the stack
    stack = ice.Stack(args.stackfile)

    # Compute mean
    mean = np.nanmean(stack[args.key][()], axis=0)

    # Load reference SAR image
    if args.ref is not None:
        sar = ice.Raster(rasterfile=args.ref)
        if sar.hdr != raster.hdr:
            sar.resample(raster.hdr)
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
                        extent=raster.hdr.extent)
    im = ax.imshow(mean, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap,
                   extent=stack.hdr.extent, alpha=args.alpha)

    plt.show()

    if args.save is not None:
        out = ice.Raster(data=mean, hdr=stack.hdr)
        out.write_gdal(args.save, epsg=3413)


if __name__ == '__main__':
    args = parse()
    main(args)

# end of file
