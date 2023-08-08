#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import iceutils as ice
import argparse
import sys

line = None

def parse():
    parser = argparse.ArgumentParser(description="""
        Create a kml/kmz of a raster.""")
    parser.add_argument('rasterfile', type=str,
        help='Input raster to render.')
    parser.add_argument('-b', action='store', type=int, default=1, dest='band',
        help='Raster band. Default: 1.')
    parser.add_argument('--dims', action='store', type=int, nargs=2, default=[None, None],
        help='Dimensions for raster warped to EPSG:4326. Default preserves dimensions.')
    parser.add_argument('-cmap', action='store', type=str, default='turbo',
        help='Matplotlib cmap to use for displaying raster. Default: turbo.')
    parser.add_argument('-clim', action='store', type=float, nargs=2, default=[None, None],
        help='Color limits for display.')
    parser.add_argument('--dpi', action='store', type=int, default=300,
        help='DPI for png image. Default: 300.')
    parser.add_argument('--colorbar', action='store_true',
        help='Add colorbar to image.')
    parser.add_argument('-o', action='store', type=str, default=None, dest='output',
        help='Output name for kml/kmz file.')
    return parser.parse_args()

def main(args):

    # Load the raster
    r = ice.Raster(args.rasterfile, band=args.band)

    # Do warping if necessary
    if r.hdr.epsg != 4326:
        print('warping')
        if args.dims[0] is None:
            target_dims = None
        else:
            target_dims = tuple(args.dims)
        r = ice.warp(r, target_epsg=4326, order=1, target_dims=target_dims)

    if args.output is not None:
        filename = args.output
    else:
        ext = args.rasterfile.split('.')[-1]
        filename = args.rasterfile.replace(ext, 'kmz')

    # Perform warping if necessary
    vmin, vmax = args.clim
    ice.render_kml(r, filename, dpi=args.dpi, cmap=args.cmap, clim=(vmin, vmax),
                   colorbar=args.colorbar)

if __name__ == '__main__':
    args = parse()
    main(args)

# end of file
