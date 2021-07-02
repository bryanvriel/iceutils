#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import iceutils as ice
import argparse
import sys

line = None

def parse():
    parser = argparse.ArgumentParser(description="""
        View a raster.""")
    parser.add_argument('rasterfile', type=str,
        help='Input raster to display.')
    parser.add_argument('-cmap', action='store', type=str, default='turbo',
        help='Matplotlib cmap to use for displaying raster. Default: turbo.')
    parser.add_argument('-clim', action='store', type=float, nargs=2, default=[None, None],
        help='Color limits for display.')
    parser.add_argument('-xscale', action='store', type=float, default=1.0,
        help='Scale factor for raster coordinates. Default: 1.0.')
    return parser.parse_args()

def main(args):

    # Load the raster
    r = ice.Raster(args.rasterfile)

    fig, ax = plt.subplots()
    vmin, vmax = args.clim
    cmap = ice.get_cmap(args.cmap)
    im = ax.imshow(r.data, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap,
                   extent=args.xscale*r.hdr.extent)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = parse()
    main(args)

# end of file
