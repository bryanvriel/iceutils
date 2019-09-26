#!/usr/bin/env python3

import numpy as np
import argparse
import iceutils as ice
import sys

def parse():
    parser = argparse.ArgumentParser(description="""
        Crop stack spatially and temporally.""")
    parser.add_argument('stackfile', metavar='Stackfile', type=str,
        help='Input stack file to subset from.')
    parser.add_argument('output', type=str,
        help='Output cropped stack file.')
    parser.add_argument('-projWin', action='store', type=float, nargs=4, default=None,
        help='Subwindow based on geographic coordinates.')
    parser.add_argument('-srcWin', action='store', type=int, nargs=4, default=None,
        help='Subwindow based on image coordinates and window size.')
    parser.add_argument('-chunks', action='store', type=int, nargs=2, default=[128, 128],
        help='Output HDF5 chunk size. Default: 128 128')
    parser.add_argument('-tWin', action='store', type=float, nargs=2, default=None,
        help='Temporal subwindow in decimal years.')
    return parser.parse_args()

def main(args):

    # Check if nothing is passed
    if args.projWin is None and args.srcWin is None and args.tWin is None:
        print('No cropping parameters provided.')
        return

    # Load stack
    stack = ice.Stack(args.stackfile)

    # Construct spatial slices
    if args.projWin is not None:

        # Unpack projWin parameters
        x0, y0, x1, y1 = args.projWin

        # Convert geographic coordinates to image coordinates
        i0, j0 = stack.hdr.xy_to_imagecoord(x0, y0)
        i1, j1 = stack.hdr.xy_to_imagecoord(x1, y1)
        islice = slice(i0, i1)
        jslice = slice(j0, j1)

    elif args.srcWin is not None:

        # Unpack srcWin parameters
        j0, i0, xsize, ysize = args.srcWin
        j1 = j0 + xsize
        i1 = i0 + ysize
        islice = slice(i0, i1)
        jslice = slice(j0, j1)

    else:
        islice = slice(0, stack.Ny)
        jslice = slice(0, stack.Nx)

    # Construct temporal subset if provided
    tdec = stack.tdec
    if args.tWin is not None:
        t0, tf = args.tWin
        k0 = np.argmin(np.abs(tdec - t0))
        k1 = np.argmin(np.abs(tdec - tf))
        tslice = slice(k0, k1)
        tdec = tdec[tslice]
    else:
        tslice = slice(0, stack.Nt)

    # Make meshgrid of coordinates
    X, Y = stack.hdr.meshgrid()
    # Apply slices
    if islice is not None and jslice is not None:
        X = X[islice, jslice]
        Y = Y[islice, jslice]

    # Create RasterInfo header
    hdr = ice.RasterInfo(X=X, Y=Y)

    # Create output stack
    ostack = ice.Stack(args.output, mode='w')
    ostack.initialize(tdec, hdr, data=True, weights=True,
                      chunks=(1, args.chunks[0], args.chunks[1]))

    # Manually fill in the data
    try:
        ostack['data'][:, :, :] = stack['data'][tslice, islice, jslice]
    except KeyError:
        ostack['data'][:, :, :] = stack['igram'][tslice, islice, jslice]
    ostack['weights'][:, :, :] = stack['weights'][tslice, islice, jslice]

    
if __name__ == '__main__':
    # Parse command line arguments
    args = parse()
    # Run main
    main(args)

# end of file
