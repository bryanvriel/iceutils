#!/usr/bin/env python3

import numpy as np
import argparse
import iceutils as ice
import sys

def parse():
    parser = argparse.ArgumentParser(description="""
        Resample raster/stack to another.""")
    parser.add_argument('raster', type=str,
        help='Input raster/stack file.')
    parser.add_argument('reference', type=str,
        help='Reference raster/stack file.')
    parser.add_argument('output', type=str,
        help='Output raster/stack file.')
    parser.add_argument('-keys', action='store', type=str, nargs='+', default=['data', 'weights'],
        help='List of stack keys to resample. Default: data, weights.')
    return parser.parse_args()

def main(args):
    
    # Load the raster/stack and reference raster/stack
    if args.raster.endswith('.h5'):

        # Input stack
        inobj = ice.Stack(args.raster)

        # Reference stack
        ref = ice.Stack(args.reference)

        # Initialize output stack
        outobj = ice.Stack(args.output, mode='w')
        outobj.initialize(inobj.tdec, ref.hdr, data=False, weights=False)

        # Loop over keys to resample
        for key in args.keys:
            print('Resampling', key)
            inobj.resample(ref.hdr, outobj, key=key)

    else:
        
        # Input raster
        inobj = ice.Raster(rasterfile=args.raster)

        # Reference raster
        ref = ice.Raster(rasterfile=args.reference)

        # Resample
        inobj.resample(ref.hdr)

        # Write to disk
        raster.write_gdal(args.output, epsg=3413)

        
if __name__ == '__main__':
    # Parse command line arguments
    args = parse()
    # Run main
    main(args)

# end of file
