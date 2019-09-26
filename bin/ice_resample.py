#!/usr/bin/env python3

import numpy as np
import argparse
import iceutils as ice
import sys

def parse():
    parser = argparse.ArgumentParser(description="""
        Resample raster to another.""")
    parser.add_argument('raster', type=str,
        help='Input raster file.')
    parser.add_argument('reference', type=str,
        help='Reference raster file.')
    parser.add_argument('output', type=str,
        help='Output raster file.')
    return parser.parse_args()

def main(args):

    # Load the raster
    raster = ice.Raster(rasterfile=args.raster)

    # Load the reference
    ref = ice.Raster(rasterfile=args.reference)

    # Resample
    raster.resample(ref.hdr)

    # Write
    raster.write_gdal(args.output, epsg=3413)
        
if __name__ == '__main__':
    # Parse command line arguments
    args = parse()
    # Run main
    main(args)

# end of file
