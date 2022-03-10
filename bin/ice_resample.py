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
    parser.add_argument('-epsg', action='store', type=int, default=None,
                        help='Force outputs EPSG code.')
    parser.add_argument('-srs_epsg', action='store', type=int, default=None,
                        help='Force source EPSG code.')
    parser.add_argument('-driver', action='store', type=str, default='ENVI',
                        help='GDAL driver. Default: ENVI.')
    parser.add_argument('-b', action='store', type=int, default=1, dest='band',
                        help='Raster band to resample. Default: 1.')
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
        inobj = ice.Raster(rasterfile=args.raster, band=args.band)
        if args.srs_epsg is not None:
            in_epsg = args.srs_epsg
            inobj.hdr._epsg = in_epsg
        else:
            in_epsg = inobj.hdr.epsg
        print(inobj.hdr.projWin)

        # Reference raster
        ref_hdr = ice.RasterInfo(args.reference)
        if args.epsg is not None:
            out_epsg = args.epsg
        else:
            out_epsg = ref_hdr.epsg
        print(ref_hdr.projWin)

        # Direct resample if EPSG codes are the same
        if in_epsg == out_epsg:
            print('Resampling')
            inobj.resample(ref_hdr)
        else:
            print('Warping from %d to %d' % (in_epsg, out_epsg))
            inobj = ice.warp(inobj, target_epsg=out_epsg, target_hdr=ref_hdr,
                             order=3, mode='constant', cval=np.nan)

        # Write to disk
        inobj.write_gdal(args.output, epsg=out_epsg, driver=args.driver)

        
if __name__ == '__main__':
    # Parse command line arguments
    args = parse()
    # Run main
    main(args)

# end of file
