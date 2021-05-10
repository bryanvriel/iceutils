#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

import iceutils as ice

def parse():
    parser = argparse.ArgumentParser(description="""
        Return raster/stack geographic information.""")
    parser.add_argument('rasterfile', type=str, help='Input raster file to query.')
    parser.add_argument('-match', action='store_true',
                        help='Force approximate projection match.')
    return parser.parse_args()

def main(args):

    # Get rasterinfo
    if args.rasterfile.endswith('.h5'):
        stack = ice.Stack(args.rasterfile)
        hdr = stack.hdr
        tdec = stack.tdec
        
    else:
        hdr = ice.RasterInfo(args.rasterfile, match=args.match)
        tdec = None

    print('Image shape: (%d, %d)' % (hdr.ny, hdr.nx))

    print('Geographic extent: %f %f %f %f' % tuple(hdr.extent))

    print('Geographic spacing: (dy = %f, dx = %f)' % (hdr.dy, hdr.dx))

    if tdec is not None:
        print('Time span: %f -> %f' % (tdec[0], tdec[-1]))
        print('Median time spacing: %f' % np.median(np.diff(tdec)))

    print('EPSG:', hdr.epsg)



if __name__ == '__main__':
    args = parse()
    main(args)

# end of file
