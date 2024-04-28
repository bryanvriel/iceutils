#!/usr/bin/env python3

# externals
import numpy as np
import argparse
import sys
import os

# personals
import iceutils as ice


def parse():
    parser = argparse.ArgumentParser(
        description="""
        Fit long term trends to data and remove."""
    )
    parser.add_argument(
        "stackfile",
        metavar="Stackfile",
        type=str,
        help="Input stack file to subset from.",
    )
    parser.add_argument(
        "-o,--outdir",
        action="store",
        type=str,
        default="output",
        help='Output directory. Default: "output".',
        dest="outdir",
    )
    parser.add_argument(
        "-dkey",
        action="store",
        type=str,
        default="data",
        help="HDF5 Dataset name for time series data. Default: data.",
    )
    parser.add_argument(
        "-penalty",
        "--penalty",
        action="store",
        default=1.0,
        type=float,
        dest="penalty",
        help="Penalty for inversion problem. Default: 1.0",
    )
    parser.add_argument(
        "-rw_iter",
        action="store",
        type=int,
        default=1,
        help="Number of sparse re-weighting iterations. Default: 1.",
    )
    parser.add_argument(
        "-interp",
        metavar="N",
        action="store",
        dest="interp",
        type=int,
        help="Output interpolated time series to have length N.",
    )
    parser.add_argument(
        "-solver",
        action="store",
        default="ridge",
        help="Specify the solver type {ridge, lasso, omp}. Default: ridge.",
    )
    parser.add_argument(
        "-n_min",
        action="store",
        type=int,
        default=20,
        help="Minimum number of observations to perform inversion. Default: 20.",
    )
    parser.add_argument(
        "-n_proc",
        action="store",
        type=int,
        default=1,
        help="Number of processors for multiprocessing. Default: 1.",
    )
    parser.add_argument(
        "-prior_cov",
        action="store_true",
        help="Use prior covariance matrix function defined in time collection file.",
    )
    parser.add_argument(
        "-no_weights",
        action="store_true",
        help="Do not use data weights during the inversion.",
    )
    parser.add_argument(
        "-mask",
        action="store",
        type=str,
        default=None,
        help="Raster of mask of valid pixels to invert.",
    )
    parser.add_argument(
        "-cleaned_stack",
        action="store",
        type=str,
        default=None,
        help="Output cleaned stack filename. Default: None.",
    )
    parser.add_argument(
        "-n_iter",
        action="store",
        type=int,
        default=1,
        help="Number of least squares iterations for outlier removal. Default: 1.",
    )
    parser.add_argument(
        "-n_std",
        action="store",
        type=float,
        default=3.0,
        help="Number of stddev. for outlier removal. Default: 3.0",
    )
    parser.add_argument(
        "-user",
        action="store",
        type=str,
        default="userCollection.py",
        help="Python file defining time function collection. Default: userCollection.py.",
    )
    return parser.parse_args()


def main(args):

    # Load stack
    stack = ice.Stack(args.stackfile)

    # Make sure output directory exists
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    # Launch solver
    ice.tseries.inversion(
        stack,
        args.user,
        args.outdir,
        nt_out=args.interp,
        dkey=args.dkey,
        solver_type=args.solver,
        n_proc=args.n_proc,
        regParam=args.penalty,
        rw_iter=args.rw_iter,
        n_min=args.n_min,
        no_weights=args.no_weights,
        prior_cov=args.prior_cov,
        mask_raster=args.mask,
        cleaned_stack=args.cleaned_stack,
        n_iter=args.n_iter,
        n_std=args.n_std,
    )


if __name__ == "__main__":
    # Parse command line arguments
    args = parse()
    # Run main
    main(args)

# end of file
