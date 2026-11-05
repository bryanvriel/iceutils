"""
Shared command-line argument helpers.
"""

import argparse


def parse_indexer(value):
    """
    Parse a dimension indexer written as ``DIM=INDEX``.
    """
    try:
        dim, index = value.split('=', 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'Indexers must be provided as DIM=INDEX.'
        ) from exc

    if not dim:
        raise argparse.ArgumentTypeError('Indexer dimension cannot be empty.')

    try:
        index = int(index)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'Indexer index must be an integer.'
        ) from exc

    return dim, index
