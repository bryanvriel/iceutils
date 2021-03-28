#!/urs/bin/env python3

# externals
import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import iceutils as ice

def parse():
    """ Parse command line args for script """
    parser = argparse.ArgumentParser(description='Render an animated representation of the Stack.')

    # Required
    parser.add_argument('stackfile', type=str,
        help='Input Stack file to animate.')

    # Optional (alphabetized)
    parser.add_argument('-a', '-alpha', '--alpha', action='store', type=float, default=1.0,
        help='Image alpha for displaying raster. Default: 1.0.')
    parser.add_argument('-clabel', '--clabel', action='store', type=str, default='Velocity (m/yr)',
        help='Color bar label for animated plot. Default: Velocity (m/yr).')
    parser.add_argument('-clim', '--clim', action='store', type=float, nargs=2, default=[None, None], 
        help='Color limits for display.')
    parser.add_argument('-cmap', '--cmap', action='store', type=str, default='GMT_haxby',
        help='Matplotlib colormap to use for displaying raster. Default: GMT_haxby.')
    parser.add_argument('-dpi', '--dpi', action='store', type=int, default=250, 
        help='Image quality for animation. Default: 250.')
    parser.add_argument('-figsize', '--figsize', action='store', type=int, nargs=2, default=[9, 9],
        help='Figure size for animation. Default: (9,9).')
    parser.add_argument('-fps', '--fps', action='store', type=int, default=5, 
        help='Frames per second for the animation. Default: 5.')
    parser.add_argument('-k', '-key', '--key', action='store', type=str, default='data', 
        help='Dataset to view. Default: data.')
    parser.add_argument('-r', '-ref', '--ref', action='store', type=str, default=None,
        help='Reference SAR raster for background. Default: None.')
    parser.add_argument('-s', '-save', '--save', action='store', type=str, default='animation.mp4', 
        help='Name of file to save animation as. Default: animation.mp4.')
    parser.add_argument('-show', '--show', action='store_true', default=False,
        help='Show generated animation. Default: False.')
    parser.add_argument('-t', '-title', '--title', action='store', type=str, default='',
        help='Title in animated plot')
    parser.add_argument('-x', '-xlabel', '--xlabel', action='store', type=str, default='X (m)',
        help='x axis label in animated plot. Default: X (m).')
    parser.add_argument('-y', '-ylabel', '--ylabel', action='store', type=str, default='Y (m)',
        help='y axis label in animated plot. Default: Y (m).')
    return parser.parse_args()

def main(args):
    # Load Stack
    stack = ice.Stack(args.stackfile)
    
    # Load reference SAR image
    if args.ref is not None:
        sar = ice.Raster(rasterfile=args.ref)
        if sar.hdr != stack.hdr:
            sar.resample(stack.hdr)
        db = 10.0 * np.log10(sar.data)
        low = np.percentile(db.ravel(), 5)
        high = np.percentile(db.ravel(), 99.9)
    else:
        db = None

    # Set up animation
    fig, ax = plt.subplots(figsize=args.figsize)
    data = stack._datasets[args.key]
    cmap = ice.get_cmap(args.cmap)

    # Add ref image if using
    if db is not None:
        ax.imshow(db, aspect='auto', cmap='gray', vmin=low, vmax=high,
                        extent=stack.hdr.extent)

    im = ax.imshow(data[0], extent=stack.hdr.extent, cmap=cmap, clim=args.clim,
        alpha=args.alpha)

    # Create title
    datestr = ice.tdec2datestr(stack.tdec[0])
    tx = ax.set_title(args.title + ' ' + datestr, fontweight='bold')

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)

    # Add colorbar
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(args.clabel)

    # Update the frame
    def animate(i):
        im.set_data(data[i])
        datestr = ice.tdec2datestr(stack.tdec[i])
        tx.set_text(args.title + ' ' + datestr)

    fig.set_tight_layout(True)

    print('Generating animation and saving to', args.save)
    interval = 1000/args.fps # Convert fps to interval in milliseconds
    anim = animation.FuncAnimation(fig, animate, interval=interval, frames=len(data), 
        repeat=True)
    anim.save(args.save, dpi=args.dpi)

    if args.show:
        plt.show()

if __name__ == '__main__':
    args = parse()
    main(args)

# end of file