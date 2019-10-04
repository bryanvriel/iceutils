#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def get_cmap(cmap):

    try:
        cmap = plt.get_cmap(cmap)
    except ValueError:
        from mpl_toolkits.basemap import cm
        try:
            cmap = getattr(cm, cmap)
        except AttributeError:
            print('Cannot find cmap %s. Using viridis.' % cmap)
            cmap = plt.get_cmap('viridis')

    return cmap

# end of file
