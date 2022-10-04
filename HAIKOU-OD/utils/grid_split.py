import math
import pandas as pd
import numpy as np
import gmplot
import json
import os
import urllib.request
import sys
import time
import datetime
import backoff


def dd(x, y):
    return y[0] - x[0], y[1] - x[1]


def getPoint(root, up, right, gmap, x, y):
    upx, upy = dd(root, up)
    rgx, rgy = dd(root, right)
    sr = float(x) / float(gmap[0])
    su = float(y) / float(gmap[1])
    return root[0] + upx * su + rgx * sr, root[1] + upy * su + rgy * sr


def getGirdDict(grid_map=(5, 15)):
    '''
        for Haikou small
    '''
    _upleft = (378, 1056)
    _upright = (388, 992)  # not used
    _downleft = (195, 1029)
    _downright = (205, 965)
    pad_longitube = 110
    pad_latitube = 19
    scale_longitube = 1e3
    scale_latitube = 1e3
    grid_map = (5, 15)
    # grid_map=(1,1)

    co_map = []
    for i in range(grid_map[1] + 1):
        row = []
        for j in range(grid_map[0] + 1):
            xx, yy = getPoint(_downleft, _upleft, _downright, grid_map, j, i)
            # row.append((xx, yy))
            row.append((xx / scale_longitube + pad_longitube, yy / scale_latitube + pad_latitube))
        co_map.append(row)

    grid_dict = {}
    for i in range(1, grid_map[1] + 1):
        for j in range(1, grid_map[0] + 1):
            coord = [co_map[i - 1][j - 1], co_map[i][j - 1],
                     co_map[i][j], co_map[i - 1][j]]
            grid_dict[str(j - 1) + " " + str(i - 1)] = (coord)

    return grid_dict


def plot_on_map(gmap, latitudes, longitudes):
    gmap.plot(latitudes, longitudes, 'red', edge_width=3)


def plot_grid():
    grid_dict = getGirdDict()
    keys = grid_dict.keys()
    ''' for Haikou small '''
    gmap = gmplot.GoogleMapPlotter(20.015, 110.315, 12)

    for grid in keys:
        print(grid)
        print(grid_dict[grid])
        lat = []
        lon = []
        for location in grid_dict[grid]:
            lon_, lat_ = location
            lat.append(lat_)
            lon.append(lon_)
        plot_on_map(gmap, lat, lon)
    gmap.draw("mymap.html")


if __name__ == "__main__":
    plot_grid()
    # get_poi()
