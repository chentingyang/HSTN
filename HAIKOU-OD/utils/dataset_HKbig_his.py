# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import numpy as np
from numpy import newaxis
import pandas as pd
import math
from datetime import datetime
from sklearn.preprocessing import normalize, StandardScaler

np.random.seed(1337)  # for reproducibility
T = 48
map_height = 15
map_width = 5
map_hw = map_height * map_width
mean = 1.3637122475900558
std = 4.187584911837618
N = 75
a = [1, 2, 3]


# Geographic Neighbor Adjacency Matrix
def generate_adj_matrix():
    adj_matrix = np.zeros(shape=(N, N))

    for i in range(N):
        if i % 5 == 0:
            if i == 0:
                adj_matrix[i][i + 1] = 1
                adj_matrix[i][i + 5] = 1
            elif i == 70:
                adj_matrix[i][i + 1] = 1
                adj_matrix[i][i - 5] = 1
            else:
                adj_matrix[i][i + 1] = 1
                adj_matrix[i][i + 5] = 1
                adj_matrix[i][i - 5] = 1
        elif (i + 1) % 5 == 0:
            if i == 4:
                adj_matrix[i][i - 1] = 1
                adj_matrix[i][i + 5] = 1
            elif i == 74:
                adj_matrix[i][i - 1] = 1
                adj_matrix[i][i - 5] = 1
            else:
                adj_matrix[i][i - 1] = 1
                adj_matrix[i][i + 5] = 1
                adj_matrix[i][i - 5] = 1
        elif i in [1, 2, 3]:
            adj_matrix[i][i + 1] = 1
            adj_matrix[i][i - 1] = 1
            adj_matrix[i][i + 5] = 1
        elif i in [71, 72, 73]:
            adj_matrix[i][i + 1] = 1
            adj_matrix[i][i - 1] = 1
            adj_matrix[i][i - 5] = 1
        else:
            adj_matrix[i][i + 1] = 1
            adj_matrix[i][i - 1] = 1
            adj_matrix[i][i + 5] = 1
            adj_matrix[i][i - 5] = 1

    # adj_matrix[range(adj_matrix.shape[0]), range(adj_matrix.shape[1])] += 1
    return normalize(adj_matrix, 'l1')


# Generate a dynamic adjacency matrix (according to whether there is traffic flow between the two sections at this moment)
def generate_dynamic_adj_matrix(data):
    '''
        input: OD matrix of this interval, shape = [N, N]
    '''
    semantic = data + data.T
    semantic[range(semantic.shape[0]), range(semantic.shape[1])] = semantic[range(semantic.shape[0]), range(
        semantic.shape[1])] / 2
    semantic[range(semantic.shape[0]), range(semantic.shape[1])] += 1  # 添加自环

    return normalize(semantic, 'l1')


def load_data(odmax, timestep, scaler=True):
    '''
        expectation:
        X = (sample, timestep, map_height * map_width, map_height, map_width)
        Y = (sample, map_height * map_width, map_height, map_width)
        weather = (sample, timestep, ?)
    '''
    oddata = '../data/oddatabig.npy'
    weather = '../data/weather.npy'

    print("*************************")
    print("load data")
    print("*************************")
    oddata = np.load(oddata, allow_pickle=True, encoding='bytes')[()]  # （8827， 75， 15， 5）
    weather = np.load(weather, allow_pickle=True, encoding='bytes')[()]  # (8827, 7)

    print("*************************")
    print("load data done")
    print("*************************")
    print("*************************")
    print("generate sequence")
    print("*************************")

    # generate semantic neighbors
    data = np.array(oddata)
    data = np.reshape(data, (-1, N, N))

    semantic = []
    for graph in data:
        semantic.append(generate_dynamic_adj_matrix(graph))

    semantic = np.array(semantic)  # (len, N, N)

    # generate geographic neib
    geo = generate_adj_matrix()

    oddata = {0: oddata}
    weather = {0: weather}
    if scaler:
        for i in oddata.keys():
            oddata[i] = oddata[i] * 2.0 / odmax - 1.0  # (-1, 1)
            # oddata[i] = (oddata[i] - mean) / std  #standardscaler

    o = []
    w = []
    n = []
    y = []
    s = []

    for i in oddata.keys():
        oddata_set = oddata[i]
        weather_set = weather[i]

        o.append(np.concatenate([oddata_set[T + i:i - timestep, newaxis, ...] for i in range(timestep)], axis=1))
        y.append(oddata_set[T + timestep:, ...])

        n.append(oddata_set[timestep:-T, ...])
        w.append(np.concatenate([weather_set[T + i:i - timestep, newaxis, ...] for i in range(timestep)], axis=1))

        s.append(np.concatenate([semantic[T + i:i - timestep, newaxis, ...] for i in range(timestep)], axis=1))

    o = np.concatenate(o)
    y = np.concatenate(y)
    w = np.concatenate(w)
    s = np.concatenate(s)
    n = np.concatenate(n)
    print(geo.shape)

    print("*************************")
    print("generate sequence done")
    print("*************************")

    return o, y, w, s, geo, n
