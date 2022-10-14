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

    return normalize(adj_matrix, 'l1')


# Generate a dynamic adjacency matrix (according to whether there is traffic flow between the two sections at this moment)
def generate_dynamic_adj_matrix(data):
    '''
        input: OD matrix of this interval, shape = [N, N]
    '''
    semantic = data + data.T
    semantic[range(semantic.shape[0]), range(semantic.shape[1])] = semantic[range(semantic.shape[0]), range(
        semantic.shape[1])] / 2
    semantic[range(semantic.shape[0]), range(semantic.shape[1])] += 1

    return normalize(semantic, 'l1')


def load_data(odmax, timestep, scaler=True):
    '''
        expectation:
        o = (sample, timestep, map_height * map_width, map_height, map_width), od data sequence
        y = (sample, map_height * map_width, map_height, map_width), ground_truth
        w = (sample, timestep, ?), meterological data sequence
        s = (timestep, map_height * map_width, map_height * map_width), semantic neb_matrix sequence
        geo = (map_height * map_width, map_height * map_width), adjacency neb_matrix
    '''
    oddata = './data/oddatasmall.npy'
    weather = './data/weather.npy'

    print("*************************")
    print("load data")
    print("*************************")
    oddata = np.load(oddata, allow_pickle=True, encoding='bytes')[()]
    weather = np.load(weather, allow_pickle=True, encoding='bytes')[()]

    print("*************************")
    print("load data done")
    print("*************************")

    sets = len(oddata.keys())

    # generate semantic neb_matrix (based on the bidirectional traffic flow)
    data = np.array(oddata[0])
    data = np.reshape(data, (-1, N, N))
    semantic = []
    for graph in data:
        semantic.append(generate_dynamic_adj_matrix(graph))

    semantic = np.array(semantic)

    # generate adjacency neb_matrix
    geo = generate_adj_matrix()

    if scaler:
        for i in oddata.keys():
            oddata[i] = oddata[i] * 2.0 / odmax - 1.0  # (-1, 1)
            # oddata[i] = (oddata[i] - mean) / std  #standardscaler

    o = []
    w = []
    y = []
    s = []

    for i in oddata.keys():
        oddata_set = oddata[i]
        weather_set = weather[i]

        o.append(np.concatenate([oddata_set[T + i:i - timestep, newaxis, ...] for i in range(timestep)], axis=1))
        y.append(oddata_set[T + timestep:, ...])
        w.append(np.concatenate([weather_set[T + i:i - timestep, newaxis, ...] for i in range(timestep)], axis=1))
        s.append(np.concatenate([semantic[T + i:i - timestep, newaxis, ...] for i in range(timestep)], axis=1))

    o = np.concatenate(o)
    y = np.concatenate(y)
    w = np.concatenate(w)
    s = np.concatenate(s)

    return o, y, s, geo, w


def load_data_seq(odmax, timestep, seq_out_len, scaler=True):
    '''
        expectation:
        o = (sample, timestep, map_height * map_width, map_height, map_width), od data sequence
        y = (sample, seq_out_len, map_height * map_width, map_height, map_width), ground_truth
        w = (sample, timestep, ?), meterological data sequence
        s = (timestep, map_height * map_width, map_height * map_width), semantic neb_matrix sequence
        geo = (map_height * map_width, map_height * map_width), adjacency neb_matrix
    '''
    oddata = './data/oddatasmall.npy'
    weather = './data/weather.npy'

    print("*************************")
    print("load data")
    print("*************************")
    oddata = np.load(oddata, allow_pickle=True, encoding='bytes')[()]
    weather = np.load(weather, allow_pickle=True, encoding='bytes')[()]

    print("*************************")
    print("load data done")
    print("*************************")
    print("*************************")
    print("generate sequence")
    print("*************************")

    sets = len(oddata.keys())

    # generate semantic neb_matrix (based on the bidirectional traffic flow)
    data = np.array(oddata[0])
    data = np.reshape(data, (-1, N, N))
    semantic = []
    for graph in data:
        semantic.append(generate_dynamic_adj_matrix(graph))

    semantic = np.array(semantic)

    # generate adjacency neb_matrix
    geo = generate_adj_matrix()

    if scaler:
        for i in oddata.keys():
            oddata[i] = oddata[i] * 2.0 / odmax - 1.0  # (-1, 1)
            # oddata[i] = (oddata[i] - mean) / std  #standardscaler

    o = []
    w = []
    y = []
    s = []

    for i in oddata.keys():
        oddata_set = oddata[i]
        weather_set = weather[i]

        o.append(np.concatenate([oddata_set[T + i:i - timestep, newaxis, ...] for i in range(timestep)], axis=1))
        y.append( \
            np.concatenate([oddata_set[T+i+timestep:i-timestep, newaxis, ...] for i in range(seq_out_len)], axis=1))
        w.append(np.concatenate([weather_set[T + i:i - timestep, newaxis, ...] for i in range(timestep)], axis=1))
        s.append(np.concatenate([semantic[T + i:i - timestep, newaxis, ...] for i in range(timestep)], axis=1))

    o = np.concatenate(o)
    y = np.concatenate(y)
    w = np.concatenate(w)
    s = np.concatenate(s)

    print("*************************")
    print("generate sequence done")
    print("*************************")

    return o, y, s, geo, w
