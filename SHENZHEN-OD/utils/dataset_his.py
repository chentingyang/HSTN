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

N = 172
odmax = 988


def generate_adj_matrix(adj_matrix):
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
    semantic[range(semantic.shape[0]), range(semantic.shape[1])] += 1

    return normalize(semantic, 'l1')


def load_data(odmax, timestep, scaler=True):
    '''
        expectation:
        o = (sample, timestep, num_nodes, num_nodes), od data sequence
        y = (sample, num_nodes, num_nodes), ground_truth
        w = (sample, timestep, ?), meterological data sequence
        s = (timestep, num_nodes, num_nodes), semantic neb_matrix sequence
        geo = (num_nodes, num_nodes), adjacency neb_matrix
    '''
    oddata = './data/oddata.npy'
    weather = './data/weatherdata.npy'
    matrix = './data/matrix.npy'

    print("*************************")
    print("load data")
    print("*************************")
    oddata = np.load(oddata, allow_pickle=True, encoding='bytes')[()]
    weather = np.load(weather, allow_pickle=True, encoding='bytes')[()]
    matrix = np.load(matrix, allow_pickle=True, encoding='bytes')[()]
    print("*************************")
    print("load data done")
    print("*************************")

    # generate semantic neb_matrix (based on the bidirectional traffic flow)
    data = np.array(oddata)
    data = np.reshape(data, (-1, N, N))

    semantic = []
    for graph in data:
        semantic.append(generate_dynamic_adj_matrix(graph))
    semantic = np.array(semantic)  # (len, N, N)

    # generate adjacency neb_matrix
    geo = generate_adj_matrix(matrix)

    oddata = {0: oddata}
    weather = {0: weather}
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

        o.append(np.concatenate([oddata_set[i:i - timestep, newaxis, ...] for i in range(timestep)], axis=1))
        y.append(oddata_set[timestep:, ...])
        w.append(np.concatenate([weather_set[i:i - timestep, newaxis, ...] for i in range(timestep)], axis=1))
        s.append(np.concatenate([semantic[i:i - timestep, newaxis, ...] for i in range(timestep)], axis=1))

    o = np.concatenate(o)
    y = np.concatenate(y)
    w = np.concatenate(w)
    s = np.concatenate(s)

    return o, y, w, s, geo


def load_data_seq(odmax, timestep, seq_out_len, scaler=True):
    '''
        expectation:
        o = (sample, timestep, num_nodes, num_nodes), od data sequence
        y = (sample, seq_out_len, num_nodes, num_nodes), ground_truth
        w = (sample, timestep, ?), meterological data sequence
        s = (timestep, num_nodes, num_nodes), semantic neb_matrix sequence
        geo = (num_nodes, num_nodes), adjacency neb_matrix
    '''
    oddata = './data/oddata.npy'
    weather = './data/weatherdata.npy'
    matrix = './data/matrix.npy'

    print("*************************")
    print("load data")
    print("*************************")
    oddata = np.load(oddata, allow_pickle=True, encoding='bytes')[()]
    weather = np.load(weather, allow_pickle=True, encoding='bytes')[()]
    matrix = np.load(matrix, allow_pickle=True, encoding='bytes')[()]
    print("*************************")
    print("load data done")
    print("*************************")
    print("*************************")
    print("generate sequence")
    print("*************************")

    # generate semantic neb_matrix (based on the bidirectional traffic flow)
    data = np.array(oddata)
    data = np.reshape(data, (-1, N, N))

    semantic = []
    for graph in data:
        semantic.append(generate_dynamic_adj_matrix(graph))
    semantic = np.array(semantic)  # (len, N, N)

    # generate adjacency neb_matrix
    geo = generate_adj_matrix(matrix)

    oddata = {0: oddata}
    weather = {0: weather}
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

        o.append(np.concatenate([oddata_set[i:i - 2 * timestep, newaxis, ...] for i in range(timestep)], axis=1))
        y.append( \
            np.concatenate([oddata_set[i + timestep:i - timestep, newaxis, ...] for i in range(seq_out_len)], axis=1))
        w.append(np.concatenate([weather_set[i:i - 2 * timestep, newaxis, ...] for i in range(timestep)], axis=1))
        s.append(np.concatenate([semantic[i:i - 2 * timestep, newaxis, ...] for i in range(timestep)], axis=1))

    o = np.concatenate(o)
    y = np.concatenate(y)
    w = np.concatenate(w)
    s = np.concatenate(s)

    print("*************************")
    print("generate sequence done")
    print("*************************")

    return o, y, w, s, geo
