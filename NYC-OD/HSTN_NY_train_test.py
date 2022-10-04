import numpy as np
import tensorflow as tf

from utils.dataset_NY_his import load_data
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.keras.callbacks import LearningRateScheduler
import HSTN_model_NY
import pysnooper
import os

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

np.random.seed(1337)

# arguments
timestep = 5
N = 75
h = 15
w = 5
dim = 128
rate = 0.3
odmax = 241  # value of the max element in OD matrix
T = 48  ## number of time intervals in one day
days_test = 60
len_test = T * days_test
learning_rate = 0.001
num_epochs = 600
batch_size = 64

model = HSTN_model_NY.AttnSeq2Seq(N, h, w, dim, rate, timestep, out_seq_len=1, is_seq=False)

model = model.call()

# split train / test
X, Y, _, semantic, geo, weather = load_data(odmax, timestep)
geo = np.tile(np.reshape(geo, (1, 1, N, N)), (X.shape[0], X.shape[1], 1, 1))  # (len, t, N, N)
len_train = (X.shape[0] - len_test) // batch_size * batch_size  # 14528

X_train, X_test = X[:len_train], X[-len_test:]
Y_train, Y_test = Y[:len_train], Y[-len_test:]
weather_train, weather_test = weather[:len_train], weather[-len_test:]
semantic_train, semantic_test = semantic[:len_train], semantic[-len_test:]
geo_train, geo_test = geo[:len_train], geo[-len_test:]

X_train = [X_train, weather_train, semantic_train, geo_train]

X_test = [X_test, weather_test, semantic_test, geo_test]


# 改变学习率
def scheduler(epoch):
    if epoch < 400:
        return learning_rate
    elif epoch < 500:
        return learning_rate * 0.1
    else:
        return learning_rate * 0.01


change_Lr = LearningRateScheduler(scheduler)


def create_model():
    try:
        model_file = '/home/cty/py_file/CSTN-master/CSTN-master/model_dir/HSTN_model_NY.h5'
        print("loading: " + str(model_file))
        model.load_weights(model_file)
        return model

    except OSError:
        print('no file')
        return model


# 使用snoop装饰器
@pysnooper.snoop()
def train(change_lr=True):
    model = create_model()
    print(model.summary())
    # 共训练700个epoch, bs=64
    if change_lr:  # 阶段性改变lr

        model.fit(X_train, Y_train, verbose=1, batch_size=64, epochs=num_epochs, shuffle=True, callbacks=[change_Lr])

    else:  # 不改变lr
        model.fit(X_train, Y_train, verbose=1, batch_size=64, epochs=num_epochs, shuffle=True)

    model.save_weights('/home/cty/py_file/CSTN-master/CSTN-master/model_dir/HSTN_model_NY.h5')


def evaluate():
    model = create_model()
    print(model.evaluate(X_train, Y_train, batch_size=64, verbose=1))
    print(model.evaluate(X_test, Y_test, batch_size=64, verbose=1))


train()
evaluate()
