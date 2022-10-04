import numpy as np
import tensorflow as tf

from utils.dataset_his import load_data
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.keras.callbacks import LearningRateScheduler
import HSTN_model_SZ
import pysnooper
import os


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)





np.random.seed(1337)

#arguments
dim = 128
rate = 0.3
odmax = 988 #value of the max element in the OD matrix
T = 33 # number of time intervals in one day
days_test = 32
len_test = T * days_test
learning_rate = 0.001
num_epochs = 600
batch_size = 64
num_heads = 4
N = 172
h = 0
w = num_heads
timestep = 5

model = HSTN_model_SZ.AttnSeq2Seq(N, num_heads, dim, rate, timestep, out_seq_len=1, is_seq=False)

model = model.call()

#split train / test
X, Y, weather, semantic, geo, _ = load_data(odmax, timestep)
geo = np.tile(np.reshape(geo, (1, 1, N, N)), (X.shape[0], X.shape[1], 1, 1))
len_train = (X.shape[0] - len_test) // batch_size * batch_size 


X_train, X_test = X[:len_train], X[-len_test:]
Y_train, Y_test = Y[:len_train], Y[-len_test:]
weather_train, weather_test = weather[:len_train], weather[-len_test:]
semantic_train, semantic_test = semantic[:len_train], semantic[-len_test:]
geo_train, geo_test = geo[:len_train], geo[-len_test:]

X_train = [X_train, weather_train, semantic_train, geo_train]

X_test = [X_test, weather_test, semantic_test, geo_test]







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
		model_file = '/home/cty/py_file/CSTN-master/CSTN-master/Metro_OD/model_dir/HSTN_model_SZ.h5'
		print("loading: " + str(model_file))
		model.load_weights(model_file)
		return model

	except OSError:
		print('no file')
		return model


@pysnooper.snoop()
def train(change_lr=True):
	model = create_model()
	print(model.summary())

	if change_lr:

		model.fit(X_train, Y_train, verbose=1, batch_size=64, epochs=num_epochs, shuffle=True, callbacks=[change_Lr])

	else:
		model.fit(X_train, Y_train, verbose=1, batch_size=64, epochs=num_epochs, shuffle=True)

	model.save_weights('/home/cty/py_file/CSTN-master/CSTN-master/Metro_OD/model_dir/HSTN_model_SZ.h5')

def evaluate():
	model = create_model()
	print(model.evaluate(X_train, Y_train, batch_size=64, verbose=1))				
	print(model.evaluate(X_test, Y_test, batch_size=64, verbose=1))


train()
evaluate()