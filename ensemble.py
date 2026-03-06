from LSTMONE import Lstm1
from LSTMTWO import Lstm2
from LSTMTHREE import Lstm31, Lstm32, Lstm33
from MLP import mlp
from Data_writer import data_writer
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow.keras.optimizers as optimizers
import sklearn.metrics as metrics
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

tf.disable_eager_execution()
np.random.seed(5)
tf.set_random_seed(5)


def data_fill(data, batchsize):
    a = np.zeros([batchsize - 1, data.shape[1], data.shape[2]])
    b = np.concatenate((data, a), axis=0)
    return b


df = pd.read_excel('汇总.xlsx', header=0,
                   usecols=[1, 2, 3, 4, 6, 10, 11, 12]).values  # 1, 2, 3, 4发生量，6消耗量，10, 11, 12柜位

rawdata = df.astype('float32')

state_data = pd.read_excel('汇总.xlsx', header=0, usecols=[16]).values
print(np.shape(rawdata))


def form_aux(y1_index, y2_index, y3_index, y4_index):
    train_x1_2 = []
    train_x1_3 = []
    train_x1_4 = []
    shunxu = []
    for index in y1_index:
        i2 = 0
        i3 = 0
        i4 = 0
        m = index
        while m >= 0:
            if (m in y2_index):
                i2 = m
                train_x1_2.append(rawdata[max(0, m - x_len): m])
                break
            m -= 1
        m = index
        while m >= 0:
            if (m in y3_index):
                i3 = m
                train_x1_3.append(rawdata[max(0, m - x_len): m])
                break
            m -= 1
        m = index
        while m >= 0:
            if (m in y4_index):
                i4 = m
                train_x1_4.append(rawdata[max(0, m - x_len): m])
                break
            m -= 1
        if (i2 != 0 and i3 != 0 and i4 != 0):
            if (i2 < i3 and i3 < i4):
                shunxu.append(0)
            elif (i2 < i4 and i4 < i3):
                shunxu.append(1)
            elif (i3 < i2 and i2 < i4):
                shunxu.append(2)
            elif (i3 < i4 and i4 < i2):
                shunxu.append(3)
            elif (i4 < i2 and i2 < i3):
                shunxu.append(4)
            elif (i4 < i3 and i3 < i2):
                shunxu.append(5)
    return train_x1_2, train_x1_3, train_x1_4, shunxu


def form_con(shunxu, state_width):
    con = np.zeros((len(shunxu), 6, state_width))
    for i in range(len(shunxu)):
        if shunxu[i] == 0:
            t = np.zeros((6, state_width))
            t[0:1, :] = np.ones((1, state_width))
            con[i] = t
        elif shunxu[i] == 1:
            t = np.zeros((6, state_width))
            t[1:2, :] = np.ones((1, state_width))
            con[i] = t
        elif shunxu[i] == 2:
            t = np.zeros((6, state_width))
            t[2:3, :] = np.ones((1, state_width))
            con[i] = t
        elif shunxu[i] == 3:
            t = np.zeros((6, state_width))
            t[3:4, :] = np.ones((1, state_width))
            con[i] = t
        elif shunxu[i] == 4:
            t = np.zeros((6, state_width))
            t[4:5, :] = np.ones((1, state_width))
            con[i] = t
        elif shunxu[i] == 5:
            t = np.zeros((6, state_width))
            t[5:, :] = np.ones((1, state_width))
            con[i] = t
    return con


x_len = 60
pred_len = 60
batch_size = 64
learningrate = 0.0001
x_input_width = 8
state_width = 64
out_width = 3
verification_size = 100
traintimes = 200

# train_data1 = rawdata[]
train_data = rawdata[:22400]
test_data = rawdata[22460:28000]
print(train_data)
max_value = np.max(train_data)
min_value = np.min(train_data)
print("max_value = ", max_value)
print("min_value=", min_value)
# mean_value = np.mean(train_data)
mean_value = train_data.mean(0)
print(mean_value)

std_value = train_data.std(0)
print(std_value)
# train_data = wavelet_denoising(train_data)
# test_data = wavelet_denoising(test_data)
train_data = (train_data - mean_value) / std_value
rawdata = (rawdata - mean_value) / std_value
test_data = (test_data - mean_value) / std_value

train_size = int((len(train_data) - x_len - pred_len) * 0.8)
val_size = len(train_data) - x_len - pred_len - train_size

print(len(train_data))

y1_index = []
y2_index = []
y3_index = []
y4_index = []
index = []

# train_size = int((len(train_data)-x_len-pred_len))

print(len(train_data))
for i in range(28000):
    if ((state_data[i + x_len] == 7) or (state_data[i + x_len] == 5)):
        y1_index.append(i + x_len)
    elif (state_data[i + x_len] == 6 or state_data[i + x_len] == 4):
        y2_index.append(i + x_len)
    elif (state_data[i + x_len] == 3):
        y3_index.append(i + x_len)
    elif (state_data[i + x_len] == 2):
        y4_index.append(i + x_len)

# train_x1_2, train_x1_3, train_x1_4, shunxu1 = form_aux(np.arange(y3_index[0], train_size+x_len), y2_index, y3_index, y4_index)
# print(train_x1_4)
# train_x2_1, train_x2_3, train_x2_4, shunxu2 = form_aux(np.arange(y3_index[0], train_size+x_len), y1_index, y3_index, y4_index)
# train_x3_1, train_x3_2, train_x3_4, shunxu3 = form_aux(np.arange(y3_index[0], train_size+x_len), y1_index, y2_index, y4_index)
# train_x4_1, train_x4_2, train_x4_3, shunxu4 = form_aux(np.arange(y3_index[0], train_size+x_len), y1_index, y2_index, y3_index)


train_x = []
train_y = []
for j in range(train_size):
    # train_x.append(train_data[j:j + 1 * x_len, :])
    train_y.append(train_data[j + 1 * x_len, -3:])

# train_x = np.array(train_x)
train_y = np.array(train_y)
# shape1 = min(np.shape(train_x1_2)[0], np.shape(train_x1_3)[0], np.shape(train_x1_4)[0])
# shape2 = min(np.shape(train_x2_1)[0], np.shape(train_x2_3)[0], np.shape(train_x2_4)[0])
# shape3 = min(np.shape(train_x3_1)[0], np.shape(train_x3_2)[0], np.shape(train_x3_4)[0])
# shape4 = min(np.shape(train_x4_2)[0], np.shape(train_x4_3)[0], np.shape(train_x4_1)[0])
# shape = min(shape1, shape2, shape3, shape4)
# train_x = train_x[np.shape(train_x)[0]-shape:]
# train_x1_2 = train_x1_2[np.shape(train_x1_2)[0]-shape:]
# train_x1_3 = train_x1_3[np.shape(train_x1_3)[0]-shape:]
# train_x1_4 = train_x1_4[np.shape(train_x1_4)[0]-shape:]
# # train_x2 = train_x[np.shape(train_x)[0]-shape]
# train_x2_1 = train_x2_1[np.shape(train_x2_1)[0]-shape:]
# train_x2_3 = train_x2_3[np.shape(train_x2_3)[0]-shape:]
# train_x2_4 = train_x2_4[np.shape(train_x2_4)[0]-shape:]
# # train_x3 = train_x[np.shape(train_x)[0]-shape]
# train_x3_1 = train_x3_1[np.shape(train_x3_1)[0]-shape:]
# train_x3_2 = train_x3_2[np.shape(train_x3_2)[0]-shape:]
# train_x3_4 = train_x3_4[np.shape(train_x3_4)[0]-shape:]
# # train_x4 = train_x[np.shape(train_x)[0]-shape]
# train_x4_1 = train_x4_1[np.shape(train_x4_1)[0]-shape:]
# train_x4_2 = train_x4_2[np.shape(train_x4_2)[0]-shape:]
# train_x4_3 = train_x4_3[np.shape(train_x4_3)[0]-shape:]
# 
# shunxu1 = shunxu1[np.shape(shunxu1)[0]-shape:]
# shunxu2 = shunxu2[np.shape(shunxu2)[0]-shape:]
# shunxu3 = shunxu3[np.shape(shunxu3)[0]-shape:]
# shunxu4 = shunxu4[np.shape(shunxu4)[0]-shape:]
# 
# train_x1_2 = np.array(train_x1_2)
# train_x1_3 = np.array(train_x1_3)
# train_x1_4 = np.array(train_x1_4)
# train_x2_1 = np.array(train_x2_1)
# train_x2_3 = np.array(train_x2_3)
# train_x2_4 = np.array(train_x2_4)
# train_x3_1 = np.array(train_x3_1)
# train_x3_2 = np.array(train_x3_2)
# train_x3_4 = np.array(train_x3_4)
# train_x4_1 = np.array(train_x4_1)
# train_x4_2 = np.array(train_x4_2)
# train_x4_3 = np.array(train_x4_3)
# 
# print(np.shape(train_x))
# print(np.shape(train_x1_2))
# print(np.shape(train_x2_1))
# 
# train_con1 = form_con(shunxu1, 128)
# train_con2 = form_con(shunxu2, 64)
# train_con3 = form_con(shunxu3, 64)
# train_con4 = form_con(shunxu4, 64)
# 
# data_writer(train_x.reshape(np.shape(train_x)[0], -1), "./data/ensemble/trx.csv")
# data_writer(train_x1_2.reshape(np.shape(train_x)[0], -1), "./data/ensemble/trx12.csv")
# data_writer(train_x1_3.reshape(np.shape(train_x)[0], -1), "./data/ensemble/trx13.csv")
# data_writer(train_x1_4.reshape(np.shape(train_x)[0], -1), "./data/ensemble/trx14.csv")
# data_writer(train_x2_1.reshape(np.shape(train_x)[0], -1), "./data/ensemble/trx21.csv")
# data_writer(train_x2_3.reshape(np.shape(train_x)[0], -1), "./data/ensemble/trx23.csv")
# data_writer(train_x2_4.reshape(np.shape(train_x)[0], -1), "./data/ensemble/trx24.csv")
# data_writer(train_x3_1.reshape(np.shape(train_x)[0], -1), "./data/ensemble/trx31.csv")
# data_writer(train_x3_2.reshape(np.shape(train_x)[0], -1), "./data/ensemble/trx32.csv")
# data_writer(train_x3_4.reshape(np.shape(train_x)[0], -1), "./data/ensemble/trx34.csv")
# data_writer(train_x4_1.reshape(np.shape(train_x)[0], -1), "./data/ensemble/trx41.csv")
# data_writer(train_x4_2.reshape(np.shape(train_x)[0], -1), "./data/ensemble/trx42.csv")
# data_writer(train_x4_3.reshape(np.shape(train_x)[0], -1), "./data/ensemble/trx43.csv")
# # data_writer(train_input5, "./data/ensemble/input5.xlsx", 'input5')
# 
# data_writer(train_con1.reshape(np.shape(train_con1)[0], -1), "./data/traincon1.csv")
# data_writer(train_con2.reshape(np.shape(train_con2)[0], -1), "./data/traincon2.csv")
# data_writer(train_con3.reshape(np.shape(train_con3)[0], -1), "./data/traincon3.csv")
# data_writer(train_con4.reshape(np.shape(train_con4)[0], -1), "./data/traincon4.csv")

train_x = pd.read_csv('./data/ensemble/trx.csv').values
train_x1_2 = pd.read_csv('./data/ensemble/trx12.csv').values
train_x1_3 = pd.read_csv('./data/ensemble/trx13.csv').values
train_x1_4 = pd.read_csv('./data/ensemble/trx14.csv').values
train_x2_1 = pd.read_csv('./data/ensemble/trx21.csv').values
train_x2_3 = pd.read_csv('./data/ensemble/trx23.csv').values
train_x2_4 = pd.read_csv('./data/ensemble/trx24.csv').values
train_x3_1 = pd.read_csv('./data/ensemble/trx31.csv').values
train_x3_2 = pd.read_csv('./data/ensemble/trx32.csv').values
train_x3_4 = pd.read_csv('./data/ensemble/trx34.csv').values
train_x4_1 = pd.read_csv('./data/ensemble/trx41.csv').values
train_x4_2 = pd.read_csv('./data/ensemble/trx42.csv').values
train_x4_3 = pd.read_csv('./data/ensemble/trx43.csv').values
train_con1 = pd.read_csv('./data/traincon1.csv').values
train_con2 = pd.read_csv('./data/traincon2.csv').values
train_con3 = pd.read_csv('./data/traincon3.csv').values
train_con4 = pd.read_csv('./data/traincon4.csv').values

train_x = train_x[:, 1:].reshape((-1, x_len, 8))
train_x1_2 = train_x1_2[:, 1:].reshape((-1, x_len, 8))
train_x1_3 = train_x1_3[:, 1:].reshape((-1, x_len, 8))
train_x1_4 = train_x1_4[:, 1:].reshape((-1, x_len, 8))
train_x2_1 = train_x2_1[:, 1:].reshape((-1, x_len, 8))
train_x2_3 = train_x2_3[:, 1:].reshape((-1, x_len, 8))
train_x2_4 = train_x2_4[:, 1:].reshape((-1, x_len, 8))
train_x3_1 = train_x3_1[:, 1:].reshape((-1, x_len, 8))
train_x3_2 = train_x3_2[:, 1:].reshape((-1, x_len, 8))
train_x3_4 = train_x3_4[:, 1:].reshape((-1, x_len, 8))
train_x4_1 = train_x4_1[:, 1:].reshape((-1, x_len, 8))
train_x4_2 = train_x4_2[:, 1:].reshape((-1, x_len, 8))
train_x4_3 = train_x4_3[:, 1:].reshape((-1, x_len, 8))
print(train_con1)
train_con1 = train_con1[:, 1:].reshape((np.shape(train_con1)[0], 6, -1))
train_con2 = train_con2[:, 1:].reshape((np.shape(train_con2)[0], 6, -1))
train_con3 = train_con3[:, 1:].reshape((np.shape(train_con3)[0], 6, -1))
train_con4 = train_con4[:, 1:].reshape((np.shape(train_con4)[0], 6, -1))
train_con4 = np.concatenate((train_con4, train_con4), axis=2)
train_y = train_y[np.shape(train_y)[0] - np.shape(train_x)[0]:]
print(np.shape(train_con4))

val_x = []
val_y = []

# val_x1_2, val_x1_3, val_x1_4, shunxu1 = form_aux(np.arange(train_size+x_len, train_size+val_size+x_len), y2_index, y3_index, y4_index)
# val_x2_1, val_x2_3, val_x2_4, shunxu2 = form_aux(np.arange(train_size+x_len, train_size+val_size+x_len), y1_index, y3_index, y4_index)
# val_x3_1, val_x3_2, val_x3_4, shunxu3 = form_aux(np.arange(train_size+x_len, train_size+val_size+x_len), y1_index, y2_index, y4_index)
# val_x4_1, val_x4_2, val_x4_3, shunxu4 = form_aux(np.arange(train_size+x_len, train_size+val_size+x_len), y1_index, y2_index, y3_index)
# 
# print(val_size)
# print(train_size)
for i in range(val_size):
    # val_x.append(train_data[train_size + i: train_size + i + x_len])
    val_y.append(train_data[train_size + i + x_len, -3:])  #:i+4*x_len])

# val_x = np.array(val_x)
val_y = np.array(val_y)
# shape1 = min(np.shape(val_x1_2)[0], np.shape(val_x1_3)[0], np.shape(val_x1_4)[0])
# shape2 = min(np.shape(val_x2_1)[0], np.shape(val_x2_3)[0], np.shape(val_x2_4)[0])
# shape3 = min(np.shape(val_x3_1)[0], np.shape(val_x3_2)[0], np.shape(val_x3_4)[0])
# shape4 = min(np.shape(val_x4_2)[0], np.shape(val_x4_3)[0], np.shape(val_x4_1)[0])
# shape = min(shape1, shape2, shape3, shape4)
# val_x = val_x[np.shape(val_x)[0]-shape:]
# val_x1_2 = val_x1_2[np.shape(val_x1_2)[0]-shape:]
# val_x1_3 = val_x1_3[np.shape(val_x1_3)[0]-shape:]
# val_x1_4 = val_x1_4[np.shape(val_x1_4)[0]-shape:]
# # val_x2 = val_x[np.shape(val_x)[0]-shape]
# val_x2_1 = val_x2_1[np.shape(val_x2_1)[0]-shape:]
# val_x2_3 = val_x2_3[np.shape(val_x2_3)[0]-shape:]
# val_x2_4 = val_x2_4[np.shape(val_x2_4)[0]-shape:]
# # val_x3 = val_x[np.shape(val_x)[0]-shape3]
# val_x3_1 = val_x3_1[np.shape(val_x3_1)[0]-shape:]
# val_x3_2 = val_x3_2[np.shape(val_x3_2)[0]-shape:]
# val_x3_4 = val_x3_4[np.shape(val_x3_4)[0]-shape:]
# # val_x4 = val_x[np.shape(val_x)[0]-shape4]
# val_x4_1 = val_x4_1[np.shape(val_x4_1)[0]-shape:]
# val_x4_2 = val_x4_2[np.shape(val_x4_2)[0]-shape:]
# val_x4_3 = val_x4_3[np.shape(val_x4_3)[0]-shape:]
# shunxu1 = shunxu1[np.shape(shunxu1)[0]-shape:]
# shunxu2 = shunxu2[np.shape(shunxu2)[0]-shape:]
# shunxu3 = shunxu3[np.shape(shunxu3)[0]-shape:]
# shunxu4 = shunxu4[np.shape(shunxu4)[0]-shape:]

val_x = pd.read_csv('./data/ensemble/valx.csv').values
print("val-x")
print(val_x)
val_x1_2 = pd.read_csv('./data/ensemble/valx12.csv').values
val_x1_3 = pd.read_csv('./data/ensemble/valx13.csv').values
val_x1_4 = pd.read_csv('./data/ensemble/valx14.csv').values
val_x2_1 = pd.read_csv('./data/ensemble/valx21.csv').values
val_x2_3 = pd.read_csv('./data/ensemble/valx23.csv').values
val_x2_4 = pd.read_csv('./data/ensemble/valx24.csv').values
val_x3_1 = pd.read_csv('./data/ensemble/valx31.csv').values
val_x3_2 = pd.read_csv('./data/ensemble/valx32.csv').values
val_x3_4 = pd.read_csv('./data/ensemble/valx34.csv').values
val_x4_1 = pd.read_csv('./data/ensemble/valx41.csv').values
val_x4_2 = pd.read_csv('./data/ensemble/valx42.csv').values
val_x4_3 = pd.read_csv('./data/ensemble/valx43.csv').values
val_con1 = pd.read_csv('./data/valcon1.csv').values
val_con2 = pd.read_csv('./data/valcon2.csv').values
val_con3 = pd.read_csv('./data/valcon3.csv').values
val_con4 = pd.read_csv('./data/valcon4.csv').values

val_x = val_x[:, 1:].reshape((-1, x_len, 8))
val_x1_2 = val_x1_2[:, 1:].reshape((-1, x_len, 8))
val_x1_3 = val_x1_3[:, 1:].reshape((-1, x_len, 8))
val_x1_4 = val_x1_4[:, 1:].reshape((-1, x_len, 8))
val_x2_1 = val_x2_1[:, 1:].reshape((-1, x_len, 8))
val_x2_3 = val_x2_3[:, 1:].reshape((-1, x_len, 8))
val_x2_4 = val_x2_4[:, 1:].reshape((-1, x_len, 8))
val_x3_1 = val_x3_1[:, 1:].reshape((-1, x_len, 8))
val_x3_2 = val_x3_2[:, 1:].reshape((-1, x_len, 8))
val_x3_4 = val_x3_4[:, 1:].reshape((-1, x_len, 8))
val_x4_1 = val_x4_1[:, 1:].reshape((-1, x_len, 8))
val_x4_2 = val_x4_2[:, 1:].reshape((-1, x_len, 8))
val_x4_3 = val_x4_3[:, 1:].reshape((-1, x_len, 8))
val_con1 = val_con1[:, 1:].reshape((np.shape(val_con1)[0], 6, -1))
val_con2 = val_con2[:, 1:].reshape((np.shape(val_con2)[0], 6, -1))
val_con3 = val_con3[:, 1:].reshape((np.shape(val_con3)[0], 6, -1))
val_con4 = val_con4[:, 1:].reshape((np.shape(val_con4)[0], 6, -1))
val_con4 = np.concatenate((val_con4, val_con4), axis=2)
val_y = val_y[np.shape(val_y)[0] - np.shape(val_x)[0]:]

print(np.shape(val_x))
print(np.shape(val_x1_2))
# print(np.shape(val_x2_1))
# 
# val_x1_2 = np.array(val_x1_2)
# val_x1_3 = np.array(val_x1_3)
# val_x1_4 = np.array(val_x1_4)
# val_x2_1 = np.array(val_x2_1)
# val_x2_3 = np.array(val_x2_3)
# val_x2_4 = np.array(val_x2_4)
# val_x3_1 = np.array(val_x3_1)
# val_x3_2 = np.array(val_x3_2)
# val_x3_4 = np.array(val_x3_4)
# val_x4_1 = np.array(val_x4_1)
# val_x4_2 = np.array(val_x4_2)
# val_x4_3 = np.array(val_x4_3)
# 
# val_con1 = form_con(shunxu1, 128)
# val_con2 = form_con(shunxu2, 64)
# val_con3 = form_con(shunxu3, 64)
# val_con4 = form_con(shunxu4, 64)

# data_writer(val_con1.reshape(np.shape(val_con1)[0], -1), "./data/valcon1.csv")
# data_writer(val_con2.reshape(np.shape(val_con2)[0], -1), "./data/valcon2.csv")
# data_writer(val_con3.reshape(np.shape(val_con3)[0], -1), "./data/valcon3.csv")
# data_writer(val_con4.reshape(np.shape(val_con4)[0], -1), "./data/valcon4.csv")
# 
# data_writer(val_x.reshape(np.shape(val_x)[0], -1), "./data/ensemble/valx.csv")
# data_writer(val_x1_2.reshape(np.shape(val_x)[0], -1), "./data/ensemble/valx12.csv")
# data_writer(val_x1_3.reshape(np.shape(val_x)[0], -1), "./data/ensemble/valx13.csv")
# data_writer(val_x1_4.reshape(np.shape(val_x)[0], -1), "./data/ensemble/valx14.csv")
# data_writer(val_x2_1.reshape(np.shape(val_x)[0], -1), "./data/ensemble/valx21.csv")
# data_writer(val_x2_3.reshape(np.shape(val_x)[0], -1), "./data/ensemble/valx23.csv")
# data_writer(val_x2_4.reshape(np.shape(val_x)[0], -1), "./data/ensemble/valx24.csv")
# data_writer(val_x3_1.reshape(np.shape(val_x)[0], -1), "./data/ensemble/valx31.csv")
# data_writer(val_x3_2.reshape(np.shape(val_x)[0], -1), "./data/ensemble/valx32.csv")
# data_writer(val_x3_4.reshape(np.shape(val_x)[0], -1), "./data/ensemble/valx34.csv")
# data_writer(val_x4_1.reshape(np.shape(val_x)[0], -1), "./data/ensemble/valx41.csv")
# data_writer(val_x4_2.reshape(np.shape(val_x)[0], -1), "./data/ensemble/valx42.csv")
# data_writer(val_x4_3.reshape(np.shape(val_x)[0], -1), "./data/ensemble/valx43.csv")

test_x = []
test_y = []
for j in range(len(test_data) - x_len):
    # test_x.append(test_data[j:j + 1 * x_len, :])
    test_y.append(test_data[j + 1 * x_len, -3:])

# test_x = np.array(test_x)
# test_x = np.reshape(test_x, (test_x.shape[0], x_len, -1))

# test_y = np.array(test_y).reshape(test_x.shape[0], -1)

# test_x1_2, test_x1_3, test_x1_4, shunxu1 = form_aux(np.arange(22460+x_len, 28000), y2_index, y3_index, y4_index)
# test_x2_1, test_x2_3, test_x2_4, shunxu2 = form_aux(np.arange(22460+x_len, 28000), y1_index, y3_index, y4_index)
# test_x3_1, test_x3_2, test_x3_4, shunxu3 = form_aux(np.arange(22460+x_len, 28000), y1_index, y2_index, y4_index)
# test_x4_1, test_x4_2, test_x4_3, shunxu4 = form_aux(np.arange(22460+x_len, 28000), y1_index, y2_index, y3_index)
#
# shape1 = min(np.shape(test_x1_2)[0], np.shape(test_x1_3)[0], np.shape(test_x1_4)[0])
# shape2 = min(np.shape(test_x2_1)[0], np.shape(test_x2_3)[0], np.shape(test_x2_4)[0])
# shape3 = min(np.shape(test_x3_1)[0], np.shape(test_x3_2)[0], np.shape(test_x3_4)[0])
# shape4 = min(np.shape(test_x4_2)[0], np.shape(test_x4_3)[0], np.shape(test_x4_1)[0])
# shape = min(shape1, shape2, shape3, shape4)
# test_x = test_x[np.shape(test_x)[0]-shape:]
# test_y = test_y[np.shape(test_y)[0]-shape:]
# test_x1_2 = test_x1_2[np.shape(test_x1_2)[0]-shape:]
# test_x1_3 = test_x1_3[np.shape(test_x1_3)[0]-shape:]
# test_x1_4 = test_x1_4[np.shape(test_x1_4)[0]-shape:]
# # test_x2 = test_x[np.shape(test_x)[0]-shape:]
# test_x2_1 = test_x2_1[np.shape(test_x2_1)[0]-shape:]
# test_x2_3 = test_x2_3[np.shape(test_x2_3)[0]-shape:]
# test_x2_4 = test_x2_4[np.shape(test_x2_4)[0]-shape:]
# # test_x3 = test_x[np.shape(test_x)[0]-shape3]
# test_x3_1 = test_x3_1[np.shape(test_x3_1)[0]-shape:]
# test_x3_2 = test_x3_2[np.shape(test_x3_2)[0]-shape:]
# test_x3_4 = test_x3_4[np.shape(test_x3_4)[0]-shape:]
# # test_x4 = test_x[np.shape(test_x)[0]-shape4]
# test_x4_1 = test_x4_1[np.shape(test_x4_1)[0]-shape:]
# test_x4_2 = test_x4_2[np.shape(test_x4_2)[0]-shape:]
# test_x4_3 = test_x4_3[np.shape(test_x4_3)[0]-shape:]
#
# shunxu1 = shunxu1[np.shape(shunxu1)[0]-shape:]
# shunxu2 = shunxu2[np.shape(shunxu2)[0]-shape:]
# shunxu3 = shunxu3[np.shape(shunxu3)[0]-shape:]
# shunxu4 = shunxu4[np.shape(shunxu4)[0]-shape:]
test_x = pd.read_csv('./data/ensemble/testx.csv').values
test_x1_2 = pd.read_csv('./data/ensemble/testx12.csv').values
test_x1_3 = pd.read_csv('./data/ensemble/testx13.csv').values
test_x1_4 = pd.read_csv('./data/ensemble/testx14.csv').values
test_x2_1 = pd.read_csv('./data/ensemble/testx21.csv').values
test_x2_3 = pd.read_csv('./data/ensemble/testx23.csv').values
test_x2_4 = pd.read_csv('./data/ensemble/testx24.csv').values
test_x3_1 = pd.read_csv('./data/ensemble/testx31.csv').values
test_x3_2 = pd.read_csv('./data/ensemble/testx32.csv').values
test_x3_4 = pd.read_csv('./data/ensemble/testx34.csv').values
test_x4_1 = pd.read_csv('./data/ensemble/testx41.csv').values
test_x4_2 = pd.read_csv('./data/ensemble/testx42.csv').values
test_x4_3 = pd.read_csv('./data/ensemble/testx43.csv').values
test_con1 = pd.read_csv('./data/testcon1.csv').values
test_con2 = pd.read_csv('./data/testcon2.csv').values
test_con3 = pd.read_csv('./data/testcon3.csv').values
test_con4 = pd.read_csv('./data/testcon4.csv').values
# 
test_y = test_y[np.shape(test_y)[0] - np.shape(test_x)[0]:]

test_x = test_x[:, 1:].reshape((-1, x_len, 8))
test_x1_2 = test_x1_2[:, 1:].reshape((-1, x_len, 8))
test_x1_3 = test_x1_3[:, 1:].reshape((-1, x_len, 8))
test_x1_4 = test_x1_4[:, 1:].reshape((-1, x_len, 8))
test_x2_1 = test_x2_1[:, 1:].reshape((-1, x_len, 8))
test_x2_3 = test_x2_3[:, 1:].reshape((-1, x_len, 8))
test_x2_4 = test_x2_4[:, 1:].reshape((-1, x_len, 8))
test_x3_1 = test_x3_1[:, 1:].reshape((-1, x_len, 8))
test_x3_2 = test_x3_2[:, 1:].reshape((-1, x_len, 8))
test_x3_4 = test_x3_4[:, 1:].reshape((-1, x_len, 8))
test_x4_1 = test_x4_1[:, 1:].reshape((-1, x_len, 8))
test_x4_2 = test_x4_2[:, 1:].reshape((-1, x_len, 8))
test_x4_3 = test_x4_3[:, 1:].reshape((-1, x_len, 8))

test_con1 = test_con1[:, 1:].reshape((np.shape(test_con1)[0], 6, -1))
test_con2 = test_con2[:, 1:].reshape((np.shape(test_con2)[0], 6, -1))
test_con3 = test_con3[:, 1:].reshape((np.shape(test_con3)[0], 6, -1))
test_con4 = test_con4[:, 1:].reshape((np.shape(test_con4)[0], 6, -1))
test_con4 = np.concatenate((test_con4, test_con4), axis=2)
test_x1_2 = np.array(test_x1_2)
test_x1_3 = np.array(test_x1_3)
test_x1_4 = np.array(test_x1_4)
test_x2_1 = np.array(test_x2_1)
test_x2_3 = np.array(test_x2_3)
test_x2_4 = np.array(test_x2_4)
test_x3_1 = np.array(test_x3_1)
test_x3_2 = np.array(test_x3_2)
test_x3_4 = np.array(test_x3_4)
test_x4_1 = np.array(test_x4_1)
test_x4_2 = np.array(test_x4_2)
test_x4_3 = np.array(test_x4_3)
test_y = np.array(test_y)
# test_con1 = form_con(shunxu1, 128)
# test_con2 = form_con(shunxu2, 64)
# test_con3 = form_con(shunxu3, 64)
# test_con4 = form_con(shunxu4, 64)

# data_writer(test_con1.reshape(np.shape(test_con1)[0], -1), "./data/testcon1.xlsx", "con1")
# data_writer(test_con2.reshape(np.shape(test_con2)[0], -1), "./data/testcon2.xlsx", "con2")
# data_writer(test_con3.reshape(np.shape(test_con3)[0], -1), "./data/testcon3.xlsx", "con3")
# data_writer(test_con4.reshape(np.shape(test_con4)[0], -1), "./data/testcon4.xlsx", "con4")

# print(np.shape(test_x))
# print(np.shape(test_x1_2))
# print(np.shape(test_y))

# data_writer(test_x.reshape(np.shape(test_x)[0], -1), "./data/ensemble/testx.xlsx", 'input1')
# data_writer(test_x1_2.reshape(np.shape(test_x)[0], -1), "./data/ensemble/testx12.xlsx", 'input2')
# data_writer(test_x1_3.reshape(np.shape(test_x)[0], -1), "./data/ensemble/testx13.xlsx", 'input3')
# data_writer(test_x1_4.reshape(np.shape(test_x)[0], -1), "./data/ensemble/testx14.xlsx", 'input4')
# data_writer(test_x2_1.reshape(np.shape(test_x)[0], -1), "./data/ensemble/testx21.xlsx", 'input2')
# data_writer(test_x2_3.reshape(np.shape(test_x)[0], -1), "./data/ensemble/testx23.xlsx", 'input3')
# data_writer(test_x2_4.reshape(np.shape(test_x)[0], -1), "./data/ensemble/testx24.xlsx", 'input4')
# data_writer(test_x3_1.reshape(np.shape(test_x)[0], -1), "./data/ensemble/testx31.xlsx", 'input2')
# data_writer(test_x3_2.reshape(np.shape(test_x)[0], -1), "./data/ensemble/testx32.xlsx", 'input3')
# data_writer(test_x3_4.reshape(np.shape(test_x)[0], -1), "./data/ensemble/testx34.xlsx", 'input4')
# data_writer(test_x4_1.reshape(np.shape(test_x)[0], -1), "./data/ensemble/testx41.xlsx", 'input2')
# data_writer(test_x4_2.reshape(np.shape(test_x)[0], -1), "./data/ensemble/testx42.xlsx", 'input3')
# data_writer(test_x4_3.reshape(np.shape(test_x)[0], -1), "./data/ensemble/testx43.xlsx", 'input4')

g1 = tf.Graph()
with g1.as_default():
    batch_size = 64
    learningrate = 0.001
    x_input_width = 8
    y_input_width = 8
    z_input_width = 8
    m_input_width = 8
    state_width = 128
    output_width = 3
    output_width1 = 32
    # train_size=300
    # verification_size=100
    # test_size=100
    traintimes = 100
    space_weight = 0.2
    time_weight = 0.2

    # 1first layer
    lstm11 = Lstm1(z_input_width, state_width, batch_size)

    # 1second layer
    lstm21 = Lstm2(y_input_width, state_width, batch_size, time_weight, space_weight)

    # 1third layer
    lstm31 = Lstm2(m_input_width, state_width, batch_size, time_weight, space_weight)

    # forth layer
    lstm41 = Lstm31(x_input_width, state_width, batch_size, time_weight, space_weight)

    # 2third layer
    lstm51 = Lstm2(m_input_width, state_width, batch_size, time_weight, space_weight)

    # 2second layer
    lstm61 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)

    # 2first layer
    lstm71 = Lstm1(y_input_width, state_width, batch_size)

    # 3third layer
    lstm81 = Lstm2(m_input_width, state_width, batch_size, time_weight, space_weight)
    # 3second layer
    lstm91 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 3first layer
    lstm101 = Lstm1(y_input_width, state_width, batch_size)

    # 4third layer
    lstm111 = Lstm2(m_input_width, state_width, batch_size, time_weight, space_weight)
    # 4second layer
    lstm121 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 4first layer
    lstm131 = Lstm1(y_input_width, state_width, batch_size)

    # 5third layer
    lstm141 = Lstm2(m_input_width, state_width, batch_size, time_weight, space_weight)
    # 5second layer
    lstm151 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 5first layer
    lstm161 = Lstm1(y_input_width, state_width, batch_size)

    # 6third layer
    lstm171 = Lstm2(m_input_width, state_width, batch_size, time_weight, space_weight)
    # 6second layer
    lstm181 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 6first layer
    lstm191 = Lstm1(y_input_width, state_width, batch_size)

    # MLP
    mlp1 = mlp(output_width, state_width, output_width1, batch_size)

    # print(s==tf.Tensor(1))

    x1 = tf.placeholder(tf.float32, [batch_size, x_len, x_input_width])
    y1 = tf.placeholder(tf.float32, [batch_size, x_len, y_input_width])
    m1 = tf.placeholder(tf.float32, [batch_size, x_len, m_input_width])
    z1 = tf.placeholder(tf.float32, [batch_size, x_len, z_input_width])
    control1 = tf.placeholder(tf.float32, [batch_size, 6, state_width])
    label1 = tf.placeholder(tf.float32, [batch_size, 3])

    h11_c1 = lstm11.forward(y1)

    h21_c1 = lstm21.forward(h11_c1, z1)

    h31_c1 = lstm31.forward(h21_c1, m1)
    h71_c1 = lstm71.forward(y1)

    h61_c1 = lstm61.forward(h71_c1, m1)
    h51_c1 = lstm51.forward(h61_c1, z1)

    h101_c1 = lstm101.forward(z1)
    h91_c1 = lstm91.forward(h101_c1, y1)
    h81_c1 = lstm81.forward(h91_c1, m1)

    h131_c1 = lstm131.forward(z1)
    h121_c1 = lstm121.forward(h131_c1, m1)
    h111_c1 = lstm111.forward(h121_c1, y1)

    h161_c1 = lstm161.forward(m1)
    h151_c1 = lstm151.forward(h161_c1, y1)
    h141_c1 = lstm141.forward(h151_c1, z1)

    h191_c1 = lstm191.forward(m1)
    h181_c1 = lstm181.forward(h161_c1, z1)
    h171_c1 = lstm171.forward(h151_c1, y1)
    # h42=lstm42.forward(h52,h41)
    # h43=lstm43.forward(h53,h42)
    h41_c1, onet = lstm41.forward(h31_c1, h51_c1, h81_c1, h111_c1, h141_c1, h171_c1, x1, control1)
    # h32=lstm32.forward(h42,h22,h31)
    # h33=lstm33.forward(h43,h23,h32)
    z1_c1 = mlp1.forward(h41_c1)

    loss = tf.reduce_mean(tf.square(label1 - z1_c1))
    train_op1 = tf.train.AdamOptimizer(learningrate).minimize(loss)
    init_op1 = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())

# with tf.Session(graph=g1) as sess:
#     # saver = tf.train.Saver()
#     # saver = tf.train.import_meta_graph('./model_checkpoint/MyModel.meta')
#     saver.restore(sess, "./model_checkpoint/condition1_wc_0.2_wt0.2/c1")
#     # graph = tf.get_default_graph()
#     # graphdef = graph.as_graph_def()
#     # summary_write = tf.summary.FileWriter("./graph", graph)
#     # summary_write.close()
#     # input_x1 = graph.get_operation_by_name("Placeholder:0")
#     # input_x2 = graph.get_operation_by_name("Placeholder_1:0")
#     # label1 = graph.get_operation_by_name("Placeholder_2:0")
#     # label2 = graph.get_operation_by_name("Placeholder_3:0")
#     # input_x1 = tf.placeholder(tf.float32, [batch_size, x_len, 8])
#     # input_x2 = tf.placeholder(tf.float32, [batch_size, x_len, 8])
#     # z1 =graph.get_operation_by_name("add_781:0")
#     # z2 =graph.get_operation_by_name("add_783:0")

# train_input1 = np.zeros([batch_size, 3])
# for i in range(np.shape(train_x)[0] // batch_size):
#     trx1 = sess.run(z1_c1, feed_dict={x1: train_x[i * batch_size:(i + 1) * batch_size],
#                                 y1: train_x1_2[i * batch_size:(i + 1) * batch_size],
#                                  z1: train_x1_3[i * batch_size:(i + 1) * batch_size],
#                                  m1: train_x1_4[i * batch_size:(i + 1) * batch_size],
#                                  control1: train_con1[i * batch_size:(i + 1) * batch_size, :, :]})
#     train_input1 = np.vstack((train_input1, trx1))
#
#
# train_input1 = train_input1[batch_size:, :]
# print(np.shape(train_input1))
# # data_writer(train_input1, "./data/ensemble/input1.xlsx", 'input1')
# df = pd.DataFrame(train_input1)
# df.to_csv("./data/ensemble/wc0.2_wt0.2_input1.csv", sep=',')
# val_input1 = np.zeros([batch_size, 3])
# for i in range(np.shape(val_x)[0] // batch_size):
#     vax1 = sess.run(z1_c1, feed_dict={x1: val_x[i * batch_size:(i + 1) * batch_size],
#                                     y1: val_x1_2[i * batch_size:(i + 1) * batch_size],
#                                     z1: val_x1_3[i * batch_size:(i + 1) * batch_size],
#                                     m1: val_x1_4[i*batch_size:(i+1)*batch_size],
#                                  control1: val_con1[i * batch_size:(i + 1) * batch_size, :, :]})
#     val_input1 = np.vstack((val_input1, vax1))
#
# val_input1 = val_input1[batch_size:, :]
# print(np.shape(val_input1))
# # data_writer(val_input1, "./data/ensemble/val_input1.xlsx", 'input1')
# df = pd.DataFrame(val_input1)
# df.to_csv("./data/ensemble/wc0.2_wt0.2_valinput1.csv", sep=',')

with tf.Graph().as_default() as g2:
    batch_size = 32
    learningrate = 0.001
    x_input_width = 8
    y_input_width = 8
    z_input_width = 8
    m_input_width = 8
    state_width = 64
    output_width = 3
    output_width1 = 32
    # train_size=300
    # verification_size=100
    # test_size=100
    traintimes = 100
    space_weight = 0.8
    time_weight = 1

    # 1first layer
    lstm11 = Lstm1(z_input_width, state_width, batch_size)

    # 1second layer
    lstm21 = Lstm2(y_input_width, state_width, batch_size, time_weight, space_weight)

    # 1third layer
    lstm31 = Lstm2(m_input_width, state_width, batch_size, time_weight, space_weight)

    # forth layer
    lstm41 = Lstm31(x_input_width, state_width, batch_size, time_weight, space_weight)

    # 2third layer
    lstm51 = Lstm2(m_input_width, state_width, batch_size, time_weight, space_weight)

    # 2second layer
    lstm61 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)

    # 2first layer
    lstm71 = Lstm1(y_input_width, state_width, batch_size)

    # 3third layer
    lstm81 = Lstm2(m_input_width, state_width, batch_size, time_weight, space_weight)
    # 3second layer
    lstm91 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 3first layer
    lstm101 = Lstm1(y_input_width, state_width, batch_size)

    # 4third layer
    lstm111 = Lstm2(m_input_width, state_width, batch_size, time_weight, space_weight)
    # 4second layer
    lstm121 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 4first layer
    lstm131 = Lstm1(y_input_width, state_width, batch_size)

    # 5third layer
    lstm141 = Lstm2(m_input_width, state_width, batch_size, time_weight, space_weight)
    # 5second layer
    lstm151 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 5first layer
    lstm161 = Lstm1(y_input_width, state_width, batch_size)

    # 6third layer
    lstm171 = Lstm2(m_input_width, state_width, batch_size, time_weight, space_weight)
    # 6second layer
    lstm181 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 6first layer
    lstm191 = Lstm1(y_input_width, state_width, batch_size)

    # MLP
    mlp1 = mlp(output_width, state_width, output_width1, batch_size)

    # print(s==tf.Tensor(1))

    x = tf.placeholder(tf.float32, [batch_size, x_len, x_input_width])
    y = tf.placeholder(tf.float32, [batch_size, x_len, y_input_width])
    m = tf.placeholder(tf.float32, [batch_size, x_len, m_input_width])
    z = tf.placeholder(tf.float32, [batch_size, x_len, z_input_width])
    control = tf.placeholder(tf.float32, [batch_size, 6, state_width])
    label = tf.placeholder(tf.float32, [batch_size, 3])

    h11 = lstm11.forward(y)

    h21 = lstm21.forward(h11, z)

    h31 = lstm31.forward(h21, m)
    h71 = lstm71.forward(y)

    h61 = lstm61.forward(h71, m)
    h51 = lstm51.forward(h61, z)

    h101 = lstm101.forward(z)
    h91 = lstm91.forward(h101, y)
    h81 = lstm81.forward(h91, m)

    h131 = lstm131.forward(z)
    h121 = lstm121.forward(h131, m)
    h111 = lstm111.forward(h121, y)

    h161 = lstm161.forward(m)
    h151 = lstm151.forward(h161, y)
    h141 = lstm141.forward(h151, z)

    h191 = lstm191.forward(m)
    h181 = lstm181.forward(h161, z)
    h171 = lstm171.forward(h151, y)
    # h42=lstm42.forward(h52,h41)
    # h43=lstm43.forward(h53,h42)
    h41, onet = lstm41.forward(h31, h51, h81, h111, h141, h171, x, control)
    # h32=lstm32.forward(h42,h22,h31)
    # h33=lstm33.forward(h43,h23,h32)
    pre = mlp1.forward(h41)
    loss = tf.reduce_mean(tf.square(label - pre))
    train_op = tf.train.AdamOptimizer(learningrate).minimize(loss)
    init_op = tf.global_variables_initializer()
    saver1 = tf.train.Saver(tf.global_variables())
# batch_size=32
# with tf.Session(graph=g1) as sess:
# saver = tf.train.Saver()
# saver1.restore(sess, "./model_checkpoint/condition2/c2")
# train_input2 = np.zeros([batch_size, 3])
# for i in range(np.shape(train_x)[0] // batch_size):
#     x2 = sess.run(pre, feed_dict={x: train_x[i * batch_size:(i + 1) * batch_size],
#                                             y: train_x2_1[i * batch_size:(i + 1) * batch_size],
#                                              z: train_x2_3[i * batch_size:(i + 1) * batch_size],
#                                              m: train_x2_4[i*batch_size:(i+1)*batch_size],
#                                              control: train_con2[i*batch_size:(i+1)*batch_size]})
#
#     train_input2 = np.vstack((train_input2, x2))
#
# train_input2 = train_input2[batch_size:, :]
#
# print(np.shape(train_input2))
#
#     val_input2 = np.zeros([batch_size, 3])
#     for i in range(np.shape(val_x)[0] // batch_size):
#         x2 = sess.run(pre, feed_dict={x: val_x[i * batch_size:(i + 1) * batch_size],
#                                                 y: val_x2_1[i * batch_size:(i + 1) * batch_size],
#                                                  z: val_x2_3[i * batch_size:(i + 1) * batch_size],
#                                                  m: val_x2_4[i*batch_size:(i+1)*batch_size],
#                                                 control: val_con2[i*batch_size:(i+1)*batch_size]})
#         val_input2 = np.vstack((val_input2, x2))
#
#     val_input2 = val_input2[batch_size:, :]
#     print(np.shape(val_input2))
#
#     saver.restore(sess, "./model_checkpoint/condition3/c3")
#     train_input3 = np.zeros([batch_size, 3])
#     for i in range(np.shape(train_x)[0] // batch_size):
#         x3 = sess.run(pre, feed_dict={x: train_x[i * batch_size:(i + 1) * batch_size],
#                                      y: train_x3_1[i * batch_size:(i + 1) * batch_size],
#                                      z: train_x3_2[i * batch_size:(i + 1) * batch_size],
#                                      m: train_x3_4[i * batch_size:(i + 1) * batch_size],
#                                      control: train_con3[i*batch_size:(i+1)*batch_size]})
#
#         train_input3 = np.vstack((train_input3, x3))
#
#     train_input3 = train_input3[batch_size:, :]
#
#     print(np.shape(train_input3))
#
#     val_input3 = np.zeros([batch_size, 3])
#     for i in range(np.shape(val_x)[0] // batch_size):
#         x3 = sess.run(pre, feed_dict={x: val_x[i * batch_size:(i + 1) * batch_size],
#                                      y: val_x3_1[i * batch_size:(i + 1) * batch_size],
#                                      z: val_x3_2[i * batch_size:(i + 1) * batch_size],
#                                      m: val_x3_4[i * batch_size:(i + 1) * batch_size],
#                                      control: val_con3[i*batch_size:(i+1)*batch_size]})
#         val_input3 = np.vstack((val_input3, x3))
#
#     val_input3 = val_input3[batch_size:, :]
#     print(np.shape(val_input3))
#
# saver.restore(sess, "./model_checkpoint/condition4_c1_0.01_0.01/c4")
# batch_size = 64
# train_input4 = np.zeros([batch_size, 3])
# for i in range(np.shape(train_x)[0] // batch_size):
#     x4 = sess.run(z1_c1, feed_dict={x1: train_x[i * batch_size:(i + 1) * batch_size],
#                                  y1: train_x4_1[i * batch_size:(i + 1) * batch_size],
#                                  z1: train_x4_2[i * batch_size:(i + 1) * batch_size],
#                                  m1: train_x4_3[i * batch_size:(i + 1) * batch_size],
#                                  control1: train_con4[i*batch_size:(i+1)*batch_size]})
#
#     train_input4 = np.vstack((train_input4, x4))
#
# train_input4 = train_input4[batch_size:, :]
#
# print(np.shape(train_input4))
#
# val_input4 = np.zeros([batch_size, 3])
# for i in range(np.shape(val_x)[0] // batch_size):
#     x4 = sess.run(z1_c1, feed_dict={x1: val_x[i * batch_size:(i + 1) * batch_size],
#                                  y1: val_x4_1[i * batch_size:(i + 1) * batch_size],
#                                  z1: val_x4_2[i * batch_size:(i + 1) * batch_size],
#                                  m1: val_x4_3[i * batch_size:(i + 1) * batch_size],
#                                  control1: val_con4[i*batch_size:(i+1)*batch_size]})
#     val_input4 = np.vstack((val_input4, x4))
#
# val_input4 = val_input4[batch_size:, :]
# print(np.shape(val_input4))

# saver = tf.train.import_meta_graph('./model_checkpoint/quanju/LSTM.meta')
# saver.restore(sess, "./model_checkpoint/quanju/LSTM")
#
# graph = tf.get_default_graph()
# input_x1 = graph.get_operation_by_name("xs:0")
# label1 = graph.get_operation_by_name("ys:0")
# z1 = graph.get_operation_by_name("op_to_restore:0")


model5 = keras.models.load_model("G:/researchproject/DCGNet/model_mout_去压力.h5")

train_input5 = model5.predict(train_x)
val_input5 = model5.predict(val_x)

train_input1 = pd.read_csv('./data/ensemble/wc0.2_wt0.2_input1.csv').values
train_input2 = pd.read_csv('./data/ensemble/input2.csv').values
train_input3 = pd.read_csv('./data/ensemble/input3.csv').values
train_input4 = pd.read_csv('./data/ensemble/c1_0.01_0.01input4.csv').values
val_input1 = pd.read_csv('./data/ensemble/wc0.2_wt0.2_valinput1.csv').values
val_input2 = pd.read_csv('./data/ensemble/valinput2.csv').values
val_input3 = pd.read_csv('./data/ensemble/valinput3.csv').values
val_input4 = pd.read_csv('./data/ensemble/c1_0.01_0.01valinput4.csv').values

# train_input5 = train_input5[y3_index[0]-x_len:]
# val_input5 = val_input5[y3_index[0]-x_len:]

# data_writer(train_input5, "./input5.xlsx", "input5")
train_input5 = train_input5[:len(train_input1)]
val_input5 = val_input5[:len(val_input1)]
train_input2 = train_input2[:len(train_input1)]
val_input2 = val_input2[:len(val_input1)]
train_input3 = train_input3[:len(train_input1)]
val_input3 = val_input3[:len(val_input1)]
train_input4 = train_input4[:len(train_input1)]
val_input4 = val_input4[:len(val_input1)]
train_y = train_y[:len(train_input1)]
val_y = val_y[:len(val_input1)]
train_y = train_y.reshape(train_y.shape[0], -1, 1)
val_y = val_y.reshape(val_y.shape[0], -1, 1)

# data_writer(train_input1, "./data/ensemble/input1.csv")
# data_writer(train_input2, "./data/ensemble/input2.csv")
# data_writer(train_input3, "./data/ensemble/input3.csv")
# data_writer(train_input4, "./data/ensemble/c1_0.01_0.01input4.csv")
# data_writer(train_input5, "./data/ensemble/input5.csv")
# data_writer(val_input2, "./data/ensemble/valinput2.csv")
# data_writer(val_input3, "./data/ensemble/valinput3.csv")
# data_writer(val_input4, "./data/ensemble/c1_0.01_0.01valinput4.csv")
# data_writer(val_input5, "./data/ensemble/valinput5.csv")

input1 = train_input1[:, 1:].reshape(-1, 3, 1)
input2 = train_input2[:, 1:].reshape(-1, 3, 1)
input3 = train_input3[:, 1:].reshape(-1, 3, 1)
# input4 = train_input4[:, 1:].reshape(-1, 3, 1)
input4 = train_input4[:, 1:].reshape(-1, 3, 1)
input5 = train_input5.reshape(-1, 3, 1)

val_input1 = val_input1[:, 1:].reshape(-1, 3, 1)
val_input2 = val_input2[:, 1:].reshape(-1, 3, 1)
val_input3 = val_input3[:, 1:].reshape(-1, 3, 1)
# val_input4 = val_input4[:, 1:].reshape(-1, 3, 1)
val_input4 = val_input4[:, 1:].reshape(-1, 3, 1)
val_input5 = val_input5.reshape(-1, 3, 1)

Input1 = keras.layers.Input(shape=(3, 1))
Input2 = keras.layers.Input(shape=(3, 1))
Input3 = keras.layers.Input(shape=(3, 1))
Input4 = keras.layers.Input(shape=(3, 1))
Input5 = keras.layers.Input(shape=(3, 1))

input_x = keras.layers.concatenate([Input1, Input2, Input3, Input4, Input5], axis=2)

ex = keras.layers.Dense(50, activation='relu')(input_x)
# x = keras.layers.Dropout(0.2)(x)
ex = keras.layers.Dense(50, activation='relu')(ex)
# x = keras.layers.Dropout(0.2)(x)

# x_weight = keras.layers.Dense(5, activation='softmax')(x)
# x = keras.layers.Multiply()([x_weight, input_x])
# out = keras.layers.Add()([x[:,:,0],x[:,:,1],x[:,:,2],x[:,:,3],x[:,:,4]])
out = keras.layers.Dense(1)(ex)
# out = keras.layers.Reshape((3))(out)
model = keras.models.Model(inputs=[Input1, Input2, Input3, Input4, Input5], outputs=[out])

learning_rate = 0.001
epoch = 500
batch_size = 64
optimizer = optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='mse')
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=2, mode='min')
model_path = 'model_checkpoint/DCGNet'
modelcheckpoint = ModelCheckpoint(model_path + 'NET{}.h5'.format(str(i)), monitor='val_loss', mode='min')
reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, verbose=2)

history = model.fit([input1, input2, input3, input4, input5], train_y,
                    epochs=epoch,
                    batch_size=batch_size,
                    shuffle=False,
                    validation_data=([val_input1, val_input2, val_input3, val_input4, val_input5], val_y),
                    verbose=2,
                    callbacks=[early_stopping])

model.save("./modelh5_check/正则多工况模型集成relulayer1_50_layer2_50_lr1efu3_dp0.h5")

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(range(len(loss)), loss, 'b-', label='train loss')
plt.plot(range(len(loss)), val_loss, 'r-', label='val loss')
plt.xlabel('epoch', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.legend(loc='best', fontsize=15)
plt.show()

# data_writer(loss, "./loss保存/ensemble/layer1_40_layer2_40_lr1efu3/trloss.csv")
# data_writer(val_loss, "./loss保存/ensemble/layer1_40_layer2_40_lr1efu3/valloss.csv")

model = keras.models.load_model("./modelh5_check/正则多工况模型集成relulayer1_50_layer2_50_lr1efu3_dp0.h5")

sess = tf.Session(graph=g1)
sess1 = tf.Session(graph=g2)

Zushu = 50
pred_y_sum = []
for i in range(Zushu):
    test_x_in = np.array(test_x[i * 60]).reshape((-1, x_len, 8))
    pred_y = []
    for j in range(60):
        print("第", j, "次迭代")
        saver.restore(sess, "./model_checkpoint/condition1_wc_0.2_wt0.2/c1")
        tx1 = sess.run(z1_c1, feed_dict={x1: data_fill(test_x_in, 64),
                                         y1: data_fill(test_x1_2[i * 60].reshape((-1, x_len, 8)), 64),
                                         z1: data_fill(test_x1_3[i * 60].reshape((-1, x_len, 8)), 64),
                                         m1: data_fill(test_x1_4[i * 60].reshape((-1, x_len, 8)), 64),
                                         control1: data_fill(test_con1[i * 60, :, :].reshape((1, 6, -1)), 64)})

        # loss = tf.reduce_mean(tf.square(label - z1))
        # train_op = tf.train.AdamOptimizer(learningrate).minimize(loss)
        # init_op = tf.global_variables_initializer()
        # saver = tf.train.Saver(tf.global_variables())
        # with tf.Session(graph=g2) as sess1:
        saver1.restore(sess1, "./model_checkpoint/condition2/c2")
        tx2 = sess1.run(pre, feed_dict={x: data_fill(test_x_in, 32),
                                        y: data_fill(test_x2_1[i * 60].reshape((-1, x_len, 8)), 32),
                                        z: data_fill(test_x2_3[i * 60].reshape((-1, x_len, 8)), 32),
                                        m: data_fill(test_x2_4[i * 60].reshape((-1, x_len, 8)), 32),
                                        control: data_fill(test_con2[i * 60].reshape((1, 6, -1)), 32)})
        saver1.restore(sess1, "./model_checkpoint/condition3/c3")
        tx3 = sess1.run(pre, feed_dict={x: data_fill(test_x_in, 32),
                                        y: data_fill(test_x3_1[i * 60].reshape((-1, x_len, 8)), 32),
                                        z: data_fill(test_x3_2[i * 60].reshape((-1, x_len, 8)), 32),
                                        m: data_fill(test_x3_4[i * 60].reshape((-1, x_len, 8)), 32),
                                        control: data_fill(test_con3[i * 60].reshape((1, 6, -1)), 32)})
        saver.restore(sess, "./model_checkpoint/condition4_c1_0.01_0.01/c4")
        tx4 = sess.run(z1_c1, feed_dict={x1: data_fill(test_x_in, 64),
                                         y1: data_fill(test_x4_1[i * 60].reshape((-1, x_len, 8)), 64),
                                         z1: data_fill(test_x4_2[i * 60].reshape((-1, x_len, 8)), 64),
                                         m1: data_fill(test_x4_3[i * 60].reshape((-1, x_len, 8)), 64),
                                         control1: data_fill(test_con4[i * 60].reshape((1, 6, -1)), 64)})
        # x1 = model1.predict(test_x_in[:, x_len-x_len1:], batch_size=1)
        # x2 = model2.predict(test_x_in[:, x_len-x_len2:], batch_size=1)
        # x3 = (teata_set1[0] * model3_0.predict(test_x_in) + teata_set1[1] * model3_1.predict(test_x_in) + teata_set1[2] * model3_2.predict(test_x_in) + teata_set1[3] * model3_3.predict(test_x_in) + teata_set1[4] * model3_4.predict(test_x_in))/teata_sum1
        # x4 = (teata_set2[0] * model4_0.predict(test_x_in) + teata_set2[1] * model4_1.predict(test_x_in) + teata_set2[2] * model4_2.predict(test_x_in) + teata_set2[3] * model4_3.predict(test_x_in) + teata_set2[4] * model4_4.predict(test_x_in))/teata_sum2
        # x3 = model3.predict(test_x_in[:, x_len - x_len3:], batch_size=1)
        # x4 = model4.predict(test_x_in[:, x_len - x_len4:], batch_size=1)
        tx5 = model5.predict(test_x_in, batch_size=1)

        tx1 = tx1[0].reshape(-1, 3, 1)
        tx2 = tx2[0].reshape(-1, 3, 1)
        tx3 = tx3[0].reshape(-1, 3, 1)
        tx4 = tx4[0].reshape(-1, 3, 1)
        tx5 = tx5.reshape(-1, 3, 1)

        # pre_y = model.predict([tx1, tx2, tx3, tx4, tx5])
        pre_y = (tx1 + tx2 + tx3 + tx4 + tx5) / 5
        test_x_in[:, :-1, :] = test_x_in[:, 1:, :]
        test_x_in[:, -1, -3:] = pre_y[:, :, 0]
        test_x_in[0, -1, :-3] = test_x[i * 60 + j + 1][-1, :-3]
        pred_y.append(pre_y)

    pred_y_sum.append(pred_y)

pred_y_sum = np.array(pred_y_sum).reshape(-1, 60, 3)
pred_y_sum = pred_y_sum * std_value[-3:] + mean_value[-3:]

test_y = test_y.reshape(-1, 3) * std_value[-3:] + mean_value[-3:]

test_y_set = []
for i in range(Zushu):
    test_y_set.append(test_y[i * 60:i * 60 + pred_len])
test_y_set = np.array(test_y_set).reshape(-1, 60, 3)

data_writer(pred_y_sum[:, :, 0], "./结果保存/relu_50_50/1-1柜预测值.csv")
data_writer(pred_y_sum[:, :, 1], "./结果保存/relu_50_50/2-1柜预测值.csv")
data_writer(pred_y_sum[:, :, 2], "./结果保存/relu_50_50/3-1柜预测值.csv")

rmse_record1 = []
mape_record1 = []
rmse_record2 = []
mape_record2 = []
rmse_record3 = []
mape_record3 = []
rmse_record4 = []
mape_record4 = []
mae_record1 = []
mae_record2 = []
mae_record3 = []
mae_record4 = []
for i in range(Zushu):
    a1 = np.sqrt(metrics.mean_squared_error(test_y[i * 60:i * 60 + pred_len, -3], pred_y_sum[i, :, -3]))
    b1 = metrics.mean_absolute_percentage_error(test_y[i * 60:i * 60 + pred_len, -3], pred_y_sum[i, :, -3])
    c1 = metrics.mean_absolute_error(test_y[i * 60:i * 60 + pred_len, -3], pred_y_sum[i, :, -3])
    a2 = np.sqrt(metrics.mean_squared_error(test_y[i * 60:i * 60 + pred_len, -2], pred_y_sum[i, :, -2]))
    b2 = metrics.mean_absolute_percentage_error(test_y[i * 60:i * 60 + pred_len, -2], pred_y_sum[i, :, -2])
    c2 = metrics.mean_absolute_error(test_y[i * 60:i * 60 + pred_len, -2], pred_y_sum[i, :, -2])
    a3 = np.sqrt(metrics.mean_squared_error(test_y[i * 60:i * 60 + pred_len, -1], pred_y_sum[i, :, -1]))
    b3 = metrics.mean_absolute_percentage_error(test_y[i * 60:i * 60 + pred_len, -1], pred_y_sum[i, :, -1])
    c3 = metrics.mean_absolute_error(test_y[i * 60:i * 60 + pred_len, -1], pred_y_sum[i, :, -1])
    a4 = np.sqrt(metrics.mean_squared_error(test_y[i * 60:i * 60 + pred_len, :], pred_y_sum[i, :, :]))
    b4 = metrics.mean_absolute_percentage_error(test_y[i * 60:i * 60 + pred_len, :], pred_y_sum[i, :, :])
    c4 = metrics.mean_absolute_error(test_y[i * 60:i * 60 + pred_len, :], pred_y_sum[i, :, :])
    rmse_record1.append(a1)
    rmse_record2.append(a2)
    rmse_record3.append(a3)
    rmse_record4.append(a4)
    mape_record1.append(b1)
    mape_record2.append(b2)
    mape_record3.append(b3)
    mape_record4.append(b4)
    mae_record1.append(c1)
    mae_record2.append(c2)
    mae_record3.append(c3)
    mae_record4.append(c4)

    y1 = np.array(pred_y_sum)[i, :, -3]
    y2 = np.array(test_y[i * 60:i * 60 + pred_len])[:, -3]
    plt.plot(y1, 'r', label='prediction')
    plt.plot(y2, 'b', label='true')
    plt.legend(loc='upper right', fontsize=20)
    # plt.savefig("./双集成5/1柜/mout_预测值与真实值对比1_{}.png".format(i))
    # plt.savefig("./多工况集成/1柜/mout_预测值与真实值对比1_{}.png".format(i))
    # plt.savefig("./TR_MMD多工况集成/1柜/mout_预测值与真实值对比1_{}.png".format(i))
    plt.show()

    y1 = np.array(pred_y_sum)[i, :, -2]
    y2 = np.array(test_y[i * 60:i * 60 + pred_len])[:, -2]
    plt.plot(y1, 'r', label='prediction')
    plt.plot(y2, 'b', label='true')
    plt.legend(loc='upper right', fontsize=20)
    # plt.savefig("./双集成5/2柜/mout_预测值与真实值对比2_{}.png".format(i))
    # plt.savefig("./多工况集成/2柜/mout_预测值与真实值对比2_{}.png".format(i))
    # plt.savefig("./TR_MMD多工况集成/2柜/mout_预测值与真实值对比2_{}.png".format(i))
    plt.show()

    y1 = np.array(pred_y_sum)[i, :, -1]
    y2 = np.array(test_y[i * 60:i * 60 + pred_len])[:, -1]
    plt.plot(y1, 'r', label='prediction')
    plt.plot(y2, 'b', label='true')
    plt.legend(loc='upper right', fontsize=20)
    # plt.savefig("./双集成5/3柜/mout_预测值与真实值对比3_{}.png".format(i))
    # plt.savefig("./多工况集成/3柜/mout_预测值与真实值对比3_{}.png".format(i))
    # plt.savefig("./TR_MMD多工况集成/3柜/mout_预测值与真实值对比3_{}.png".format(i))
    plt.show()

print(np.mean(rmse_record1))
print(np.mean(mape_record1))
print(np.mean(mae_record1))
print(np.mean(rmse_record2))
print(np.mean(mape_record2))
print(np.mean(mae_record2))
print(np.mean(rmse_record3))
print(np.mean(mape_record3))
print(np.mean(mae_record3))
