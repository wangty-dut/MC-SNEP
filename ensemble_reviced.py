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
                   usecols=[1, 2, 3, 4, 6, 10, 11, 12]).values # 生产数据

rawdata = df.astype('float32')

state_data = pd.read_excel('汇总.xlsx', header=0, usecols=[16]).values  #工况数据
print(np.shape(rawdata))


def form_aux(y1_index, y1_index1, y2_index, y3_index, y4_index):
    train_x1_2 = []
    train_x1_3 = []
    train_x1_4 = []
    train_x1_1 = []
    shunxu = []
    for index in y1_index:
        i2 = 0
        i3 = 0
        i4 = 0
        i1 = 0
        m = index-1
        while m >= 0:
            if (m in y2_index):
                i2 = m
                train_x1_2.append(rawdata[max(0, m - x_len): m])
                break
            m -= 1
        m = index-1
        while m >= 0:
            if (m in y3_index):
                i3 = m
                train_x1_3.append(rawdata[max(0, m - x_len): m])
                break
            m -= 1
        m = index-1
        while m >= 0:
            if (m in y4_index):
                i4 = m
                train_x1_4.append(rawdata[max(0, m - x_len): m])
                break
            m -= 1
        m = max(i2, i3, i4)
        while m >= 0:
            if (m in y1_index1):
                i1 = m
                train_x1_1.append(rawdata[max(0, m - x_len): m])
                break
            m -= 1

        if (i1 != 0 and i2 != 0 and i3 != 0 and i4 != 0):
            if (max(i1, i2, i3, i4)==i3 and max(i1, i2, i4)==i2):
                shunxu.append(0)
            elif (max(i1, i2, i3, i4)==i2 and max(i1, i3, i4)==i3):
                shunxu.append(1)
            elif (max(i1, i2, i3, i4)==i4 and max(i1, i2, i3)==i2):
                shunxu.append(2)
            elif (max(i1, i2, i3, i4)==i2 and max(i1, i3, i4)==i4):
                shunxu.append(3)
            elif (max(i1, i2, i3, i4)==i4 and max(i1, i2, i3)==i3):
                shunxu.append(4)
            elif (max(i1, i2, i3, i4)==i3 and max(i1, i2, i4)==i4):
                shunxu.append(5)
            elif (max(i1, i2, i3, i4)==i2 and max(i1, i3, i4)==i1):
                shunxu.append(6)
            elif (max(i1, i2, i3, i4)==i3 and max(i1, i2, i4)==i1):
                shunxu.append(7)
            elif (max(i1, i2, i3, i4)==i4 and max(i1, i2, i3)==i1):
                shunxu.append(8)
    return train_x1_2, train_x1_3, train_x1_4, train_x1_1, shunxu


def form_con(shunxu, state_width):
    con = np.zeros((len(shunxu), 9, state_width))
    for i in range(len(shunxu)):
        if shunxu[i] == 0:
            t = np.zeros((9, state_width))
            t[0:1, :] = np.ones((1, state_width))
            con[i] = t
        elif shunxu[i] == 1:
            t = np.zeros((9, state_width))
            t[1:2, :] = np.ones((1, state_width))
            con[i] = t
        elif shunxu[i] == 2:
            t = np.zeros((9, state_width))
            t[2:3, :] = np.ones((1, state_width))
            con[i] = t
        elif shunxu[i] == 3:
            t = np.zeros((9, state_width))
            t[3:4, :] = np.ones((1, state_width))
            con[i] = t
        elif shunxu[i] == 4:
            t = np.zeros((9, state_width))
            t[4:5, :] = np.ones((1, state_width))
            con[i] = t
        elif shunxu[i] == 5:
            t = np.zeros((9, state_width))
            t[5:6, :] = np.ones((1, state_width))
            con[i] = t
        elif shunxu[i] == 6:
            t = np.zeros((9, state_width))
            t[6:7, :] = np.ones((1, state_width))
            con[i] = t
        elif shunxu[i] == 7:
            t = np.zeros((9, state_width))
            t[7:8, :] = np.ones((1, state_width))
            con[i] = t
        elif shunxu[i] == 8:
            t = np.zeros((9, state_width))
            t[8:, :] = np.ones((1, state_width))
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

# train_x1_2, train_x1_3, train_x1_4, train_x1_1, shunxu1 = form_aux(np.arange(y3_index[0], train_size+x_len), y1_index, y2_index, y3_index, y4_index)
# print(train_x1_4)
# train_x2_1, train_x2_3, train_x2_4, train_x2_2, shunxu2 = form_aux(np.arange(y3_index[0], train_size+x_len), y2_index, y1_index, y3_index, y4_index)
# train_x3_1, train_x3_2, train_x3_4, train_x3_3, shunxu3 = form_aux(np.arange(y3_index[0], train_size+x_len), y3_index, y1_index, y2_index, y4_index)
# train_x4_1, train_x4_2, train_x4_3, train_x4_4, shunxu4 = form_aux(np.arange(y3_index[0], train_size+x_len), y4_index, y1_index, y2_index, y3_index)
#
#
# train_x = []
train_y = []
for j in range(train_size):
    # train_x.append(train_data[j:j + 1 * x_len, :])
    train_y.append(train_data[j + 1 * x_len, -3:])
#
# train_x = np.array(train_x)
train_y = np.array(train_y)
# shape1 = min(np.shape(train_x1_2)[0], np.shape(train_x1_3)[0], np.shape(train_x1_4)[0], np.shape(train_x1_1)[0])
# shape2 = min(np.shape(train_x2_1)[0], np.shape(train_x2_3)[0], np.shape(train_x2_4)[0], np.shape(train_x2_2)[0])
# shape3 = min(np.shape(train_x3_1)[0], np.shape(train_x3_2)[0], np.shape(train_x3_4)[0], np.shape(train_x3_3)[0])
# shape4 = min(np.shape(train_x4_2)[0], np.shape(train_x4_3)[0], np.shape(train_x4_1)[0], np.shape(train_x4_4)[0])
# shape = min(shape1, shape2, shape3, shape4)
# train_x = train_x[np.shape(train_x)[0]-shape:]
# train_x1_2 = train_x1_2[np.shape(train_x1_2)[0]-shape:]
# train_x1_3 = train_x1_3[np.shape(train_x1_3)[0]-shape:]
# train_x1_4 = train_x1_4[np.shape(train_x1_4)[0]-shape:]
# train_x1_1 = train_x1_1[np.shape(train_x1_1)[0]-shape:]
# # train_x2 = train_x[np.shape(train_x)[0]-shape]
# train_x2_1 = train_x2_1[np.shape(train_x2_1)[0]-shape:]
# train_x2_3 = train_x2_3[np.shape(train_x2_3)[0]-shape:]
# train_x2_4 = train_x2_4[np.shape(train_x2_4)[0]-shape:]
# train_x2_2 = train_x2_2[np.shape(train_x2_2)[0]-shape:]
# # train_x3 = train_x[np.shape(train_x)[0]-shape]
# train_x3_1 = train_x3_1[np.shape(train_x3_1)[0]-shape:]
# train_x3_2 = train_x3_2[np.shape(train_x3_2)[0]-shape:]
# train_x3_4 = train_x3_4[np.shape(train_x3_4)[0]-shape:]
# train_x3_3 = train_x3_3[np.shape(train_x3_3)[0]-shape:]
# # train_x4 = train_x[np.shape(train_x)[0]-shape]
# train_x4_1 = train_x4_1[np.shape(train_x4_1)[0]-shape:]
# train_x4_2 = train_x4_2[np.shape(train_x4_2)[0]-shape:]
# train_x4_3 = train_x4_3[np.shape(train_x4_3)[0]-shape:]
# train_x4_4 = train_x4_4[np.shape(train_x4_4)[0]-shape:]
#
# shunxu1 = shunxu1[np.shape(shunxu1)[0]-shape:]
# shunxu2 = shunxu2[np.shape(shunxu2)[0]-shape:]
# shunxu3 = shunxu3[np.shape(shunxu3)[0]-shape:]
# shunxu4 = shunxu4[np.shape(shunxu4)[0]-shape:]
#
# train_x1_2 = np.array(train_x1_2)
# train_x1_3 = np.array(train_x1_3)
# train_x1_4 = np.array(train_x1_4)
# train_x1_1 = np.array(train_x1_1)
# train_x2_1 = np.array(train_x2_1)
# train_x2_3 = np.array(train_x2_3)
# train_x2_4 = np.array(train_x2_4)
# train_x2_2 = np.array(train_x2_2)
# train_x3_1 = np.array(train_x3_1)
# train_x3_2 = np.array(train_x3_2)
# train_x3_4 = np.array(train_x3_4)
# train_x3_3 = np.array(train_x3_3)
# train_x4_1 = np.array(train_x4_1)
# train_x4_2 = np.array(train_x4_2)
# train_x4_3 = np.array(train_x4_3)
# train_x4_4 = np.array(train_x4_4)
#
# print(np.shape(train_x))
# print(np.shape(train_x1_2))
# print(np.shape(train_x2_1))
#
# train_con1 = form_con(shunxu1, 64)
# train_con2 = form_con(shunxu2, 64)
# train_con3 = form_con(shunxu3, 64)
# train_con4 = form_con(shunxu4, 64)
# #
# data_writer(train_x.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx.csv")
# data_writer(train_x1_2.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx12.csv")
# data_writer(train_x1_3.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx13.csv")
# data_writer(train_x1_4.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx11.csv")
# data_writer(train_x1_1.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx14.csv")
# data_writer(train_x2_1.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx21.csv")
# data_writer(train_x2_3.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx23.csv")
# data_writer(train_x2_4.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx24.csv")
# data_writer(train_x2_2.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx22.csv")
# data_writer(train_x3_1.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx31.csv")
# data_writer(train_x3_2.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx32.csv")
# data_writer(train_x3_4.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx34.csv")
# data_writer(train_x3_3.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx33.csv")
# data_writer(train_x4_1.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx41.csv")
# data_writer(train_x4_2.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx42.csv")
# data_writer(train_x4_3.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx43.csv")
# data_writer(train_x4_4.reshape(np.shape(train_x)[0], -1), "./data/ensemble_reviced/trx44.csv")
# # data_writer(train_input5, "./data/ensemble/input5.xlsx", 'input5')
#
# data_writer(train_con1.reshape(np.shape(train_con1)[0], -1), "./data/ensemble_reviced/traincon1.csv")
# data_writer(train_con2.reshape(np.shape(train_con2)[0], -1), "./data/ensemble_reviced/traincon2.csv")
# data_writer(train_con3.reshape(np.shape(train_con3)[0], -1), "./data/ensemble_reviced/traincon3.csv")
# data_writer(train_con4.reshape(np.shape(train_con4)[0], -1), "./data/ensemble_reviced/traincon4.csv")

train_x = pd.read_csv('./data/ensemble_reviced/trx.csv').values
train_x1_2 = pd.read_csv('./data/ensemble_reviced/trx12.csv').values
train_x1_3 = pd.read_csv('./data/ensemble_reviced/trx13.csv').values
train_x1_4 = pd.read_csv('./data/ensemble_reviced/trx14.csv').values
train_x1_1 = pd.read_csv('./data/ensemble_reviced/trx11.csv').values
train_x2_1 = pd.read_csv('./data/ensemble_reviced/trx21.csv').values
train_x2_3 = pd.read_csv('./data/ensemble_reviced/trx23.csv').values
train_x2_4 = pd.read_csv('./data/ensemble_reviced/trx24.csv').values
train_x2_2 = pd.read_csv('./data/ensemble_reviced/trx22.csv').values
train_x3_1 = pd.read_csv('./data/ensemble_reviced/trx31.csv').values
train_x3_2 = pd.read_csv('./data/ensemble_reviced/trx32.csv').values
train_x3_4 = pd.read_csv('./data/ensemble_reviced/trx34.csv').values
train_x3_3 = pd.read_csv('./data/ensemble_reviced/trx33.csv').values
train_x4_1 = pd.read_csv('./data/ensemble_reviced/trx41.csv').values
train_x4_2 = pd.read_csv('./data/ensemble_reviced/trx42.csv').values
train_x4_3 = pd.read_csv('./data/ensemble_reviced/trx43.csv').values
train_x4_4 = pd.read_csv('./data/ensemble_reviced/trx44.csv').values
train_con1 = pd.read_csv('./data/ensemble_reviced/traincon1.csv').values
train_con2 = pd.read_csv('./data/ensemble_reviced/traincon2.csv').values
train_con3 = pd.read_csv('./data/ensemble_reviced/traincon3.csv').values
train_con4 = pd.read_csv('./data/ensemble_reviced/traincon4.csv').values

train_x = train_x[:, 1:].reshape((-1, x_len, 8))
train_x1_2 = train_x1_2[:, 1:].reshape((-1, x_len, 8))
train_x1_3 = train_x1_3[:, 1:].reshape((-1, x_len, 8))
train_x1_4 = train_x1_4[:, 1:].reshape((-1, x_len, 8))
train_x1_1 = train_x1_1[:, 1:].reshape((-1, x_len, 8))
train_x2_1 = train_x2_1[:, 1:].reshape((-1, x_len, 8))
train_x2_3 = train_x2_3[:, 1:].reshape((-1, x_len, 8))
train_x2_4 = train_x2_4[:, 1:].reshape((-1, x_len, 8))
train_x2_2 = train_x2_2[:, 1:].reshape((-1, x_len, 8))
train_x3_1 = train_x3_1[:, 1:].reshape((-1, x_len, 8))
train_x3_2 = train_x3_2[:, 1:].reshape((-1, x_len, 8))
train_x3_4 = train_x3_4[:, 1:].reshape((-1, x_len, 8))
train_x3_3 = train_x3_3[:, 1:].reshape((-1, x_len, 8))
train_x4_1 = train_x4_1[:, 1:].reshape((-1, x_len, 8))
train_x4_2 = train_x4_2[:, 1:].reshape((-1, x_len, 8))
train_x4_3 = train_x4_3[:, 1:].reshape((-1, x_len, 8))
train_x4_4 = train_x4_4[:, 1:].reshape((-1, x_len, 8))
print(train_con1)
train_con1 = train_con1[:, 1:].reshape((np.shape(train_con1)[0], 9, -1))
train_con2 = train_con2[:, 1:].reshape((np.shape(train_con2)[0], 9, -1))
train_con3 = train_con3[:, 1:].reshape((np.shape(train_con3)[0], 9, -1))
train_con4 = train_con4[:, 1:].reshape((np.shape(train_con4)[0], 9, -1))
# train_con4 = np.concatenate((train_con4, train_con4), axis=2)
train_y = train_y[np.shape(train_y)[0] - np.shape(train_x)[0]:]
print(np.shape(train_con4))
print(np.shape(train_con1))
print(np.shape(train_con2))
print(np.shape(train_con3))

# val_x = []
val_y = []

# val_x1_2, val_x1_3, val_x1_4, val_x1_1, shunxu1 = form_aux(np.arange(train_size+x_len, train_size+val_size+x_len), y1_index, y2_index, y3_index, y4_index)
# val_x2_1, val_x2_3, val_x2_4, val_x2_2, shunxu2 = form_aux(np.arange(train_size+x_len, train_size+val_size+x_len), y2_index, y1_index, y3_index, y4_index)
# val_x3_1, val_x3_2, val_x3_4, val_x3_3, shunxu3 = form_aux(np.arange(train_size+x_len, train_size+val_size+x_len), y3_index, y1_index, y2_index, y4_index)
# val_x4_1, val_x4_2, val_x4_3, val_x4_4, shunxu4 = form_aux(np.arange(train_size+x_len, train_size+val_size+x_len), y4_index, y1_index, y2_index, y3_index)
#
# print(val_size)
# print(train_size)
for i in range(val_size):
    # val_x.append(train_data[train_size + i: train_size + i + x_len])
    val_y.append(train_data[train_size + i + x_len, -3:])  #:i+4*x_len])

val_y = np.array(val_y)
# shape1 = min(np.shape(val_x1_2)[0], np.shape(val_x1_3)[0], np.shape(val_x1_4)[0], np.shape(val_x1_1)[0])
# shape2 = min(np.shape(val_x2_1)[0], np.shape(val_x2_3)[0], np.shape(val_x2_4)[0], np.shape(val_x2_2)[0])
# shape3 = min(np.shape(val_x3_1)[0], np.shape(val_x3_2)[0], np.shape(val_x3_4)[0], np.shape(val_x3_3)[0])
# shape4 = min(np.shape(val_x4_2)[0], np.shape(val_x4_3)[0], np.shape(val_x4_1)[0], np.shape(val_x4_4)[0])
# shape = min(shape1, shape2, shape3, shape4)
# val_x = val_x[np.shape(val_x)[0]-shape:]
# val_x1_2 = val_x1_2[np.shape(val_x1_2)[0]-shape:]
# val_x1_3 = val_x1_3[np.shape(val_x1_3)[0]-shape:]
# val_x1_4 = val_x1_4[np.shape(val_x1_4)[0]-shape:]
# val_x1_1 = val_x1_1[np.shape(val_x1_1)[0]-shape:]
# # val_x2 = val_x[np.shape(val_x)[0]-shape]
# val_x2_1 = val_x2_1[np.shape(val_x2_1)[0]-shape:]
# val_x2_3 = val_x2_3[np.shape(val_x2_3)[0]-shape:]
# val_x2_4 = val_x2_4[np.shape(val_x2_4)[0]-shape:]
# val_x2_2 = val_x2_2[np.shape(val_x2_2)[0]-shape:]
# # val_x3 = val_x[np.shape(val_x)[0]-shape]
# val_x3_1 = val_x3_1[np.shape(val_x3_1)[0]-shape:]
# val_x3_2 = val_x3_2[np.shape(val_x3_2)[0]-shape:]
# val_x3_4 = val_x3_4[np.shape(val_x3_4)[0]-shape:]
# val_x3_3 = val_x3_3[np.shape(val_x3_3)[0]-shape:]
# # val_x4 = val_x[np.shape(val_x)[0]-shape]
# val_x4_1 = val_x4_1[np.shape(val_x4_1)[0]-shape:]
# val_x4_2 = val_x4_2[np.shape(val_x4_2)[0]-shape:]
# val_x4_3 = val_x4_3[np.shape(val_x4_3)[0]-shape:]
# val_x4_4 = val_x4_4[np.shape(val_x4_4)[0]-shape:]
#
# shunxu1 = shunxu1[np.shape(shunxu1)[0]-shape:]
# shunxu2 = shunxu2[np.shape(shunxu2)[0]-shape:]
# shunxu3 = shunxu3[np.shape(shunxu3)[0]-shape:]
# shunxu4 = shunxu4[np.shape(shunxu4)[0]-shape:]

val_x = pd.read_csv('./data/ensemble_reviced/valx.csv').values
# print("val-x")
# print(val_x)
val_x1_2 = pd.read_csv('./data/ensemble_reviced/valx12.csv').values
val_x1_3 = pd.read_csv('./data/ensemble_reviced/valx13.csv').values
val_x1_4 = pd.read_csv('./data/ensemble_reviced/valx14.csv').values
val_x1_1 = pd.read_csv('./data/ensemble_reviced/valx11.csv').values
val_x2_1 = pd.read_csv('./data/ensemble_reviced/valx21.csv').values
val_x2_3 = pd.read_csv('./data/ensemble_reviced/valx23.csv').values
val_x2_4 = pd.read_csv('./data/ensemble_reviced/valx24.csv').values
val_x2_2 = pd.read_csv('./data/ensemble_reviced/valx22.csv').values
val_x3_1 = pd.read_csv('./data/ensemble_reviced/valx31.csv').values
val_x3_2 = pd.read_csv('./data/ensemble_reviced/valx32.csv').values
val_x3_4 = pd.read_csv('./data/ensemble_reviced/valx34.csv').values
val_x3_3 = pd.read_csv('./data/ensemble_reviced/valx33.csv').values
val_x4_1 = pd.read_csv('./data/ensemble_reviced/valx41.csv').values
val_x4_2 = pd.read_csv('./data/ensemble_reviced/valx42.csv').values
val_x4_3 = pd.read_csv('./data/ensemble_reviced/valx43.csv').values
val_x4_4 = pd.read_csv('./data/ensemble_reviced/valx44.csv').values
val_con1 = pd.read_csv('./data/ensemble_reviced/valcon1.csv').values
val_con2 = pd.read_csv('./data/ensemble_reviced/valcon2.csv').values
val_con3 = pd.read_csv('./data/ensemble_reviced/valcon3.csv').values
val_con4 = pd.read_csv('./data/ensemble_reviced/valcon4.csv').values

val_x = val_x[:, 1:].reshape((-1, x_len, 8))
val_x1_2 = val_x1_2[:, 1:].reshape((-1, x_len, 8))
val_x1_3 = val_x1_3[:, 1:].reshape((-1, x_len, 8))
val_x1_4 = val_x1_4[:, 1:].reshape((-1, x_len, 8))
val_x1_1 = val_x1_1[:, 1:].reshape((-1, x_len, 8))
val_x2_1 = val_x2_1[:, 1:].reshape((-1, x_len, 8))
val_x2_3 = val_x2_3[:, 1:].reshape((-1, x_len, 8))
val_x2_4 = val_x2_4[:, 1:].reshape((-1, x_len, 8))
val_x2_2 = val_x2_2[:, 1:].reshape((-1, x_len, 8))
val_x3_1 = val_x3_1[:, 1:].reshape((-1, x_len, 8))
val_x3_2 = val_x3_2[:, 1:].reshape((-1, x_len, 8))
val_x3_4 = val_x3_4[:, 1:].reshape((-1, x_len, 8))
val_x3_3 = val_x3_3[:, 1:].reshape((-1, x_len, 8))
val_x4_1 = val_x4_1[:, 1:].reshape((-1, x_len, 8))
val_x4_2 = val_x4_2[:, 1:].reshape((-1, x_len, 8))
val_x4_3 = val_x4_3[:, 1:].reshape((-1, x_len, 8))
val_x4_4 = val_x4_4[:, 1:].reshape((-1, x_len, 8))
val_con1 = val_con1[:, 1:].reshape((np.shape(val_con1)[0], 9, -1))
val_con2 = val_con2[:, 1:].reshape((np.shape(val_con2)[0], 9, -1))
val_con3 = val_con3[:, 1:].reshape((np.shape(val_con3)[0], 9, -1))
val_con4 = val_con4[:, 1:].reshape((np.shape(val_con4)[0], 9, -1))
# val_con4 = np.concatenate((val_con4, val_con4), axis=2)
val_y = val_y[np.shape(val_y)[0] - np.shape(val_x)[0]:]

# print(np.shape(val_x))
# print(np.shape(val_x1_2))
# print(np.shape(val_x2_1))
#
# val_x = np.array(val_x)
# val_x1_2 = np.array(val_x1_2)
# val_x1_3 = np.array(val_x1_3)
# val_x1_4 = np.array(val_x1_4)
# val_x1_1 = np.array(val_x1_1)
# val_x2_1 = np.array(val_x2_1)
# val_x2_3 = np.array(val_x2_3)
# val_x2_4 = np.array(val_x2_4)
# val_x2_2 = np.array(val_x2_2)
# val_x3_1 = np.array(val_x3_1)
# val_x3_2 = np.array(val_x3_2)
# val_x3_4 = np.array(val_x3_4)
# val_x3_3 = np.array(val_x3_3)
# val_x4_1 = np.array(val_x4_1)
# val_x4_2 = np.array(val_x4_2)
# val_x4_3 = np.array(val_x4_3)
# val_x4_4 = np.array(val_x4_4)
#
# val_con1 = form_con(shunxu1, 64)
# val_con2 = form_con(shunxu2, 64)
# val_con3 = form_con(shunxu3, 64)
# val_con4 = form_con(shunxu4, 64)
#
# data_writer(val_con1.reshape(np.shape(val_con1)[0], -1), "./data/ensemble_reviced/valcon1.csv")
# data_writer(val_con2.reshape(np.shape(val_con2)[0], -1), "./data/ensemble_reviced/valcon2.csv")
# data_writer(val_con3.reshape(np.shape(val_con3)[0], -1), "./data/ensemble_reviced/valcon3.csv")
# data_writer(val_con4.reshape(np.shape(val_con4)[0], -1), "./data/ensemble_reviced/valcon4.csv")
#
# data_writer(val_x.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx.csv")
# data_writer(val_x1_2.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx12.csv")
# data_writer(val_x1_3.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx13.csv")
# data_writer(val_x1_4.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx14.csv")
# data_writer(val_x1_1.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx11.csv")
# data_writer(val_x2_1.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx21.csv")
# data_writer(val_x2_3.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx23.csv")
# data_writer(val_x2_4.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx24.csv")
# data_writer(val_x2_2.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx22.csv")
# data_writer(val_x3_1.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx31.csv")
# data_writer(val_x3_2.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx32.csv")
# data_writer(val_x3_4.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx34.csv")
# data_writer(val_x3_3.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx33.csv")
# data_writer(val_x4_1.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx41.csv")
# data_writer(val_x4_2.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx42.csv")
# data_writer(val_x4_3.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx43.csv")
# data_writer(val_x4_4.reshape(np.shape(val_x)[0], -1), "./data/ensemble_reviced/valx44.csv")

test_x = []
test_y = []
for j in range(len(test_data) - x_len):
    test_x.append(test_data[j:j + 1 * x_len, :])
    test_y.append(test_data[j + 1 * x_len, -3:])

test_x = np.array(test_x)
test_x = np.reshape(test_x, (test_x.shape[0], x_len, -1))

test_y = np.array(test_y).reshape(test_x.shape[0], -1)

# test_x1_2, test_x1_3, test_x1_4, test_x1_1, shunxu1 = form_aux(np.arange(22460+x_len, 28000), y1_index, y2_index, y3_index, y4_index)
# test_x2_1, test_x2_3, test_x2_4, test_x2_2, shunxu2 = form_aux(np.arange(22460+x_len, 28000), y2_index, y1_index, y3_index, y4_index)
# test_x3_1, test_x3_2, test_x3_4, test_x3_3, shunxu3 = form_aux(np.arange(22460+x_len, 28000), y3_index, y1_index, y2_index, y4_index)
# test_x4_1, test_x4_2, test_x4_3, test_x4_4, shunxu4 = form_aux(np.arange(22460+x_len, 28000), y4_index, y1_index, y2_index, y3_index)

# shape1 = min(np.shape(test_x1_2)[0], np.shape(test_x1_3)[0], np.shape(test_x1_4)[0], np.shape(test_x1_1)[0])
# shape2 = min(np.shape(test_x2_1)[0], np.shape(test_x2_3)[0], np.shape(test_x2_4)[0], np.shape(test_x2_2)[0])
# shape3 = min(np.shape(test_x3_1)[0], np.shape(test_x3_2)[0], np.shape(test_x3_4)[0], np.shape(test_x3_3)[0])
# shape4 = min(np.shape(test_x4_2)[0], np.shape(test_x4_3)[0], np.shape(test_x4_1)[0], np.shape(test_x4_4)[0])
# shape = min(shape1, shape2, shape3, shape4)
# test_x = test_x[np.shape(test_x)[0]-shape:]
# test_x1_2 = test_x1_2[np.shape(test_x1_2)[0]-shape:]
# test_x1_3 = test_x1_3[np.shape(test_x1_3)[0]-shape:]
# test_x1_4 = test_x1_4[np.shape(test_x1_4)[0]-shape:]
# test_x1_1 = test_x1_1[np.shape(test_x1_1)[0]-shape:]
# # test_x2 = test_x[np.shape(test_x)[0]-shape]
# test_x2_1 = test_x2_1[np.shape(test_x2_1)[0]-shape:]
# test_x2_3 = test_x2_3[np.shape(test_x2_3)[0]-shape:]
# test_x2_4 = test_x2_4[np.shape(test_x2_4)[0]-shape:]
# test_x2_2 = test_x2_2[np.shape(test_x2_2)[0]-shape:]
# # test_x3 = test_x[np.shape(test_x)[0]-shape]
# test_x3_1 = test_x3_1[np.shape(test_x3_1)[0]-shape:]
# test_x3_2 = test_x3_2[np.shape(test_x3_2)[0]-shape:]
# test_x3_4 = test_x3_4[np.shape(test_x3_4)[0]-shape:]
# test_x3_3 = test_x3_3[np.shape(test_x3_3)[0]-shape:]
# # test_x4 = test_x[np.shape(test_x)[0]-shape]
# test_x4_1 = test_x4_1[np.shape(test_x4_1)[0]-shape:]
# test_x4_2 = test_x4_2[np.shape(test_x4_2)[0]-shape:]
# test_x4_3 = test_x4_3[np.shape(test_x4_3)[0]-shape:]
# test_x4_4 = test_x4_4[np.shape(test_x4_4)[0]-shape:]
#
# shunxu1 = shunxu1[np.shape(shunxu1)[0]-shape:]
# shunxu2 = shunxu2[np.shape(shunxu2)[0]-shape:]
# shunxu3 = shunxu3[np.shape(shunxu3)[0]-shape:]
# shunxu4 = shunxu4[np.shape(shunxu4)[0]-shape:]
test_x = pd.read_csv('./data/ensemble_reviced/testx.csv').values
test_x1_2 = pd.read_csv('./data/ensemble_reviced/testx12.csv').values
test_x1_3 = pd.read_csv('./data/ensemble_reviced/testx13.csv').values
test_x1_4 = pd.read_csv('./data/ensemble_reviced/testx14.csv').values
test_x1_1 = pd.read_csv('./data/ensemble_reviced/testx11.csv').values
test_x2_1 = pd.read_csv('./data/ensemble_reviced/testx21.csv').values
test_x2_3 = pd.read_csv('./data/ensemble_reviced/testx23.csv').values
test_x2_4 = pd.read_csv('./data/ensemble_reviced/testx24.csv').values
test_x2_2 = pd.read_csv('./data/ensemble_reviced/testx22.csv').values
test_x3_1 = pd.read_csv('./data/ensemble_reviced/testx31.csv').values
test_x3_2 = pd.read_csv('./data/ensemble_reviced/testx32.csv').values
test_x3_4 = pd.read_csv('./data/ensemble_reviced/testx34.csv').values
test_x3_3 = pd.read_csv('./data/ensemble_reviced/testx33.csv').values
test_x4_1 = pd.read_csv('./data/ensemble_reviced/testx41.csv').values
test_x4_2 = pd.read_csv('./data/ensemble_reviced/testx42.csv').values
test_x4_3 = pd.read_csv('./data/ensemble_reviced/testx43.csv').values
test_x4_4 = pd.read_csv('./data/ensemble_reviced/testx44.csv').values
test_con1 = pd.read_csv('./data/ensemble_reviced/testcon1.csv').values
test_con2 = pd.read_csv('./data/ensemble_reviced/testcon2.csv').values
test_con3 = pd.read_csv('./data/ensemble_reviced/testcon3.csv').values
test_con4 = pd.read_csv('./data/ensemble_reviced/testcon4.csv').values

test_y = test_y[np.shape(test_y)[0] - np.shape(test_x)[0]:]

test_x = test_x[:, 1:].reshape((-1, x_len, 8))
test_x1_2 = test_x1_2[:, 1:].reshape((-1, x_len, 8))
test_x1_3 = test_x1_3[:, 1:].reshape((-1, x_len, 8))
test_x1_4 = test_x1_4[:, 1:].reshape((-1, x_len, 8))
test_x1_1 = test_x1_1[:, 1:].reshape((-1, x_len, 8))
test_x2_1 = test_x2_1[:, 1:].reshape((-1, x_len, 8))
test_x2_3 = test_x2_3[:, 1:].reshape((-1, x_len, 8))
test_x2_4 = test_x2_4[:, 1:].reshape((-1, x_len, 8))
test_x2_2 = test_x2_2[:, 1:].reshape((-1, x_len, 8))
test_x3_1 = test_x3_1[:, 1:].reshape((-1, x_len, 8))
test_x3_2 = test_x3_2[:, 1:].reshape((-1, x_len, 8))
test_x3_4 = test_x3_4[:, 1:].reshape((-1, x_len, 8))
test_x3_3 = test_x3_3[:, 1:].reshape((-1, x_len, 8))
test_x4_1 = test_x4_1[:, 1:].reshape((-1, x_len, 8))
test_x4_2 = test_x4_2[:, 1:].reshape((-1, x_len, 8))
test_x4_3 = test_x4_3[:, 1:].reshape((-1, x_len, 8))
test_x4_4 = test_x4_4[:, 1:].reshape((-1, x_len, 8))
#
test_con1 = test_con1[:, 1:].reshape((np.shape(test_con1)[0], 9, -1))
test_con2 = test_con2[:, 1:].reshape((np.shape(test_con2)[0], 9, -1))
test_con3 = test_con3[:, 1:].reshape((np.shape(test_con3)[0], 9, -1))
test_con4 = test_con4[:, 1:].reshape((np.shape(test_con4)[0], 9, -1))
# test_con4 = np.concatenate((test_con4, test_con4), axis=2)
# test_x1_2 = np.array(test_x1_2)
# test_x1_3 = np.array(test_x1_3)
# test_x1_4 = np.array(test_x1_4)
# test_x1_1 = np.array(test_x1_1)
# test_x2_1 = np.array(test_x2_1)
# test_x2_3 = np.array(test_x2_3)
# test_x2_4 = np.array(test_x2_4)
# test_x2_2 = np.array(test_x2_2)
# test_x3_1 = np.array(test_x3_1)
# test_x3_2 = np.array(test_x3_2)
# test_x3_4 = np.array(test_x3_4)
# test_x3_3 = np.array(test_x3_3)
# test_x4_1 = np.array(test_x4_1)
# test_x4_2 = np.array(test_x4_2)
# test_x4_3 = np.array(test_x4_3)
# test_x4_4 = np.array(test_x4_4)
# test_y = np.array(test_y)
# test_con1 = form_con(shunxu1, 64)
# test_con2 = form_con(shunxu2, 64)
# test_con3 = form_con(shunxu3, 64)
# test_con4 = form_con(shunxu4, 64)

# data_writer(test_con1.reshape(np.shape(test_con1)[0], -1), "./data/ensemble_reviced/testcon1.xlsx")
# data_writer(test_con2.reshape(np.shape(test_con2)[0], -1), "./data/ensemble_reviced/testcon2.xlsx")
# data_writer(test_con3.reshape(np.shape(test_con3)[0], -1), "./data/ensemble_reviced/testcon3.xlsx")
# data_writer(test_con4.reshape(np.shape(test_con4)[0], -1), "./data/ensemble_reviced/testcon4.xlsx")
#
# print(np.shape(test_x))
# print(np.shape(test_x1_2))
# print(np.shape(test_y))
#
# data_writer(test_x.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx.csv")
# data_writer(test_x1_2.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx12.csv")
# data_writer(test_x1_3.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx13.csv")
# data_writer(test_x1_4.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx14.csv")
# data_writer(test_x1_1.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx11.csv")
# data_writer(test_x2_1.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx21.csv")
# data_writer(test_x2_3.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx23.csv")
# data_writer(test_x2_4.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx24.csv")
# data_writer(test_x2_2.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx22.csv")
# data_writer(test_x3_1.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx31.csv")
# data_writer(test_x3_2.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx32.csv")
# data_writer(test_x3_4.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx34.csv")
# data_writer(test_x3_3.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx33.csv")
# data_writer(test_x4_1.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx41.csv")
# data_writer(test_x4_2.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx42.csv")
# data_writer(test_x4_3.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx43.csv")
# data_writer(test_x4_4.reshape(np.shape(test_x)[0], -1), "./data/ensemble_reviced/testx44.csv")

g1 = tf.Graph()
with g1.as_default():
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

    # third layer
    lstm31 = Lstm32(m_input_width, state_width, batch_size, time_weight, space_weight)

    # 2second layer
    lstm22 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)

    # 2first layer
    lstm12 = Lstm1(y_input_width, state_width, batch_size)

    # 3second layer
    lstm23 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)

    # 3first layer
    lstm13 = Lstm1(y_input_width, state_width, batch_size)

    # 4second layer
    lstm24 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 4first layer
    lstm14 = Lstm1(y_input_width, state_width, batch_size)

    # 5second layer
    lstm25 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 5first layer
    lstm15 = Lstm1(y_input_width, state_width, batch_size)

    # 6second layer
    lstm26 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 6first layer
    lstm16 = Lstm1(y_input_width, state_width, batch_size)

    # 7second layer
    lstm27 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 7first layer
    lstm17 = Lstm1(y_input_width, state_width, batch_size)

    # 8second layer
    lstm28 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 8first layer
    lstm18 = Lstm1(y_input_width, state_width, batch_size)

    # 9second layer
    lstm29 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 9first layer
    lstm19 = Lstm1(y_input_width, state_width, batch_size)

    # MLP
    mlp1 = mlp(output_width, state_width, output_width1, batch_size)

    # print(s==tf.Tensor(1))

    x = tf.placeholder(tf.float32, [batch_size, x_len, x_input_width])
    y = tf.placeholder(tf.float32, [batch_size, x_len, y_input_width])
    m = tf.placeholder(tf.float32, [batch_size, x_len, m_input_width])
    z = tf.placeholder(tf.float32, [batch_size, x_len, z_input_width])
    n = tf.placeholder(tf.float32, [batch_size, x_len, z_input_width])
    control = tf.placeholder(tf.float32, [batch_size, 9, state_width])
    label = tf.placeholder(tf.float32, [batch_size, 3])

    h11 = lstm11.forward(y)
    # h12=lstm12.forward(h11)
    # h13=lstm13.forward(h12)
    h21 = lstm21.forward(h11, z)
    # h22=lstm22.forward(h12,h21)
    # h23=lstm23.forward(h13,h22)

    h12 = lstm12.forward(z)
    # h52=lstm52.forward(h51)
    # h53=lstm53.forward(h52)
    h22 = lstm22.forward(h12, y)

    h13 = lstm13.forward(y)
    h23 = lstm23.forward(h13, m)

    h14 = lstm14.forward(m)
    h24 = lstm24.forward(h14, y)

    h15 = lstm15.forward(z)
    h25 = lstm25.forward(h15, m)

    h16 = lstm16.forward(m)
    h26 = lstm26.forward(h16, z)

    h17 = lstm17.forward(n)
    h27 = lstm27.forward(h17, y)

    h18 = lstm18.forward(n)
    h28 = lstm28.forward(h18, z)

    h19 = lstm19.forward(n)
    h29 = lstm29.forward(h19, m)

    h41, onet = lstm31.forward(h21, h22, h23, h24, h25, h26, h27, h28, h29, x, control)
    # h32=lstm32.forward(h42,h22,h31)
    # h33=lstm33.forward(h43,h23,h32)
    z1_c1 = mlp1.forward(h41)

    loss = tf.reduce_mean(tf.square(label - z1_c1))
    train_op1 = tf.train.AdamOptimizer(learningrate).minimize(loss)
    init_op1 = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())

g2 = tf.Graph()
with g2.as_default():
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
    space_weight = 0.6
    time_weight = 0.4

    # 1first layer
    lstm11 = Lstm1(z_input_width, state_width, batch_size)

    # 1second layer
    lstm21 = Lstm2(y_input_width, state_width, batch_size, time_weight, space_weight)

    # third layer
    lstm31 = Lstm32(m_input_width, state_width, batch_size, time_weight, space_weight)

    # 2second layer
    lstm22 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)

    # 2first layer
    lstm12 = Lstm1(y_input_width, state_width, batch_size)

    # 3second layer
    lstm23 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)

    # 3first layer
    lstm13 = Lstm1(y_input_width, state_width, batch_size)

    # 4second layer
    lstm24 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 4first layer
    lstm14 = Lstm1(y_input_width, state_width, batch_size)

    # 5second layer
    lstm25 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 5first layer
    lstm15 = Lstm1(y_input_width, state_width, batch_size)

    # 6second layer
    lstm26 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 6first layer
    lstm16 = Lstm1(y_input_width, state_width, batch_size)

    # 7second layer
    lstm27 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 7first layer
    lstm17 = Lstm1(y_input_width, state_width, batch_size)

    # 8second layer
    lstm28 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 8first layer
    lstm18 = Lstm1(y_input_width, state_width, batch_size)

    # 9second layer
    lstm29 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 9first layer
    lstm19 = Lstm1(y_input_width, state_width, batch_size)

    # MLP
    mlp1 = mlp(output_width, state_width, output_width1, batch_size)

    # print(s==tf.Tensor(1))

    x1 = tf.placeholder(tf.float32, [batch_size, x_len, x_input_width])
    y1 = tf.placeholder(tf.float32, [batch_size, x_len, y_input_width])
    m1 = tf.placeholder(tf.float32, [batch_size, x_len, m_input_width])
    z1 = tf.placeholder(tf.float32, [batch_size, x_len, z_input_width])
    n1 = tf.placeholder(tf.float32, [batch_size, x_len, z_input_width])
    control1 = tf.placeholder(tf.float32, [batch_size, 9, state_width])
    label1 = tf.placeholder(tf.float32, [batch_size, 3])

    h11 = lstm11.forward(y1)
    # h12=lstm12.forward(h11)
    # h13=lstm13.forward(h12)
    h21 = lstm21.forward(h11, z1)
    # h22=lstm22.forward(h12,h21)
    # h23=lstm23.forward(h13,h22)

    h12 = lstm12.forward(z1)
    # h52=lstm52.forward(h51)
    # h53=lstm53.forward(h52)
    h22 = lstm22.forward(h12, y1)

    h13 = lstm13.forward(y1)
    h23 = lstm23.forward(h13, m1)

    h14 = lstm14.forward(m1)
    h24 = lstm24.forward(h14, y1)

    h15 = lstm15.forward(z1)
    h25 = lstm25.forward(h15, m1)

    h16 = lstm16.forward(m1)
    h26 = lstm26.forward(h16, z1)

    h17 = lstm17.forward(n1)
    h27 = lstm27.forward(h17, y1)

    h18 = lstm18.forward(n1)
    h28 = lstm28.forward(h18, z1)

    h19 = lstm19.forward(n1)
    h29 = lstm29.forward(h19, m1)

    h41, onet = lstm31.forward(h21, h22, h23, h24, h25, h26, h27, h28, h29, x1, control1)
    # h32=lstm32.forward(h42,h22,h31)
    # h33=lstm33.forward(h43,h23,h32)
    z1_c2 = mlp1.forward(h41)

    loss1 = tf.reduce_mean(tf.square(label1 - z1_c2))
    train_op11 = tf.train.AdamOptimizer(learningrate).minimize(loss1)
    init_op11 = tf.global_variables_initializer()
    saver1 = tf.train.Saver(tf.global_variables())

g3 = tf.Graph()
with g3.as_default():
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
    space_weight = 0.6
    time_weight = 0.8
    space_weight2 = 0.6
    time_weight2 = 0.4

    # 1first layer
    lstm11 = Lstm1(z_input_width, state_width, batch_size)

    # 1second layer
    lstm21 = Lstm2(y_input_width, state_width, batch_size, time_weight, space_weight)

    # third layer
    lstm31 = Lstm32(m_input_width, state_width, batch_size, time_weight, space_weight)

    # 2second layer
    lstm22 = Lstm2(z_input_width, state_width, batch_size, time_weight2, space_weight2)

    # 2first layer
    lstm12 = Lstm1(y_input_width, state_width, batch_size)

    # 3second layer
    lstm23 = Lstm2(z_input_width, state_width, batch_size, time_weight2, space_weight2)

    # 3first layer
    lstm13 = Lstm1(y_input_width, state_width, batch_size)

    # 4second layer
    lstm24 = Lstm2(z_input_width, state_width, batch_size, time_weight2, space_weight2)
    # 4first layer
    lstm14 = Lstm1(y_input_width, state_width, batch_size)

    # 5second layer
    lstm25 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 5first layer
    lstm15 = Lstm1(y_input_width, state_width, batch_size)

    # 6second layer
    lstm26 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 6first layer
    lstm16 = Lstm1(y_input_width, state_width, batch_size)

    # 7second layer
    lstm27 = Lstm2(z_input_width, state_width, batch_size, time_weight2, space_weight2)
    # 7first layer
    lstm17 = Lstm1(y_input_width, state_width, batch_size)

    # 8second layer
    lstm28 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 8first layer
    lstm18 = Lstm1(y_input_width, state_width, batch_size)

    # 9second layer
    lstm29 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 9first layer
    lstm19 = Lstm1(y_input_width, state_width, batch_size)

    # MLP
    mlp1 = mlp(output_width, state_width, output_width1, batch_size)

    # print(s==tf.Tensor(1))

    x2 = tf.placeholder(tf.float32, [batch_size, x_len, x_input_width])
    y2 = tf.placeholder(tf.float32, [batch_size, x_len, y_input_width])
    m2 = tf.placeholder(tf.float32, [batch_size, x_len, m_input_width])
    z2 = tf.placeholder(tf.float32, [batch_size, x_len, z_input_width])
    n2 = tf.placeholder(tf.float32, [batch_size, x_len, z_input_width])
    control2 = tf.placeholder(tf.float32, [batch_size, 9, state_width])
    label2 = tf.placeholder(tf.float32, [batch_size, 3])

    h11 = lstm11.forward(y2)
    # h12=lstm12.forward(h11)
    # h13=lstm13.forward(h12)
    h21 = lstm21.forward(h11, z2)
    # h22=lstm22.forward(h12,h21)
    # h23=lstm23.forward(h13,h22)

    h12 = lstm12.forward(z2)
    # h52=lstm52.forward(h51)
    # h53=lstm53.forward(h52)
    h22 = lstm22.forward(h12, y2)

    h13 = lstm13.forward(y2)
    h23 = lstm23.forward(h13, m2)

    h14 = lstm14.forward(m2)
    h24 = lstm24.forward(h14, y2)

    h15 = lstm15.forward(z2)
    h25 = lstm25.forward(h15, m2)

    h16 = lstm16.forward(m2)
    h26 = lstm26.forward(h16, z2)

    h17 = lstm17.forward(n2)
    h27 = lstm27.forward(h17, y2)

    h18 = lstm18.forward(n2)
    h28 = lstm28.forward(h18, z2)

    h19 = lstm19.forward(n2)
    h29 = lstm29.forward(h19, m2)

    h41, onet = lstm31.forward(h21, h22, h23, h24, h25, h26, h27, h28, h29, x2, control2)
    # h32=lstm32.forward(h42,h22,h31)
    # h33=lstm33.forward(h43,h23,h32)
    z1_c3 = mlp1.forward(h41)

    loss2 = tf.reduce_mean(tf.square(label2 - z1_c3))
    train_op12 = tf.train.AdamOptimizer(learningrate).minimize(loss2)
    init_op12 = tf.global_variables_initializer()
    saver2 = tf.train.Saver(tf.global_variables())

g4 = tf.Graph()
with g4.as_default():
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
    space_weight = 0.6
    time_weight = 0.8
    space_weight2 = 0.6
    time_weight2 = 0.4

    # 1first layer
    lstm11 = Lstm1(z_input_width, state_width, batch_size)

    # 1second layer
    lstm21 = Lstm2(y_input_width, state_width, batch_size, time_weight, space_weight)

    # third layer
    lstm31 = Lstm32(m_input_width, state_width, batch_size, time_weight2, space_weight2)

    # 2second layer
    lstm22 = Lstm2(z_input_width, state_width, batch_size, time_weight2, space_weight2)

    # 2first layer
    lstm12 = Lstm1(y_input_width, state_width, batch_size)

    # 3second layer
    lstm23 = Lstm2(z_input_width, state_width, batch_size, time_weight2, space_weight2)

    # 3first layer
    lstm13 = Lstm1(y_input_width, state_width, batch_size)

    # 4second layer
    lstm24 = Lstm2(z_input_width, state_width, batch_size, time_weight2, space_weight2)
    # 4first layer
    lstm14 = Lstm1(y_input_width, state_width, batch_size)

    # 5second layer
    lstm25 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 5first layer
    lstm15 = Lstm1(y_input_width, state_width, batch_size)

    # 6second layer
    lstm26 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 6first layer
    lstm16 = Lstm1(y_input_width, state_width, batch_size)

    # 7second layer
    lstm27 = Lstm2(z_input_width, state_width, batch_size, time_weight2, space_weight2)
    # 7first layer
    lstm17 = Lstm1(y_input_width, state_width, batch_size)

    # 8second layer
    lstm28 = Lstm2(z_input_width, state_width, batch_size, time_weight, space_weight)
    # 8first layer
    lstm18 = Lstm1(y_input_width, state_width, batch_size)

    # 9second layer
    lstm29 = Lstm2(z_input_width, state_width, batch_size, time_weight2, space_weight2)
    # 9first layer
    lstm19 = Lstm1(y_input_width, state_width, batch_size)

    # MLP
    mlp1 = mlp(output_width, state_width, output_width1, batch_size)

    # print(s==tf.Tensor(1))

    x3 = tf.placeholder(tf.float32, [batch_size, x_len, x_input_width])
    y3 = tf.placeholder(tf.float32, [batch_size, x_len, y_input_width])
    m3 = tf.placeholder(tf.float32, [batch_size, x_len, m_input_width])
    z3 = tf.placeholder(tf.float32, [batch_size, x_len, z_input_width])
    n3 = tf.placeholder(tf.float32, [batch_size, x_len, z_input_width])
    control3 = tf.placeholder(tf.float32, [batch_size, 9, state_width])
    label3 = tf.placeholder(tf.float32, [batch_size, 3])

    h11 = lstm11.forward(y3)
    # h12=lstm12.forward(h11)
    # h13=lstm13.forward(h12)
    h21 = lstm21.forward(h11, z3)
    # h22=lstm22.forward(h12,h21)
    # h23=lstm23.forward(h13,h22)

    h12 = lstm12.forward(z3)
    # h52=lstm52.forward(h51)
    # h53=lstm53.forward(h52)
    h22 = lstm22.forward(h12, y3)

    h13 = lstm13.forward(y3)
    h23 = lstm23.forward(h13, m3)

    h14 = lstm14.forward(m3)
    h24 = lstm24.forward(h14, y3)

    h15 = lstm15.forward(z3)
    h25 = lstm25.forward(h15, m3)

    h16 = lstm16.forward(m3)
    h26 = lstm26.forward(h16, z3)

    h17 = lstm17.forward(n3)
    h27 = lstm27.forward(h17, y3)

    h18 = lstm18.forward(n3)
    h28 = lstm28.forward(h18, z3)

    h19 = lstm19.forward(n3)
    h29 = lstm29.forward(h19, m3)

    h41, onet = lstm31.forward(h21, h22, h23, h24, h25, h26, h27, h28, h29, x3, control3)
    # h32=lstm32.forward(h42,h22,h31)
    # h33=lstm33.forward(h43,h23,h32)
    z1_c4 = mlp1.forward(h41)

    loss3 = tf.reduce_mean(tf.square(label3 - z1_c4))
    train_op13 = tf.train.AdamOptimizer(learningrate).minimize(loss3)
    init_op13 = tf.global_variables_initializer()
    saver3 = tf.train.Saver(tf.global_variables())

with tf.Session(graph=g1) as sess:
    # saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph('./model_checkpoint/MyModel.meta')
    saver.restore(sess, "./model_checkpoint_reviced/condition1/wc0.8_wt1_bs32/c1")
    # graph = tf.get_default_graph()
    # graphdef = graph.as_graph_def()
    # summary_write = tf.summary.FileWriter("./graph", graph)
    # summary_write.close()
    # input_x1 = graph.get_operation_by_name("Placeholder:0")
    # input_x2 = graph.get_operation_by_name("Placeholder_1:0")
    # label1 = graph.get_operation_by_name("Placeholder_2:0")
    # label2 = graph.get_operation_by_name("Placeholder_3:0")
    # input_x1 = tf.placeholder(tf.float32, [batch_size, x_len, 8])
    # input_x2 = tf.placeholder(tf.float32, [batch_size, x_len, 8])
    # z1 =graph.get_operation_by_name("add_781:0")
    # z2 =graph.get_operation_by_name("add_783:0")

    train_input1 = np.zeros([batch_size, 3])
    for i in range(np.shape(train_x)[0] // batch_size):
        trx1 = sess.run(z1_c1, feed_dict={x: train_x[i * batch_size:(i + 1) * batch_size],
                                    y: train_x1_2[i * batch_size:(i + 1) * batch_size],
                                     z: train_x1_3[i * batch_size:(i + 1) * batch_size],
                                     m: train_x1_4[i * batch_size:(i + 1) * batch_size],
                                     n: train_x1_1[i * batch_size:(i + 1) * batch_size],
                                     control: train_con1[i * batch_size:(i + 1) * batch_size, :, :]})
        train_input1 = np.vstack((train_input1, trx1))


    train_input1 = train_input1[batch_size:, :]
    print(np.shape(train_input1))
    # data_writer(train_input1, "./data/ensemble/input1.xlsx", 'input1')
    df = pd.DataFrame(train_input1)
    df.to_csv("./data/ensemble_reviced/wc0.8_wt1_input1.csv", sep=',')
    val_input1 = np.zeros([batch_size, 3])
    for i in range(np.shape(val_x)[0] // batch_size):
        vax1 = sess.run(z1_c1, feed_dict={x: val_x[i * batch_size:(i + 1) * batch_size],
                                        y: val_x1_2[i * batch_size:(i + 1) * batch_size],
                                        z: val_x1_3[i * batch_size:(i + 1) * batch_size],
                                        m: val_x1_4[i*batch_size:(i+1)*batch_size],
                                        n: val_x1_1[i*batch_size:(i+1)*batch_size],
                                     control: val_con1[i * batch_size:(i + 1) * batch_size, :, :]})
        val_input1 = np.vstack((val_input1, vax1))

    val_input1 = val_input1[batch_size:, :]
    print(np.shape(val_input1))
    # data_writer(val_input1, "./data/ensemble/val_input1.xlsx", 'input1')
    df = pd.DataFrame(val_input1)
    df.to_csv("./data/ensemble_reviced/wc0.8_wt1_valinput1.csv", sep=',')
#
# with tf.Session(graph=g2) as sess:
#     saver1.restore(sess, "./model_checkpoint_reviced/condition2/wc0.6_wt0.4/c2")
    # train_input2 = np.zeros([batch_size, 3])
    # for i in range(np.shape(train_x)[0] // batch_size):
    #     trx2 = sess.run(z1_c2, feed_dict={x1: train_x[i * batch_size:(i + 1) * batch_size],
    #                                             y1: train_x2_1[i * batch_size:(i + 1) * batch_size],
    #                                              z1: train_x2_3[i * batch_size:(i + 1) * batch_size],
    #                                              m1: train_x2_4[i*batch_size:(i+1)*batch_size],
    #                                              n1: train_x2_2[i*batch_size:(i+1)*batch_size],
    #                                              control1: train_con2[i*batch_size:(i+1)*batch_size]})
    #
    #     train_input2 = np.vstack((train_input2, trx2))
    #
    # train_input2 = train_input2[batch_size:, :]
    #
    # print(np.shape(train_input2))
    # data_writer(train_input2, './data/ensemble_reviced/wc0.6_wt0.4_traininput2.csv')

    # val_input2 = np.zeros([batch_size, 3])
    # for i in range(np.shape(val_x)[0] // batch_size):
    #     x2 = sess.run(z1_c2, feed_dict={x1: val_x[i * batch_size:(i + 1) * batch_size],
    #                                             y1: val_x2_1[i * batch_size:(i + 1) * batch_size],
    #                                              z1: val_x2_3[i * batch_size:(i + 1) * batch_size],
    #                                              m1: val_x2_4[i*batch_size:(i+1)*batch_size],
    #                                              n1: val_x2_2[i*batch_size:(i+1)*batch_size],
    #                                             control1: val_con2[i*batch_size:(i+1)*batch_size]})
    #     val_input2 = np.vstack((val_input2, x2))
    #
    # val_input2 = val_input2[batch_size:, :]
    # print(np.shape(val_input2))
    # data_writer(val_input2, './data/ensemble_reviced/wc0.6_wt0.4_valinput2.csv')

# with tf.Session(graph=g3) as sess:
#     saver2.restore(sess, "./model_checkpoint_reviced/newtr_condition3/wc0.6_wt0.8_c1/c3")
#     train_input3 = np.zeros([batch_size, 3])
#     for i in range(np.shape(train_x)[0] // batch_size):
#         trx3 = sess.run(z1_c3, feed_dict={x2: train_x[i * batch_size:(i + 1) * batch_size],
#                                         y2: train_x3_1[i * batch_size:(i + 1) * batch_size],
#                                         z2: train_x3_2[i * batch_size:(i + 1) * batch_size],
#                                         m2: train_x3_4[i * batch_size:(i + 1) * batch_size],
#                                         n2: train_x3_3[i * batch_size:(i + 1) * batch_size],
#                                         control2: train_con3[i*batch_size:(i+1)*batch_size]})
#
#         train_input3 = np.vstack((train_input3, trx3))
#
#     train_input3 = train_input3[batch_size:, :]
#     print(np.shape(train_input3))
#     data_writer(train_input3, './data/ensemble_reviced/traininput3.csv')
#
#     val_input3 = np.zeros([batch_size, 3])
#     for i in range(np.shape(val_x)[0] // batch_size):
#         valx3 = sess.run(z1_c3, feed_dict={x2: val_x[i * batch_size:(i + 1) * batch_size],
#                                      y2: val_x3_1[i * batch_size:(i + 1) * batch_size],
#                                      z2: val_x3_2[i * batch_size:(i + 1) * batch_size],
#                                      m2: val_x3_4[i * batch_size:(i + 1) * batch_size],
#                                      n2: val_x3_3[i * batch_size:(i + 1) * batch_size],
#                                      control2: val_con3[i*batch_size:(i+1)*batch_size]})
#         val_input3 = np.vstack((val_input3, valx3))
#
#     val_input3 = val_input3[batch_size:, :]
#     print(np.shape(val_input3))
#     data_writer(val_input3, './data/ensemble_reviced/valinput3.csv')
#
# with tf.Session(graph=g4) as sess:
#     saver3.restore(sess, "./model_checkpoint_reviced/newtr_condition4/c2_0.01_0.01/c4")
#     batch_size = 32
#     train_input4 = np.zeros([batch_size, 3])
#     for i in range(np.shape(train_x)[0] // batch_size):
#         trx4 = sess.run(z1_c4, feed_dict={x3: train_x[i * batch_size:(i + 1) * batch_size],
#                                         y3: train_x4_1[i * batch_size:(i + 1) * batch_size],
#                                         z3: train_x4_2[i * batch_size:(i + 1) * batch_size],
#                                         m3: train_x4_3[i * batch_size:(i + 1) * batch_size],
#                                         n3: train_x4_4[i * batch_size:(i + 1) * batch_size],
#                                         control3: train_con4[i*batch_size:(i+1)*batch_size]})
#
#         train_input4 = np.vstack((train_input4, trx4))
#
#     train_input4 = train_input4[batch_size:, :]
#
#     print(np.shape(train_input4))
#     data_writer(train_input4, './data/ensemble_reviced/traininput4.csv')
#
#     val_input4 = np.zeros([batch_size, 3])
#     for i in range(np.shape(val_x)[0] // batch_size):
#         valx4 = sess.run(z1_c4, feed_dict={x3: val_x[i * batch_size:(i + 1) * batch_size],
#                                         y3: val_x4_1[i * batch_size:(i + 1) * batch_size],
#                                         z3: val_x4_2[i * batch_size:(i + 1) * batch_size],
#                                         m3: val_x4_3[i * batch_size:(i + 1) * batch_size],
#                                         n3: val_x4_4[i * batch_size:(i + 1) * batch_size],
#                                         control3: val_con4[i*batch_size:(i+1)*batch_size]})
#         val_input4 = np.vstack((val_input4, valx4))
#
#     val_input4 = val_input4[batch_size:, :]
#     print(np.shape(val_input4))
#     data_writer(val_input4, './data/ensemble_reviced/valinput4.csv')
# saver = tf.train.import_meta_graph('./model_checkpoint/quanju/LSTM.meta')
# saver.restore(sess, "./model_checkpoint/quanju/LSTM")
#
# graph = tf.get_default_graph()
# input_x1 = graph.get_operation_by_name("xs:0")
# label1 = graph.get_operation_by_name("ys:0")
# z1 = graph.get_operation_by_name("op_to_restore:0")


model5 = keras.models.load_model("E:/老师材料/建青论文/论文代码整理/jianqing_parper/modelh5_reviced_check/正则多工况模型集成relulayer1_50_layer2_50_lr1efu3_dp0.h5")

train_input5 = model5.predict(train_x)
val_input5 = model5.predict(val_x)

train_input1 = pd.read_csv('./data/ensemble_reviced/wc0.6_wt0.8_input1.csv').values
train_input2 = pd.read_csv('./data/ensemble_reviced/wc0.6_wt0.4_traininput2.csv').values
train_input3 = pd.read_csv('./data/ensemble_reviced/traininput3.csv').values
train_input4 = pd.read_csv('./data/ensemble_reviced/traininput4.csv').values
val_input1 = pd.read_csv('./data/ensemble_reviced/wc0.6_wt0.8_valinput1.csv').values
val_input2 = pd.read_csv('./data/ensemble_reviced/wc0.6_wt0.4_valinput2.csv').values
val_input3 = pd.read_csv('./data/ensemble_reviced/valinput3.csv').values
val_input4 = pd.read_csv('./data/ensemble_reviced/valinput4.csv').values

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

mse1 = metrics.mean_squared_error(train_input1, train_y[:len(train_input1),:,0])
mse2 = metrics.mean_squared_error(train_input2, train_y[:len(train_input1),:,0])
mse3 = metrics.mean_squared_error(train_input3, train_y[:len(train_input1),:,0])
mse4 = metrics.mean_squared_error(train_input4, train_y[:len(train_input1),:,0])
mse5 = metrics.mean_squared_error(train_input5, train_y[:len(train_input1),:,0])

zong = mse1+mse2+mse3+mse4+mse5

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

ex = keras.layers.Dense(30, activation='relu')(input_x)
# x = keras.layers.Dropout(0.2)(x)
ex = keras.layers.Dense(30, activation='relu')(ex)
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

model.save("./modelh5_reviced_check/正则多工况模型集成relulayer1_30_layer2_30_lr1efu3_dp0.h5")

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

model = keras.models.load_model("./modelh5_reviced_check/正则多工况模型集成relulayer1_30_layer2_30_lr1efu3_dp0.h5")

sess = tf.Session(graph=g1)
sess1 = tf.Session(graph=g2)
sess2 = tf.Session(graph=g3)
sess3 = tf.Session(graph=g4)
saver.restore(sess, "./model_checkpoint_reviced/condition1/wc0.8_wt1_bs32/c1")
saver1.restore(sess1, "./model_checkpoint_reviced/condition2/wc0.6_wt0.4/c2")
saver2.restore(sess2, "./model_checkpoint_reviced/newtr_condition3/wc0.6_wt0.8_c1/c3")
saver3.restore(sess3, "./model_checkpoint_reviced/newtr_condition4/c2_0.01_0.01/c4")

Zushu = 50
pred_y_sum = []
for i in range(Zushu):
    test_x_in = np.array(test_x[i * 60]).reshape((-1, x_len, 8))
    pred_y = []
    for j in range(60):
        print("第", j, "次迭代")

        tx1 = sess.run(z1_c1, feed_dict={x: data_fill(test_x_in, 32),
                                         y: data_fill(test_x1_2[i * 60].reshape((-1, x_len, 8)), 32),
                                         z: data_fill(test_x1_3[i * 60].reshape((-1, x_len, 8)), 32),
                                         m: data_fill(test_x1_4[i * 60].reshape((-1, x_len, 8)), 32),
                                         n: data_fill(test_x1_1[i * 60].reshape((-1, x_len, 8)), 32),
                                         control: data_fill(test_con1[i * 60, :, :].reshape((1, 9, -1)), 32)})

        # loss = tf.reduce_mean(tf.square(label - z1))
        # train_op = tf.train.AdamOptimizer(learningrate).minimize(loss)
        # init_op = tf.global_variables_initializer()
        # saver = tf.train.Saver(tf.global_variables())
        # with tf.Session(graph=g2) as sess1:

        tx2 = sess1.run(z1_c2, feed_dict={x1: data_fill(test_x_in, 32),
                                        y1: data_fill(test_x2_1[i * 60].reshape((-1, x_len, 8)), 32),
                                        z1: data_fill(test_x2_3[i * 60].reshape((-1, x_len, 8)), 32),
                                        m1: data_fill(test_x2_4[i * 60].reshape((-1, x_len, 8)), 32),
                                        n1: data_fill(test_x2_2[i * 60].reshape((-1, x_len, 8)), 32),
                                        control1: data_fill(test_con2[i * 60].reshape((1, 9, -1)), 32)})

        tx3 = sess2.run(z1_c3, feed_dict={x2: data_fill(test_x_in, 32),
                                        y2: data_fill(test_x3_1[i * 60].reshape((-1, x_len, 8)), 32),
                                        z2: data_fill(test_x3_2[i * 60].reshape((-1, x_len, 8)), 32),
                                        m2: data_fill(test_x3_4[i * 60].reshape((-1, x_len, 8)), 32),
                                        n2: data_fill(test_x3_3[i * 60].reshape((-1, x_len, 8)), 32),
                                        control2: data_fill(test_con3[i * 60].reshape((1, 9, -1)), 32)})

        tx4 = sess3.run(z1_c4, feed_dict={x3: data_fill(test_x_in, 32),
                                         y3: data_fill(test_x4_1[i * 60].reshape((-1, x_len, 8)), 32),
                                         z3: data_fill(test_x4_2[i * 60].reshape((-1, x_len, 8)), 32),
                                         m3: data_fill(test_x4_3[i * 60].reshape((-1, x_len, 8)), 32),
                                         n3: data_fill(test_x4_4[i * 60].reshape((-1, x_len, 8)), 32),
                                         control3: data_fill(test_con4[i * 60].reshape((1, 9, -1)), 32)})
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

        pre_y = model.predict([tx1, tx2, tx3, tx4, tx5])
        # 平均
        # pre_y = (tx1+tx2+tx3+tx4+tx5)/5
        # 按误差加权
        # y = (tx1*mse1+tx2*mse2+tx3*mse3+tx4*mse4+tx5*mse5)/zong
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

data_writer(pred_y_sum[:, :, 0], "./修改结果保存/relu_30_30/1柜预测值.csv")
data_writer(pred_y_sum[:, :, 1], "./修改结果保存/relu_30_30/2柜预测值.csv")
data_writer(pred_y_sum[:, :, 2], "./修改结果保存/relu_30_30/3柜预测值.csv")

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



