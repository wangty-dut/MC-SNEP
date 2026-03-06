import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import math
import xlwt
import datetime
from LSTMONE import Lstm1
from LSTMTWO import Lstm2
from LSTMTHREE import Lstm31, Lstm32, Lstm33
from MLP import mlp
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from Data_writer import data_writer

tf.compat.v1.disable_eager_execution()
starttime = datetime.datetime.now()
np.random.seed(5)
tf.set_random_seed(5)

batch_size = 32
learningrate = 0.001
x_input_width = 8
y_input_width = 8
z_input_width = 8
m_input_width = 8
state_width = 64

output_width1 = 32
# train_size=300
# verification_size=100
# test_size=100
traintimes = 200
space_weight = 0.4
time_weight = 0.6


def rmse(y_true, y_pred):
    n = len(y_true)
    sum = 0
    for i in range(n):
        temp = math.pow(y_true[i] - y_pred[i], 2)
        sum = sum + temp
    rmse = math.sqrt(sum / n)
    rmse = float(rmse)
    return rmse


def mape(y_true, y_pred):
    n = len(y_true)
    sum = 0
    for i in range(n):
        temp0 = abs((y_true[i] - y_pred[i]) / y_true[i])
        sum = sum + temp0
    mape = sum / n * 100
    return mape


def mae(y_true, y_pred):
    n = len(y_true)
    sum = 0
    for i in range(n):
        temp = abs(y_true[i] - y_pred[i])
        sum = sum + temp
    mae = sum / n
    mae = float(mae)
    return mae


def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:
        median = (data[size // 2] + data[size // 2 - 1]) / 2
        data[0] = median
    if size % 2 == 1:
        median = data[(size - 1) // 2]
        data[0] = median
    return data[0]


def data_add(data, num):
    if num == 1:
        data_compose = data
    elif num == 2:
        data_compose = np.concatenate((data, data), axis=1)
    elif num > 2:
        data_compose = np.concatenate((data, data), axis=1)
        for i in range(num - 2):
            data_compose = np.concatenate((data_compose, data), axis=1)
    return data_compose


def data_fill(data, batchsize):
    a = np.zeros([batchsize - 1, data.shape[1], data.shape[2]])
    b = np.concatenate((data, a), axis=0)
    return b


def data_write(file_path, datas, col):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # establish sheet
    # write data
    i = 0
    for data in datas:
        sheet1.write(i, col, data)
        i = i + 1
    f.save(file_path)  # save doc


df = pd.read_excel('G:/researchproject/数据导出_LDG相关（02.01-）/汇总.xlsx', header=0, usecols=[1, 2, 3, 4, 6, 10, 11, 12])
rawdata = df.values  # X

state_data = pd.read_excel('G:/researchproject/数据导出_LDG相关（02.01-）/汇总.xlsx', header=0, usecols=[16]).values
print(np.shape(rawdata))

x_len = 60
pred_len = 60
output_width = 3
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

test_data = (test_data - mean_value) / std_value
rawdata = (rawdata - mean_value) /std_value


train_x1 = []
train_y1 = []
train_x2 = []
train_y2 = []
train_x3 = []
train_y3 = []
train_x4 = []
train_y4 = []
y1_index = []
y2_index = []
y3_index = []
y4_index = []

train_size = int((len(train_data) - x_len - pred_len))

print(len(train_data))
for i in range(train_size):
    if ((state_data[i + x_len] == 7) or (state_data[i + x_len] == 5)):
        train_x1.append(train_data[i: i + x_len, :])
        train_y1.append(train_data[i + x_len, -3:])
        y1_index.append(i + x_len)
    elif (state_data[i + x_len] == 6 or state_data[i + x_len] == 4):
        train_x2.append(train_data[i: i + x_len, :])
        train_y2.append(train_data[i + x_len, -3:])
        y2_index.append(i + x_len)
    elif (state_data[i + x_len] == 3):
        train_x3.append(train_data[i: i + x_len, :])
        train_y3.append(train_data[i + x_len, -3:])
        y3_index.append(i + x_len)
    elif (state_data[i + x_len] == 2):
        train_x4.append(train_data[i: i + x_len, :])
        train_y4.append(train_data[i + x_len, -3:])
        y4_index.append(i + x_len)


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


# train_x1_2, train_x1_3, train_x1_4, shunxu1 = form_aux(y1_index, y2_index, y3_index, y4_index)
#
# print(np.shape(train_x1_2))
# shape = min(np.shape(train_x1_2)[0], np.shape(train_x1_3)[0], np.shape(train_x1_4)[0])
# train_x1_2 = train_x1_2[np.shape(train_x1_2)[0]-shape:]
# train_x1_3 = train_x1_3[np.shape(train_x1_3)[0]-shape:]
# train_x1_4 = train_x1_4[np.shape(train_x1_4)[0]-shape:]
# train_x1 = train_x1[np.shape(train_x1)[0]-shape:]
# train_y1 = train_y1[np.shape(train_y1)[0]-shape:]
#
# print(np.shape(train_x1))
# print(np.shape(train_x1_2))
# print(np.shape(train_x1_3))
# print(np.shape(train_x1_4))
# print(np.shape(train_y1))
# print(shunxu1)
train_x3_1, train_x3_2, train_x3_4, train_x3_3, shunxu3 = form_aux(y3_index, y3_index, y1_index, y2_index, y4_index)

print(np.shape(train_x3_1))
shape = min(np.shape(train_x3_1)[0], np.shape(train_x3_2)[0], np.shape(train_x3_4)[0], np.shape(train_x3_3)[0])
train_x3_1 = train_x3_1[np.shape(train_x3_1)[0] - shape:]
train_x3_2 = train_x3_2[np.shape(train_x3_2)[0] - shape:]
train_x3_4 = train_x3_4[np.shape(train_x3_4)[0] - shape:]
train_x3_3 = train_x3_3[np.shape(train_x3_3)[0] - shape:]
train_x3 = train_x3[np.shape(train_x3)[0] - shape:]
train_y3 = train_y3[np.shape(train_y3)[0] - shape:]

print(np.shape(train_x3))
print(np.shape(train_x3_1))
print(np.shape(train_x3_2))
print(np.shape(train_x3_4))
print(np.shape(train_x3_3))
print(np.shape(train_y3))
print(shunxu3)

train_x4_1, train_x4_2, train_x4_3, train_x4_4, shunxu4 = form_aux(y4_index, y4_index, y1_index, y2_index, y3_index)

print(np.shape(train_x4_1))
shape = min(np.shape(train_x4_1)[0], np.shape(train_x4_2)[0], np.shape(train_x4_3)[0], np.shape(train_x4_4)[0])
train_x4_1 = train_x4_1[np.shape(train_x4_1)[0] - shape:]
train_x4_2 = train_x4_2[np.shape(train_x4_2)[0] - shape:]
train_x4_3 = train_x4_3[np.shape(train_x4_3)[0] - shape:]
train_x4_4 = train_x4_4[np.shape(train_x4_4)[0] - shape:]
train_x4 = train_x4[np.shape(train_x4)[0] - shape:]
train_y4 = train_y4[np.shape(train_y4)[0] - shape:]

print(np.shape(train_x4))
print(np.shape(train_x4_1))
print(np.shape(train_x4_2))
print(np.shape(train_x4_3))
print(np.shape(train_y4))
print(shunxu4)

train_x1 = np.array(train_x1)
# train_x1_2 = np.array(train_x1_2)
# train_x1_3 = np.array(train_x1_3)
# train_x1_4 = np.array(train_x1_4)
train_y1 = np.array(train_y1)
train_x2 = np.array(train_x2)
# train_x2_1 = np.array(train_x2_1)
# train_x2_3 = np.array(train_x2_3)
# train_x2_4 = np.array(train_x2_4)
train_y2 = np.array(train_y2)
train_x3 = np.array(train_x3)
train_x3_1 = np.array(train_x3_1)
train_x3_2 = np.array(train_x3_2)
train_x3_4 = np.array(train_x3_4)
train_x3_3 = np.array(train_x3_3)
train_y3 = np.array(train_y3)
train_x4 = np.array(train_x4)
train_x4_1 = np.array(train_x4_1)
train_x4_2 = np.array(train_x4_2)
train_x4_3 = np.array(train_x4_3)
train_x4_4 = np.array(train_x4_4)
train_y4 = np.array(train_y4)


def form_con(shunxu):
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


con = form_con(shunxu3)

print(con)
print(np.shape(train_x1))
print(np.shape(train_y1))
print(np.shape(train_x2))
print(np.shape(train_x3))
print(np.shape(train_x4))

train_len = len(train_y3)

test_x = []
test_y = []
test_state_data = []

for j in range(len(test_data) - x_len - pred_len):
    test_x.append(test_data[j:j + 1 * x_len, :])
    test_y.append(test_data[j + 1 * x_len, -3:])
    test_state_data.append(state_data[22460 + j + x_len])

test_x = np.array(test_x)
test_x = np.reshape(test_x, (test_x.shape[0], x_len, -1))

test_y = np.array(test_y).reshape(test_x.shape[0], -1)

# test_x1 = []
# test_y1 = []
# test_x2 = []
# test_y2 = []
# test_x3 = []
# test_y3 = []
# test_x4 = []
# test_y4 = []
#
index1 = []
index2 = []
index3 = []
index4 = []
for i in range(len(test_data) - x_len - pred_len):
    if ((test_state_data[i] == 7) or (test_state_data[i] == 5)):
        index1.append(i)
    elif (test_state_data[i] == 6 or test_state_data[i] == 4):
        index2.append(i)
    elif (test_state_data[i] == 3):
        index3.append(i)
    elif (test_state_data[i] == 2):
        index4.append(i)
#
# test_x1 = np.array(test_x1)
# test_y1 = np.array(test_y1)
# test_x2 = np.array(test_x2)
# test_y2 = np.array(test_y2)
# test_x3 = np.array(test_x3)
# test_y3 = np.array(test_y3)
# test_x4 = np.array(test_x4)
# test_y4 = np.array(test_y4)
#
# test_x1 = test_x1[:len(test_x2), :, :]
# test_y1 = test_y1[:len(test_x2), :]

test_x1 = []
test_y1 = []
fuzhu_y1 = []
test_x2 = []
test_y2 = []
fuzhu_y2 = []
test_x3 = []
test_y3 = []
fuzhu_y3 = []
test_x4 = []
test_y4 = []
fuzhu_y4 = []

t1_index = []
t2_index = []
t3_index = []
t4_index = []
for i in range(len(test_data) - x_len - pred_len):
    if ((test_state_data[i] == 7 or test_state_data[i] == 5) and (
            i == 0 or (test_state_data[i - 1] != 7 and test_state_data[i - 1] != 5))):
        test_x1.append(test_x[i])
        t1_index.append(i+len(train_data)+2*x_len)
        j = 1
        test_y_proto = test_y[i]
        fuzhu_y = test_x[i + 1, -1, :]
        while (i + j < len(test_state_data) - 1 and (test_state_data[i + j] == 7 or test_state_data[i + j] == 5)):
            test_y_proto = np.vstack((test_y_proto, test_y[i + j]))
            fuzhu_y = np.vstack((fuzhu_y, test_x[i + 1 + j, -1, :]))
            j = j + 1
        if (np.shape(test_y_proto) == (3,)):
            test_y_proto = test_y_proto.reshape(1, 3)

        if (np.shape(fuzhu_y) == (8,)):
            fuzhu_y = fuzhu_y.reshape(1, 8)
        test_y1.append(test_y_proto.tolist())
        fuzhu_y1.append(fuzhu_y.tolist())

    elif ((test_state_data[i] == 6 or test_state_data[i] == 4) and (
            i == 0 or (test_state_data[i - 1] != 6 and test_state_data[i - 1] != 4))):
        test_x2.append(test_x[i])
        t2_index.append(i+len(train_data)+2*x_len)
        j = 1
        test_y_proto = test_y[i]
        fuzhu_y = test_x[i + 1, -1, :]
        while (i + j < len(test_state_data) - 1 and (test_state_data[i + j] == 6 or test_state_data[i + j] == 4)):
            test_y_proto = np.vstack((test_y_proto, test_y[i + j]))
            fuzhu_y = np.vstack((fuzhu_y, test_x[i + 1 + j, -1, :]))
            j = j + 1
        if (np.shape(test_y_proto) == (3,)):
            test_y_proto = test_y_proto.reshape(1, 3)
        if (np.shape(fuzhu_y) == (8,)):
            fuzhu_y = fuzhu_y.reshape(1, 8)
        test_y2.append(test_y_proto.tolist())
        fuzhu_y2.append(fuzhu_y.tolist())

    elif (test_state_data[i] == 3 and (i == 0 or test_state_data[i - 1] != 3)):
        test_x3.append(test_x[i])
        t3_index.append(i+len(train_data)+2*x_len)
        j = 1
        test_y_proto = test_y[i]
        fuzhu_y = test_x[i + 1, -1, :]
        while (i + j < len(test_state_data) - 1 and test_state_data[i + j] == 3):
            test_y_proto = np.vstack((test_y_proto, test_y[i + j]))
            fuzhu_y = np.vstack((fuzhu_y, test_x[i + 1 + j, -1, :]))
            j = j + 1
        if (np.shape(test_y_proto) == (3,)):
            test_y_proto = test_y_proto.reshape(1, 3)
        if (np.shape(fuzhu_y) == (8,)):
            fuzhu_y = fuzhu_y.reshape(1, 8)
        test_y3.append(test_y_proto.tolist())
        fuzhu_y3.append(fuzhu_y.tolist())

    elif (test_state_data[i] == 2 and (i == 0 or test_state_data[i - 1] != 2)):
        test_x4.append(test_x[i])
        t4_index.append(i+len(train_data)+2*x_len)
        j = 1
        test_y_proto = test_y[i]
        fuzhu_y = test_x[i + 1, -1, :]
        while (i + j < len(test_state_data) - 1 and test_state_data[i + j] == 2):
            test_y_proto = np.vstack((test_y_proto, test_y[i + j]))
            fuzhu_y = np.vstack((fuzhu_y, test_x[i + 1 + j, -1, :]))
            j = j + 1
        if (np.shape(test_y_proto) == (3,)):
            test_y_proto = test_y_proto.reshape(1, 3)
        if (np.shape(fuzhu_y) == (8,)):
            fuzhu_y = fuzhu_y.reshape(1, 8)
        test_y4.append(test_y_proto.tolist())
        fuzhu_y4.append(fuzhu_y.tolist())

for i in range(len(fuzhu_y1)):
    print(i)
    print(np.shape(fuzhu_y1[i]))

print(t2_index)
print(index2)


def form_test_aux(t1_index, index2, index3, index4):
    test_x1_2 = []
    test_x1_3 = []
    test_x1_4 = []
    test1_shunxu = []
    for index in t1_index:
        i2 = 0
        i3 = 0
        i4 = 0
        m = index
        while m >= 0:
            if (m in index2):
                i2 = m
                test_x1_2.append(test_x[m])
                break
            m -= 1
        m = index
        while m >= 0:
            if (m in index3):
                i3 = m
                test_x1_3.append(test_x[m])
                break
            m -= 1
        m = index
        while m >= 0:
            if (m in index4):
                i4 = m
                test_x1_4.append(test_x[m])
                break
            m -= 1

        if (i2 != 0 and i3 != 0 and i4 != 0):
            if (i2 < i3 and i3 < i4):
                test1_shunxu.append(0)
            elif (i2 < i4 and i4 < i3):
                test1_shunxu.append(1)
            elif (i3 < i2 and i2 < i4):
                test1_shunxu.append(2)
            elif (i3 < i4 and i4 < i2):
                test1_shunxu.append(3)
            elif (i4 < i2 and i2 < i3):
                test1_shunxu.append(4)
            elif (i4 < i3 and i3 < i2):
                test1_shunxu.append(5)
    return test_x1_2, test_x1_3, test_x1_4, test1_shunxu


# test_x1_2, test_x1_3, test_x1_4, test1_shunxu = form_test_aux(t1_index, index2, index3, index4)
#
# print(test1_shunxu)
# print(np.shape(test_x1))
# print(np.shape(test_x1_2))
# print(np.shape(test_x1_3))
# print(np.shape(test_x1_4))
#
# shape = min(np.shape(test_x1)[0], np.shape(test_x1_2)[0], np.shape(test_x1_3)[0], np.shape(test_x1_4)[0])
# test_x1 = test_x1[np.shape(test_x1)[0]-shape:]
# test_x1_2 = test_x1_2[np.shape(test_x1_2)[0]-shape:]
# test_x1_3 = test_x1_3[np.shape(test_x1_3)[0]-shape:]
# test_x1_4 = test_x1_4[np.shape(test_x1_4)[0]-shape:]
# test_y1 = test_y1[np.shape(test_y1)[0]-shape:]
# fuzhu_y1 = fuzhu_y1[np.shape(fuzhu_y1)[0]-shape:]
# test1_con = form_con(test1_shunxu)

# test_x2_1, test_x2_3, test_x2_4, test2_shunxu = form_test_aux(t2_index, index1, index3, index4)
#
# print(test2_shunxu)
# print(np.shape(test2_shunxu))
# print(np.shape(test_x2))
# print(np.shape(test_x2_1))
# print(np.shape(test_x2_3))
# print(np.shape(test_x2_4))
#
# shape = min(np.shape(test_x2)[0], np.shape(test_x2_1)[0], np.shape(test_x2_3)[0], np.shape(test_x2_4)[0])
# test_x2 = test_x2[np.shape(test_x2)[0] - shape:]
# test_x2_1 = test_x2_1[np.shape(test_x2_1)[0] - shape:]
# test_x2_3 = test_x2_3[np.shape(test_x2_3)[0] - shape:]
# test_x2_4 = test_x2_4[np.shape(test_x2_4)[0] - shape:]
# test_y2 = test_y2[np.shape(test_y2)[0] - shape:]
# fuzhu_y2 = fuzhu_y2[np.shape(fuzhu_y2)[0] - shape:]
# test2_con = form_con(test2_shunxu)

test_x3_1, test_x3_2, test_x3_4, test_x3_3,test3_shunxu = form_aux(t3_index, y3_index, y1_index, y2_index, y4_index)

print(test3_shunxu)
print(np.shape(test3_shunxu))
print(np.shape(test_x3))
print(np.shape(test_x3_1))
print(np.shape(test_x3_2))
print(np.shape(test_x3_4))

shape = min(np.shape(test_x3)[0], np.shape(test_x3_1)[0], np.shape(test_x3_2)[0], np.shape(test_x3_4)[0])
test_x3 = test_x3[np.shape(test_x3)[0] - shape:]
test_x3_1 = test_x3_1[np.shape(test_x3_1)[0] - shape:]
test_x3_2 = test_x3_2[np.shape(test_x3_2)[0] - shape:]
test_x3_4 = test_x3_4[np.shape(test_x3_4)[0] - shape:]
test_y3 = test_y3[np.shape(test_y3)[0] - shape:]
fuzhu_y3 = fuzhu_y3[np.shape(fuzhu_y3)[0] - shape:]
test3_con = form_con(test3_shunxu)

# test_x4_1, test_x4_2, test_x4_3, test_x4_4, test4_shunxu = form_aux(t4_index, y4_index, y1_index, y2_index, y3_index)
#
# print(test4_shunxu)
# print(np.shape(test4_shunxu))
# print(np.shape(test_x4))
# print(np.shape(test_x4_1))
# print(np.shape(test_x4_2))
# print(np.shape(test_x4_3))
#
# shape = min(np.shape(test_x4)[0], np.shape(test_x4_1)[0], np.shape(test_x4_2)[0], np.shape(test_x4_3)[0])
# test_x4 = test_x4[np.shape(test_x4)[0] - shape:]
# test_x4_1 = test_x4_1[np.shape(test_x4_1)[0] - shape:]
# test_x4_2 = test_x4_2[np.shape(test_x4_2)[0] - shape:]
# test_x4_3 = test_x4_3[np.shape(test_x4_3)[0] - shape:]
# test_y4 = test_y4[np.shape(test_y4)[0] - shape:]
# fuzhu_y4 = fuzhu_y4[np.shape(fuzhu_y4)[0] - shape:]
# test4_con = form_con(test4_shunxu)

test_x1 = np.array(test_x1)
test_y1 = np.array(test_y1)
test_x2 = np.array(test_x2)
test_y2 = np.array(test_y2)
test_x3 = np.array(test_x3)
test_y3 = np.array(test_y3)
test_x4 = np.array(test_x4)
test_y4 = np.array(test_y4)
fuzhu_y1 = np.array(fuzhu_y1)
fuzhu_y2 = np.array(fuzhu_y2)
fuzhu_y3 = np.array(fuzhu_y3)
fuzhu_y4 = np.array(fuzhu_y4)

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

sess = tf.Session()
graph = tf.get_default_graph()

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
z1 = mlp1.forward(h41)

loss = tf.reduce_mean(tf.square(label - z1))

reader = tf.train.NewCheckpointReader('./model_checkpoint_reviced/condition1/wc0.4_wt0.6_bs32/c1')
all_variables = reader.get_variable_to_shape_map()
print(all_variables)
w1 = reader.get_tensor('Variable_113')
w11 = mlp1.weights2
tf.add_to_collection('losses', tf.reduce_sum(0.01*tf.square(w11)))
tf.add_to_collection('losses', tf.reduce_sum(0.01*tf.square(w1-w11)))
tf.add_to_collection('losses', loss)
loss = tf.add_n(tf.get_collection('losses'))


train_op = tf.train.AdamOptimizer(learningrate).minimize(loss)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())

saver1 = tf.train.Saver()
sess.run(init_op)

tf.get_variable_scope().reuse_variables()
saver1.restore(sess, './model_checkpoint_reviced/condition1/wc0.4_wt0.6_bs32/c1')


# train the network
trainloss = []
valloss = []
earstop_patience = 20
ti = 0
minloss = 100000
for i in range(traintimes):
    print("epoch", i)
    train_loss_array = []
    val_loss_array = []

    for j in range(int(train_len*0.8)//batch_size):
        train_loss, con1, _ = sess.run([loss, onet, train_op], feed_dict={x: train_x3[j*batch_size:j*batch_size+batch_size, :, :],
                                      y: train_x3_1[j*batch_size:j*batch_size+batch_size, :, :],
                                      z: train_x3_2[j*batch_size:j*batch_size+batch_size, :, :],
                                      m: train_x3_4[j*batch_size:j*batch_size+batch_size, :, :],
                                      n: train_x3_3[j*batch_size:j*batch_size+batch_size, :, :],
                                      control: con[j*batch_size:j*batch_size+batch_size, :, :],
                                      label:train_y3[j*batch_size:j*batch_size+batch_size, :]})
        train_loss_array.append(train_loss)
    trsize = int(train_len * 0.8)
    train_loss = np.mean(train_loss_array)
    for j in range((train_len - trsize) // batch_size):
        val_loss = sess.run(loss, feed_dict={
            x: train_x3[trsize+j * batch_size:trsize+j * batch_size + batch_size, :, :],
            y: train_x3_1[trsize+j * batch_size:trsize+j * batch_size + batch_size, :, :],
            z: train_x3_2[trsize+j * batch_size:trsize+j * batch_size + batch_size, :, :],
            m: train_x3_4[trsize+j * batch_size:trsize+j * batch_size + batch_size, :, :],
            n: train_x3_3[trsize+j * batch_size:trsize+j * batch_size + batch_size, :, :],
            control: con[trsize+j * batch_size:trsize+j * batch_size + batch_size, :, :],
            label: train_y3[trsize+j * batch_size:trsize+j * batch_size + batch_size, :]
            })
        val_loss_array.append(val_loss)
    val_loss = np.mean(val_loss_array)
    print("train_loss:", train_loss, "val_loss:", val_loss)
    if val_loss <= minloss:
        saver.save(sess, './model_checkpoint_reviced/condition3/wc0.4_wt0.6_c1/c3')
        minloss = val_loss
        ti = 0
    else:
        ti += 1
        if ti>earstop_patience:
            print("early_stopping at epoch", i)
            break
    trainloss.append(train_loss)
    valloss.append(val_loss)

# plot the loss
plt.plot(trainloss)
plt.plot(valloss)
plt.show()

data_writer(trainloss, "./loss保存reviced/condition3/c1_0.01_0.01/trloss.csv")
data_writer(valloss, "./loss保存reviced/condition3/c1_0.01_0.01/valloss.csv")

#     #training set
#     train_index_rmse = []
#     train_index_mape = []
#     train_index_mae = []
#     for i in range(train_size):
#         result = []
#         train_result = []
#         t1 = sess.run(z1, feed_dict={x: data_fill(traindata0[:, i], batch_size),
#                                      y: data_fill(traindata1[:, i], batch_size),
#                                      z: data_fill(traindata2[:, i], batch_size)})
#         result.append(t1[:, 0])
#         for j in range(x_input_width):
#             temp1 = float(result[0][j])
#             temp2 = temp1 * np.std(data0) + np.mean(data0)
#             train_result.append(temp2)
#         train_contrast = []
#         temp1 = trainlabel[:,i+1]
#         temp1 = temp1.tolist()
#         for i in range(x_input_width):
#             temp2 = temp1[i][0] * np.std(data0) + np.mean(data0)
#             train_contrast.append(temp2)
#         a = rmse(train_contrast, train_result)
#         b = mape(train_contrast, train_result)
#         c = mae(train_contrast, train_result)
#         train_index_rmse.append(a)
#         train_index_mape.append(b)
#         train_index_mae.append(c)
#     train_index_rmse_mean=get_median(train_index_rmse)
#     train_index_mape_mean=get_median(train_index_mape)
#     train_index_mae_mean=get_median(train_index_mae)
#
#     #validation set
#     verification_index_rmse = []
#     verification_index_mape = []
#     verification_index_mae = []
#     for i in range(verification_size):
#         result = []
#         verification_result = []
#         t1 = sess.run(z1, feed_dict={x: data_fill(traindata0[:, i + train_size], batch_size),
#                                      y: data_fill(traindata1[:, i + train_size], batch_size),
#                                      z: data_fill(traindata2[:, i + train_size], batch_size)})
#         result.append(t1[:, 0])
#         for j in range(x_input_width):
#             temp1 = float(result[0][j])
#             temp2 = temp1 * np.std(data0) + np.mean(data0)
#             verification_result.append(temp2)
#         verification_contrast = []
#         temp1 = trainlabel[:, train_size + i+1]
#         temp1 = temp1.tolist()
#         for i in range(x_input_width):
#             temp2 = temp1[i][0] * np.std(data0) + np.mean(data0)
#             verification_contrast.append(temp2)
#         a = rmse(verification_contrast, verification_result)
#         b = mape(verification_contrast, verification_result)
#         c = mae(verification_contrast, verification_result)
#         verification_index_rmse.append(a)
#         verification_index_mape.append(b)
#         verification_index_mae.append(c)
#     verification_index_rmse_mean=get_median(verification_index_rmse)
#     verification_index_mape_mean=get_median(verification_index_mape)
#     verification_index_mae_mean=get_median(verification_index_mae)
#
#     #testing set
#     test_index_rmse = []
#     test_index_mape = []
#     test_index_mae = []
#     for i in range(test_size):
#         result = []
#         test_result = []
#         t1 = sess.run(z1, feed_dict={x: data_fill(traindata0[:, i + train_size + verification_size], batch_size),
#                                      y: data_fill(traindata1[:, i + train_size + verification_size], batch_size),
#                                      z: data_fill(traindata2[:, i + train_size + verification_size], batch_size)})
#         result.append(t1[:, 0])
#         for j in range(x_input_width):
#             temp1 = float(result[0][j])
#             temp2 = temp1 * np.std(data0) + np.mean(data0)
#             test_result.append(temp2)
#         test_contrast = []
#         temp1 = trainlabel[:, train_size + verification_size + i+1]
#         temp1 = temp1.tolist()
#         for i in range(x_input_width):
#             temp2 = temp1[i][0] * np.std(data0) + np.mean(data0)
#             test_contrast.append(temp2)
#         a = rmse(test_contrast, test_result)
#         b = mape(test_contrast, test_result)
#         c = mae(test_contrast, test_result)
#         test_index_rmse.append(a)
#         test_index_mape.append(b)
#         test_index_mae.append(c)
#     test_index_rmse_mean=get_median(test_index_rmse)
#     test_index_mape_mean=get_median(test_index_mape)
#     test_index_mae_mean=get_median(test_index_mae)
#
#     print('train_index_rmse_mean=',train_index_rmse_mean)
#     print('train_index_mape_mean=',train_index_mape_mean)
#     print('train_index_mae_mean=',train_index_mae_mean)
#     print('verification_index_rmse_mean=',verification_index_rmse_mean)
#     print('verification_index_mape_mean=',verification_index_mape_mean)
#     print('verification_index_mae_mean=',verification_index_mae_mean)
#     print('test_index_rmse_mean=',test_index_rmse_mean)
#     print('test_index_mape_mean=',test_index_mape_mean)
#     print('test_index_mae_mean=',test_index_mae_mean)

# saver.restore(sess, './model_checkpoint/condition4/c4')
# predict_y3 = []
# for i in range(len(test_y3)):
#     print("i =", i)
#     pred = []
#     y1 = []
#     test_x_in = test_x3[i].reshape((-1, x_len, 8))
#     for j in range(len(test_y3[i])):
#         pred_y1 = sess.run([z1], feed_dict={x: data_fill(test_x_in, batch_size),
#                                             y: data_fill(test_x3_1[i].reshape((-1, x_len, 8)), batch_size),
#                                             z: data_fill(test_x3_2[i].reshape((-1, x_len, 8)), batch_size),
#                                             m: data_fill(test_x3_4[i].reshape((-1, x_len, 8)), batch_size),
#                                             control: data_fill(test3_con[i].reshape((-1, 6, state_width)),
#                                                                batch_size)})
#         # print(np.shape(pred_y1))
#         # print(np.shape(pred_y1[0:1]))
#         test_x_in[:, :-1, :] = test_x_in[:, 1:, :]
#         test_x_in[:, -1, -3:] = pred_y1[0][0:1]
#         test_x_in[0, -1, :-3] = fuzhu_y3[i][j][:-3]
#
#         pred.append(pred_y1[0][0:1])
#     predict_y3.append(np.array(pred).reshape(len(test_y3[i]), 3).tolist())
#
# print(predict_y3)
# # predict_y3 = np.array(predict_y3)
# for i in range(len(test_y3)):
#     test_y3[i] = test_y3[i] * std_value[-3:] + mean_value[-3:]
# # true_y1 = test_y1 * std_value[-3:] + mean_value[-3:]
# for i in range(len(test_y3)):
#     predict_y3[i] = predict_y3[i] * std_value[-3:] + mean_value[-3:]
#
# for i in range(len(predict_y3)):
#     plt.plot(predict_y3[i][:, 0])
#     plt.plot(test_y3[i][:, 0])
#     plt.show()
#
# predict_y3 = np.array(predict_y3)
# print("测试集多点预测")
# rmse_sum = 0
# for i in range(len(test_y3)):
#     rmse_sum += np.sqrt(metrics.mean_squared_error(test_y3[i], predict_y3[i]))
# print(rmse_sum / len(test_y3))
#
# mae_sum = 0
# for i in range(len(test_y3)):
#     mae_sum += metrics.mean_absolute_error(test_y3[i], predict_y3[i])
# print(mae_sum / len(test_y3))
#
# mape_sum = 0
# for i in range(len(test_y3)):
#     mape_sum += metrics.mean_absolute_percentage_error(test_y3[i], predict_y3[i])
# print(mape_sum / len(test_y3))

saver.restore(sess, './model_checkpoint_reviced/condition3/wc0.4_wt0.6_c1/c3')
predict_y3 = []
for i in range(len(test_y3)):
    print("i =", i)
    pred = []
    y1 = []
    test_x_in = test_x3[i].reshape((-1, x_len, 8))
    for j in range(len(test_y3[i])):
        pred_y1 = sess.run([z1], feed_dict={x: data_fill(test_x_in, batch_size),
                                            y: data_fill(test_x3_1[i].reshape((-1, x_len, 8)), batch_size),
                                            z: data_fill(test_x3_2[i].reshape((-1, x_len, 8)), batch_size),
                                            m: data_fill(test_x3_4[i].reshape((-1, x_len, 8)), batch_size),
                                            n: data_fill(test_x3_3[i].reshape((-1, x_len, 8)), batch_size),
                                            control: data_fill(test3_con[i].reshape((-1, 9, state_width)),
                                                               batch_size)})
        # print(np.shape(pred_y1))
        # print(np.shape(pred_y1[0:1]))
        test_x_in[:, :-1, :] = test_x_in[:, 1:, :]
        test_x_in[:, -1, -3:] = pred_y1[0][0:1]
        test_x_in[0, -1, :-3] = fuzhu_y3[i][j][:-3]

        pred.append(pred_y1[0][0:1])
    predict_y3.append(np.array(pred).reshape(len(test_y3[i]), 3).tolist())

print(predict_y3)
# predict_y4 = np.array(predict_y4)
for i in range(len(test_y3)):
    test_y3[i] = test_y3[i] * std_value[-3:] + mean_value[-3:]
# true_y1 = test_y1 * std_value[-4:] + mean_value[-4:]
for i in range(len(test_y3)):
    predict_y3[i] = predict_y3[i] * std_value[-3:] + mean_value[-3:]

for i in range(len(predict_y3)):
    plt.plot(predict_y3[i][:, 0])
    plt.plot(test_y3[i][:, 0])
    plt.show()

predict_y3 = np.array(predict_y3)
print("测试集多点预测")
rmse_sum = 0
for i in range(len(test_y3)):
    rmse_sum += np.sqrt(metrics.mean_squared_error(test_y3[i], predict_y3[i]))
print(rmse_sum / len(test_y3))

mae_sum = 0
for i in range(len(test_y3)):
    mae_sum += metrics.mean_absolute_error(test_y3[i], predict_y3[i])
print(mae_sum / len(test_y3))

mape_sum = 0
for i in range(len(test_y3)):
    mape_sum += metrics.mean_absolute_percentage_error(test_y3[i], predict_y3[i])
print(mape_sum / len(test_y3))

endtime = datetime.datetime.now()
print('TC', endtime - starttime)