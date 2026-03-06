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
learningrate = 0.0001
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
space_weight = 0.6
time_weight = 0.8
space_weight2 = 0.6
time_weight2 = 0.4


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
trsize = int(train_len*0.8)
val_x3 = train_x3[trsize:]
val_y3 = train_y3[trsize:]
print("y3_index")
print(y3_index[trsize:])

val_x3_duod=[]
val_con3=[]
val_x3_1_duod=[]
val_x3_2_duod=[]
val_x3_3_duod=[]
val_x3_4_duod=[]
val_y3_duod=[]
fuzhu_vy3=[]
for i in range(len(val_y3)):
    if(i==0 or y3_index[i+trsize]-y3_index[i+trsize-1]!=1):
        val_x3_duod.append(train_x3[i+trsize])
        val_x3_1_duod.append(train_x3_1[i+trsize])
        val_x3_2_duod.append(train_x3_2[i+trsize])
        val_x3_3_duod.append(train_x3_3[i+trsize])
        val_x3_4_duod.append(train_x3_4[i+trsize])
        val_con3.append(con[i+trsize])
        val_y_proto = val_y3[i]
        fuzhu_y = val_x3[i + 1, -1, :]
        j = 1
        while(i + j < len(val_y3) - 1 and y3_index[i+trsize+j]-y3_index[i+trsize+j-1]==1):
            val_y_proto = np.vstack((val_y_proto, val_y3[i + j]))
            fuzhu_y = np.vstack((fuzhu_y, val_x3[i + 1 + j, -1, :]))
            j = j + 1
        if (np.shape(val_y_proto) == (3,)):
            val_y_proto = val_y_proto.reshape(1, 3)

        if (np.shape(fuzhu_y) == (8,)):
            fuzhu_y = fuzhu_y.reshape(1, 8)
        val_y3_duod.append(val_y_proto.tolist())
        fuzhu_vy3.append(fuzhu_y.tolist())

print(np.shape(val_y3_duod))
print(np.shape(val_x3_duod))
for i in range(len(val_y3_duod)):
    print(i)
    print(np.shape(val_y3_duod[i]))

fuzhu_vy3 = np.array(fuzhu_vy3)

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

index1 = np.hstack((y1_index, index1))
index2 = np.hstack((y2_index, index2))
index3 = np.hstack((y3_index, index3))
index4 = np.hstack((y4_index, index4))

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

# for i in range(len(fuzhu_y1)):
#     print(i)
#     print(np.shape(fuzhu_y1[i]))

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

test_x3_1, test_x3_2, test_x3_4, test_x3_3,test3_shunxu = form_aux(t3_index, index3, index1, index2, index4)

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

reader = tf.train.NewCheckpointReader('./model_checkpoint_reviced/condition1/wc0.6_wt0.8_bs32/c1')
reader1 = tf.train.NewCheckpointReader('./model_checkpoint_reviced/condition2/wc0.6_wt0.4/c2')
all_variables = reader.get_variable_to_shape_map()
print(all_variables)
w1 = reader1.get_tensor('Variable_113')
w11 = mlp1.weights2
tf.add_to_collection('losses', tf.reduce_sum(0.01*tf.square(w11)))
tf.add_to_collection('losses', tf.reduce_sum(0.001*tf.square(w1-w11)))
tf.add_to_collection('losses', loss)
loss = tf.add_n(tf.get_collection('losses'))


train_op = tf.train.AdamOptimizer(learningrate).minimize(loss)
init_op = tf.global_variables_initializer()



l1_gg = ['l1_gate_g:0', 'l1_gate_g_1:0', 'l1_gate_g_2:0', 'l1_gate_g_3:0', 'l1_gate_g_4:0', 'l1_gate_g_5:0',
          'l1_gate_g_6:0', 'l1_gate_g_7:0', 'l1_gate_g_8:0', 'l1_gate_g_9:0', 'l1_gate_g_10:0', 'l1_gate_g_11:0',
          'l1_gate_g_12:0', 'l1_gate_g_13:0', 'l1_gate_g_14:0', 'l1_gate_g_15:0', 'l1_gate_g_16:0', 'l1_gate_g_17:0',
          'l1_gate_g_18:0', 'l1_gate_g_19:0', 'l1_gate_g_20:0', 'l1_gate_g_21:0', 'l1_gate_g_22:0', 'l1_gate_g_23:0',
          'l1_gate_g_24:0', 'l1_gate_g_25:0', 'l1_gate_g_26:0', 'l1_gate_g_27:0', 'l1_gate_g_28:0', 'l1_gate_g_29:0',
          'l1_gate_g_30:0', 'l1_gate_g_31:0', 'l1_gate_g_32:0', 'l1_gate_g_33:0', 'l1_gate_g_34:0', 'l1_gate_g_35:0']

l1_gh = ['l1_gate_h:0', 'l1_gate_h_1:0', 'l1_gate_h_2:0', 'l1_gate_h_3:0', 'l1_gate_h_4:0', 'l1_gate_h_5:0',
          'l1_gate_h_6:0', 'l1_gate_h_7:0', 'l1_gate_h_8:0', 'l1_gate_h_9:0', 'l1_gate_h_10:0', 'l1_gate_h_11:0',
          'l1_gate_h_12:0', 'l1_gate_h_13:0', 'l1_gate_h_14:0', 'l1_gate_h_15:0', 'l1_gate_h_16:0', 'l1_gate_h_17:0',
          'l1_gate_h_18:0', 'l1_gate_h_19:0', 'l1_gate_h_20:0', 'l1_gate_h_21:0', 'l1_gate_h_22:0', 'l1_gate_h_23:0',
          'l1_gate_h_24:0', 'l1_gate_h_25:0', 'l1_gate_h_26:0', 'l1_gate_h_27:0', 'l1_gate_h_28:0', 'l1_gate_h_29:0',
          'l1_gate_h_30:0', 'l1_gate_h_31:0', 'l1_gate_h_32:0', 'l1_gate_h_33:0', 'l1_gate_h_34:0', 'l1_gate_h_35:0']

l2_gg = ['l2_gate_g:0', 'l2_gate_g_1:0', 'l2_gate_g_2:0', 'l2_gate_g_3:0', 'l2_gate_g_4:0', 'l2_gate_g_5:0',
          'l2_gate_g_6:0', 'l2_gate_g_7:0', 'l2_gate_g_8:0', 'l2_gate_g_9:0', 'l2_gate_g_10:0', 'l2_gate_g_11:0',
          'l2_gate_g_12:0', 'l2_gate_g_13:0', 'l2_gate_g_14:0', 'l2_gate_g_15:0', 'l2_gate_g_16:0', 'l2_gate_g_17:0',
          'l2_gate_g_18:0', 'l2_gate_g_19:0', 'l2_gate_g_20:0', 'l2_gate_g_21:0', 'l2_gate_g_22:0', 'l2_gate_g_23:0',
          'l2_gate_g_24:0', 'l2_gate_g_25:0', 'l2_gate_g_26:0', 'l2_gate_g_27:0', 'l2_gate_g_28:0', 'l2_gate_g_29:0',
          'l2_gate_g_30:0', 'l2_gate_g_31:0', 'l2_gate_g_32:0', 'l2_gate_g_33:0', 'l2_gate_g_34:0', 'l2_gate_g_35:0',
          'l2_gate_g_36:0', 'l2_gate_g_37:0', 'l2_gate_g_38:0', 'l2_gate_g_39:0', 'l2_gate_g_40:0', 'l2_gate_g_41:0',
          'l2_gate_g_42:0', 'l2_gate_g_43:0', 'l2_gate_g_44:0', 'l2_gate_g_45:0', 'l2_gate_g_46:0', 'l2_gate_g_47:0',
          'l2_gate_g_48:0', 'l2_gate_g_49:0', 'l2_gate_g_50:0', 'l2_gate_g_51:0', 'l2_gate_g_52:0', 'l2_gate_g_53:0',
          'l2_gate_g_54:0', 'l2_gate_g_55:0', 'l2_gate_g_56:0', 'l2_gate_g_57:0', 'l2_gate_g_58:0', 'l2_gate_g_59:0',
         'l2_gate_g_60:0', 'l2_gate_g_61:0', 'l2_gate_g_62:0']

l2_gh = ['l2_gate_h:0', 'l2_gate_h_1:0', 'l2_gate_h_2', 'l2_gate_h_3', 'l2_gate_h_4', 'l2_gate_h_5',
          'l2_gate_h_6', 'l2_gate_h_7:0', 'l2_gate_h_8:0', 'l2_gate_h_9:0', 'l2_gate_h_10:0', 'l2_gate_h_11:0',
          'l2_gate_h_12:0', 'l2_gate_h_13:0', 'l2_gate_h_14:0', 'l2_gate_h_15:0', 'l2_gate_h_16:0', 'l2_gate_h_17:0',
          'l2_gate_h_18:0', 'l2_gate_h_19:0', 'l2_gate_h_20:0', 'l2_gate_h_21:0', 'l2_gate_h_22:0', 'l2_gate_h_23:0',
          'l2_gate_h_24:0', 'l2_gate_h_25:0', 'l2_gate_h_26:0', 'l2_gate_h_27:0', 'l2_gate_h_28:0', 'l2_gate_h_29:0',
          'l2_gate_h_30:0', 'l2_gate_h_31:0', 'l2_gate_h_32:0', 'l2_gate_h_33:0', 'l2_gate_h_34:0', 'l2_gate_h_35:0',
          'l2_gate_h_36:0', 'l2_gate_h_37:0', 'l2_gate_h_38:0', 'l2_gate_h_39:0', 'l2_gate_h_40:0', 'l2_gate_h_41:0',
          'l2_gate_h_42:0', 'l2_gate_h_43:0', 'l2_gate_h_44:0', 'l2_gate_h_45:0', 'l2_gate_h_46:0', 'l2_gate_h_47:0',
          'l2_gate_h_48:0', 'l2_gate_h_49:0', 'l2_gate_h_50:0', 'l2_gate_h_51:0', 'l2_gate_h_52:0', 'l2_gate_h_53:0',
          'l2_gate_h_54:0', 'l2_gate_h_55:0', 'l2_gate_h_56:0', 'l2_gate_h_57:0', 'l2_gate_h_58:0', 'l2_gate_h_59:0',
         'l2_gate_h_60:0', 'l2_gate_h_61:0', 'l2_gate_h_62:0']

# 试错
# c1_l1_gg = ['l1_gate_g_24', 'l1_gate_g_25', 'l1_gate_g_26', 'l1_gate_g_27', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             '0', '0', '0', '0', 'l1_gate_g_8', 'l1_gate_g_9',
#             'l1_gate_g_10', 'l1_gate_g_11', 'l1_gate_g_12', 'l1_gate_g_13', 'l1_gate_g_14', 'l1_gate_g_15',
#             '0', '0', '0', '0', 'l1_gate_g_4', 'l1_gate_g_5',
#             'l1_gate_g_6', 'l1_gate_g_7', 'l1_gate_g_16', 'l1_gate_g_17', 'l1_gate_g_18', 'l1_gate_g_19']
#
# c1_l1_gh = ['l1_gate_h_24', 'l1_gate_h_25', 'l1_gate_h_26', 'l1_gate_h_27', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             '0', '0', '0', '0', 'l1_gate_h_8', 'l1_gate_h_9',
#             'l1_gate_h_10', 'l1_gate_h_11', 'l1_gate_h_12', 'l1_gate_h_13', 'l1_gate_h_14', 'l1_gate_h_15',
#             '0', '0', '0', '0', 'l1_gate_h_4', 'l1_gate_h_5',
#             'l1_gate_h_6', 'l1_gate_h_7', 'l1_gate_h_16', 'l1_gate_h_17', 'l1_gate_h_18', 'l1_gate_h_19']
#
# c1_l2_gg = ['l2_gate_g_42', 'l2_gate_g_43', 'l2_gate_g_44', 'l2_gate_g_45', 'l2_gate_g_46', 'l2_gate_g_47',
#             'l2_gate_g_48', '0', '0', '0', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             '0', '0', '0', '0', 'l2_gate_g_14', 'l2_gate_g_15',
#             'l2_gate_g_16', 'l2_gate_g_17', 'l2_gate_g_18', 'l2_gate_g_19', 'l2_gate_g_20', 'l2_gate_g_21',
#             'l2_gate_g_22', 'l2_gate_g_23', 'l2_gate_g_24', 'l2_gate_g_25', 'l2_gate_g_26', 'l2_gate_g_27',
#             '0', '0', '0', '0', '0', '0',
#             '0', 'l2_gate_g_7', 'l2_gate_g_8', 'l2_gate_g_9', 'l2_gate_g_10', 'l2_gate_g_11',
#             'l2_gate_g_12', 'l2_gate_g_13', 'l2_gate_g_28', 'l2_gate_g_29', 'l2_gate_g_30', 'l2_gate_g_31',
#             'l2_gate_g_32', 'l2_gate_g_33', 'l2_gate_g_34']
#
# c1_l2_gh = ['l2_gate_h_42', 'l2_gate_h_43', 'l2_gate_h_44', 'l2_gate_h_45', 'l2_gate_h_46', 'l2_gate_h_47',
#             'l2_gate_h_48', '0', '0', '0', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             '0', '0', '0', '0', 'l2_gate_h_14', 'l2_gate_h_15',
#             'l2_gate_h_16', 'l2_gate_h_17', 'l2_gate_h_18', 'l2_gate_h_19', 'l2_gate_h_20', 'l2_gate_h_21',
#             'l2_gate_h_22', 'l2_gate_h_23', 'l2_gate_h_24', 'l2_gate_h_25', 'l2_gate_h_26', 'l2_gate_h_27',
#             '0', '0', '0', '0', '0', '0',
#             '0', 'l2_gate_h_7', 'l2_gate_h_8', 'l2_gate_h_9', 'l2_gate_h_10', 'l2_gate_h_11',
#             'l2_gate_h_12', 'l2_gate_h_13', 'l2_gate_h_28', 'l2_gate_h_29', 'l2_gate_h_30', 'l2_gate_h_31',
#             'l2_gate_h_32', 'l2_gate_h_33', 'l2_gate_h_34']
#
# c2_l1_gg = ['0', '0', '0', '0', 'l1_gate_g_24', 'l1_gate_g_25',
#             'l1_gate_g_26', 'l1_gate_g_27', 'l1_gate_g_8', 'l1_gate_g_9', 'l1_gate_g_10', 'l1_gate_g_11',
#             'l1_gate_g_12', 'l1_gate_g_13', 'l1_gate_g_14', 'l1_gate_g_15', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             'l1_gate_g_4', 'l1_gate_g_5', 'l1_gate_g_6', 'l1_gate_g_7', '0', '0',
#             '0', '0', '0', '0', '0', '0']
#
# c2_l1_gh = ['0', '0', '0', '0', 'l1_gate_h_24', 'l1_gate_h_25',
#             'l1_gate_h_26', 'l1_gate_h_27', 'l1_gate_h_8', 'l1_gate_h_9', 'l1_gate_h_10', 'l1_gate_h_11',
#             'l1_gate_h_12', 'l1_gate_h_13', 'l1_gate_h_14', 'l1_gate_h_15', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             'l1_gate_h_4', 'l1_gate_h_5', 'l1_gate_h_6', 'l1_gate_h_7', '0', '0',
#             '0', '0', '0', '0', '0', '0']
#
# c2_l2_gg = ['0', '0', '0', '0', '0', '0',
#             '0', 'l2_gate_g_42', 'l2_gate_g_43', 'l2_gate_g_44', 'l2_gate_g_45', 'l2_gate_g_46',
#             'l2_gate_g_47', 'l2_gate_g_48', 'l2_gate_g_14', 'l2_gate_g_15', 'l2_gate_g_16', 'l2_gate_g_17',
#             'l2_gate_g_18', 'l2_gate_g_19', 'l2_gate_g_20', 'l2_gate_g_21', 'l2_gate_g_22', 'l2_gate_g_23',
#             'l2_gate_g_24', 'l2_gate_g_25', 'l2_gate_g_26', 'l2_gate_g_27', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             'l2_gate_g_7', 'l2_gate_g_8', 'l2_gate_g_9', 'l2_gate_g_10', 'l2_gate_g_11', 'l2_gate_g_12',
#             'l2_gate_g_13', '0', '0', '0', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             '0', '0', '0']
#
# c2_l2_gh = ['0', '0', '0', '0', '0', '0',
#             '0', 'l2_gate_h_42', 'l2_gate_h_43', 'l2_gate_h_44', 'l2_gate_h_45', 'l2_gate_h_46',
#             'l2_gate_h_47', 'l2_gate_h_48', 'l2_gate_h_14', 'l2_gate_h_15', 'l2_gate_h_16', 'l2_gate_h_17',
#             'l2_gate_h_18', 'l2_gate_h_19', 'l2_gate_h_20', 'l2_gate_h_21', 'l2_gate_h_22', 'l2_gate_h_23',
#             'l2_gate_h_24', 'l2_gate_h_25', 'l2_gate_h_26', 'l2_gate_h_27', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             'l2_gate_h_7', 'l2_gate_h_8', 'l2_gate_h_9', 'l2_gate_h_10', 'l2_gate_h_11', 'l2_gate_h_12',
#             'l2_gate_h_13', '0', '0', '0', '0', '0',
#             '0', '0', '0', '0', '0', '0',
#             '0', '0', '0']

# 其余随机
c1_l1_gg = ['l1_gate_g_24', 'l1_gate_g_25', 'l1_gate_g_26', 'l1_gate_g_27', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', 'l1_gate_g_12', 'l1_gate_g_13', 'l1_gate_g_14', 'l1_gate_g_15',
            '0', '0', '0', '0', 'l1_gate_g_4', 'l1_gate_g_5',
            'l1_gate_g_6', 'l1_gate_g_7', '0', '0', '0', '0']

c1_l1_gh = ['l1_gate_h_24', 'l1_gate_h_25', 'l1_gate_h_26', 'l1_gate_h_27', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', 'l1_gate_h_12', 'l1_gate_h_13', 'l1_gate_h_14', 'l1_gate_h_15',
            '0', '0', '0', '0', 'l1_gate_h_4', 'l1_gate_h_5',
            'l1_gate_h_6', 'l1_gate_h_7', '0', '0', '0', '0']

c1_l2_gg = ['l2_gate_g_42', 'l2_gate_g_43', 'l2_gate_g_44', 'l2_gate_g_45', 'l2_gate_g_46', 'l2_gate_g_47',
            'l2_gate_g_48', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', 'l2_gate_g_21',
            'l2_gate_g_22', 'l2_gate_g_23', 'l2_gate_g_24', 'l2_gate_g_25', 'l2_gate_g_26', 'l2_gate_g_27',
            '0', '0', '0', '0', '0', '0',
            '0', 'l2_gate_g_7', 'l2_gate_g_8', 'l2_gate_g_9', 'l2_gate_g_10', 'l2_gate_g_11',
            'l2_gate_g_12', 'l2_gate_g_13', '0', '0', '0', '0',
            '0', '0', '0']

c1_l2_gh = ['l2_gate_h_42', 'l2_gate_h_43', 'l2_gate_h_44', 'l2_gate_h_45', 'l2_gate_h_46', 'l2_gate_h_47',
            'l2_gate_h_48', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', 'l2_gate_h_21',
            'l2_gate_h_22', 'l2_gate_h_23', 'l2_gate_h_24', 'l2_gate_h_25', 'l2_gate_h_26', 'l2_gate_h_27',
            '0', '0', '0', '0', '0', '0',
            '0', 'l2_gate_h_7', 'l2_gate_h_8', 'l2_gate_h_9', 'l2_gate_h_10', 'l2_gate_h_11',
            'l2_gate_h_12', 'l2_gate_h_13', '0', '0', '0', '0',
            '0', '0', '0']

c2_l1_gg = ['0', '0', '0', '0', 'l1_gate_g_24', 'l1_gate_g_25',
            'l1_gate_g_26', 'l1_gate_g_27', 'l1_gate_g_8', 'l1_gate_g_9', 'l1_gate_g_10', 'l1_gate_g_11',
            'l1_gate_g_12', 'l1_gate_g_13', 'l1_gate_g_14', 'l1_gate_g_15', '0', '0',
            '0', '0', '0', '0', '0', '0',
            'l1_gate_g_4', 'l1_gate_g_5', 'l1_gate_g_6', 'l1_gate_g_7', '0', '0',
            '0', '0', '0', '0', '0', '0']

c2_l1_gh = ['0', '0', '0', '0', 'l1_gate_h_24', 'l1_gate_h_25',
            'l1_gate_h_26', 'l1_gate_h_27', '0', '0', '0', '0',
            'l1_gate_h_12', 'l1_gate_h_13', 'l1_gate_h_14', 'l1_gate_h_15', '0', '0',
            '0', '0', '0', '0', '0', '0',
            'l1_gate_h_4', 'l1_gate_h_5', 'l1_gate_h_6', 'l1_gate_h_7', '0', '0',
            '0', '0', '0', '0', '0', '0']

c2_l2_gg = ['0', '0', '0', '0', '0', '0',
            '0', 'l2_gate_g_42', 'l2_gate_g_43', 'l2_gate_g_44', 'l2_gate_g_45', 'l2_gate_g_46',
            'l2_gate_g_47', 'l2_gate_g_48', '0', '0', '0', '0',
            '0', '0', '0', 'l2_gate_g_21', 'l2_gate_g_22', 'l2_gate_g_23',
            'l2_gate_g_24', 'l2_gate_g_25', 'l2_gate_g_26', 'l2_gate_g_27', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0',
            'l2_gate_g_7', 'l2_gate_g_8', 'l2_gate_g_9', 'l2_gate_g_10', 'l2_gate_g_11', 'l2_gate_g_12',
            'l2_gate_g_13', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', '0']

c2_l2_gh = ['0', '0', '0', '0', '0', '0',
            '0', 'l2_gate_h_42', 'l2_gate_h_43', 'l2_gate_h_44', 'l2_gate_h_45', 'l2_gate_h_46',
            'l2_gate_h_47', 'l2_gate_h_48', '0', '0', '0', '0',
            '0', '0', '0', 'l2_gate_h_21', 'l2_gate_h_22', 'l2_gate_h_23',
            'l2_gate_h_24', 'l2_gate_h_25', 'l2_gate_h_26', 'l2_gate_h_27', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0',
            'l2_gate_h_7', 'l2_gate_h_8', 'l2_gate_h_9', 'l2_gate_h_10', 'l2_gate_h_11', 'l2_gate_h_12',
            'l2_gate_h_13', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0',
            '0', '0', '0']


# wg0 = reader.get_tensor('l1_gate_g')
# wg1 = reader.get_tensor('l1_gate_g1')
# wg2 = reader.get_tensor('l1_gate_g2')
# wg3 = reader.get_tensor('l1_gate_g3')
# wg4 = reader.get_tensor('l1_gate_g4')
# wg5 = reader.get_tensor('l1_gate_g5')
# wg6 = reader.get_tensor('l1_gate_g6')
# wg7 = reader.get_tensor('l1_gate_g7')
# wg8 = reader.get_tensor('l1_gate_g8')
# wg9 = reader.get_tensor('l1_gate_g9')
# wg10 = reader.get_tensor('l1_gate_g10')
# wg11 = reader.get_tensor('l1_gate_g11')
# wg12 = reader.get_tensor('l1_gate_g12')
# wg13 = reader.get_tensor('l1_gate_g13')
# wg14 = reader.get_tensor('l1_gate_g14')
# wg15 = reader.get_tensor('l1_gate_g15')
# wg16 = reader.get_tensor('l1_gate_g16')
# wg17 = reader.get_tensor('l1_gate_g17')
# wg18 = reader.get_tensor('l1_gate_g18')
# wg19 = reader.get_tensor('l1_gate_g19')
# wg20 = reader.get_tensor('l1_gate_g20')
# wg21 = reader.get_tensor('l1_gate_g21')
# wg22 = reader.get_tensor('l1_gate_g22')
# wg23 = reader.get_tensor('l1_gate_g23')
# wg24 = reader.get_tensor('l1_gate_g24')
# wg25 = reader.get_tensor('l1_gate_g25')
# wg26 = reader.get_tensor('l1_gate_g26')
# wg27 = reader.get_tensor('l1_gate_g27')
# wg28 = reader.get_tensor('l1_gate_g28')
# wg29 = reader.get_tensor('l1_gate_g29')
# wg31 = reader.get_tensor('l1_gate_g31')
# wg32 = reader.get_tensor('l1_gate_g32')
# wg33 = reader.get_tensor('l1_gate_g33')
# wg34 = reader.get_tensor('l1_gate_g34')
# wg35 = reader.get_tensor('l1_gate_g35')
# wh0 = reader.get_tensor('l1_gate_h')
# wh1 = reader.get_tensor('l1_gate_h1')
# wh2 = reader.get_tensor('l1_gate_h2')
# wh3 = reader.get_tensor('l1_gate_h3')
# wh4 = reader.get_tensor('l1_gate_h4')
# wh5 = reader.get_tensor('l1_gate_h5')
# wh6 = reader.get_tensor('l1_gate_h6')
# wh7 = reader.get_tensor('l1_gate_h7')
# wh8 = reader.get_tensor('l1_gate_h8')
# wh9 = reader.get_tensor('l1_gate_h9')
# wh10 = reader.get_tensor('l1_gate_h10')
# wh11 = reader.get_tensor('l1_gate_h11')
# wh12 = reader.get_tensor('l1_gate_h12')
# wh13 = reader.get_tensor('l1_gate_h13')
# wh14 = reader.get_tensor('l1_gate_h14')
# wh15 = reader.get_tensor('l1_gate_h15')
# wh16 = reader.get_tensor('l1_gate_h16')
# wh17 = reader.get_tensor('l1_gate_h17')
# wh18 = reader.get_tensor('l1_gate_h18')
# wh19 = reader.get_tensor('l1_gate_h19')
# wh20 = reader.get_tensor('l1_gate_h20')
# wh21 = reader.get_tensor('l1_gate_h21')
# wh22 = reader.get_tensor('l1_gate_h22')
# wh23 = reader.get_tensor('l1_gate_h23')
# wh24 = reader.get_tensor('l1_gate_h24')
# wh25 = reader.get_tensor('l1_gate_h25')
# wh26 = reader.get_tensor('l1_gate_h26')
# wh27 = reader.get_tensor('l1_gate_h27')
# wh28 = reader.get_tensor('l1_gate_h28')
# wh29 = reader.get_tensor('l1_gate_h29')
# wh31 = reader.get_tensor('l1_gate_h31')
# wh32 = reader.get_tensor('l1_gate_h32')
# wh33 = reader.get_tensor('l1_gate_h33')
# wh34 = reader.get_tensor('l1_gate_h34')
# wh35 = reader.get_tensor('l1_gate_h35')



saver = tf.train.Saver(tf.global_variables())

saver1 = tf.train.Saver()
sess.run(init_op)

tf.get_variable_scope().reuse_variables()
saver1.restore(sess, './model_checkpoint_reviced/condition2/wc0.6_wt0.4/c2')

trainable_variables = tf.trainable_variables()

for v in trainable_variables:
    if v.name in l1_gg:
        i = l1_gg.index(v.name)
        if(c1_l1_gg[i]!='0'):
            sess.run(tf.assign(v, reader.get_tensor(c1_l1_gg[i])))
        elif(c2_l1_gg[i]!='0'):
            sess.run(tf.assign(v, reader1.get_tensor(c2_l1_gg[i])))
        else:
            sess.run(tf.assign(v, tf.random_normal([x_input_width, state_width], stddev=0.1)))
    elif v.name in l1_gh:
        i = l1_gh.index(v.name)
        if(c1_l1_gh[i]!='0'):
            sess.run(tf.assign(v, reader.get_tensor(c1_l1_gh[i])))
        elif(c2_l1_gh[i]!='0'):
            sess.run(tf.assign(v, reader1.get_tensor(c2_l1_gh[i])))
        else:
            sess.run(tf.assign(v, tf.random_normal([state_width, state_width], stddev=0.1)))

    elif v.name in l2_gh:
        i = l2_gh.index(v.name)
        if(c1_l2_gh[i]!='0'):
            sess.run(tf.assign(v, reader.get_tensor(c1_l2_gh[i])))
        elif(c2_l2_gh[i]!='0'):
            sess.run(tf.assign(v, reader1.get_tensor(c2_l2_gh[i])))
        else:
            sess.run(tf.assign(v, tf.random_normal([state_width, state_width], stddev=0.1)))

    elif v.name in l2_gg:
        i = l2_gg.index(v.name)
        if(c1_l2_gg[i]!='0'):
            sess.run(tf.assign(v, reader.get_tensor(c1_l2_gg[i])))
        elif(c2_l2_gg[i]!='0'):
            sess.run(tf.assign(v, reader1.get_tensor(c2_l2_gg[i])))
        else:
            sess.run(tf.assign(v, tf.random_normal([x_input_width, state_width], stddev=0.1)))


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
        saver.save(sess, './model_checkpoint_reviced/newtr_condition3/wc0.4_wt0.6_c2_suijichushi/c3')
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

data_writer(trainloss, "./loss保存reviced/newtr_condition3/wc0.4_wt0.6_c2_suijichushi/trloss.csv")
data_writer(valloss, "./loss保存reviced/newtr_condition3/wc0.4_wt0.6_c2_suijichushi/valloss.csv")


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

saver.restore(sess, './model_checkpoint_reviced/newtr_condition3/wc0.4_wt0.6_c2_suijichushi/c3')

predict_y3 = []
for i in range(len(val_y3_duod)):
    print("i =", i)
    pred = []
    y1 = []
    test_x_in = val_x3_duod[i].reshape((-1, x_len, 8))
    for j in range(len(val_y3_duod[i])):
        pred_y1 = sess.run([z1], feed_dict={x: data_fill(test_x_in, batch_size),
                                            y: data_fill(val_x3_1_duod[i].reshape((-1, x_len, 8)), batch_size),
                                            z: data_fill(val_x3_2_duod[i].reshape((-1, x_len, 8)), batch_size),
                                            m: data_fill(val_x3_4_duod[i].reshape((-1, x_len, 8)), batch_size),
                                            n: data_fill(val_x3_3_duod[i].reshape((-1, x_len, 8)), batch_size),
                                            control: data_fill(val_con3[i].reshape((-1, 9, state_width)),
                                                               batch_size)})
        # print(np.shape(pred_y1))
        # print(np.shape(pred_y1[0:1]))
        test_x_in[:, :-1, :] = test_x_in[:, 1:, :]
        test_x_in[:, -1, -3:] = pred_y1[0][0:1]
        test_x_in[0, -1, :-3] = fuzhu_vy3[i][j][:-3]

        pred.append(pred_y1[0][0:1])
    predict_y3.append(np.array(pred).reshape(len(val_y3_duod[i]), 3).tolist())

print(predict_y3)
# predict_y4 = np.array(predict_y4)
for i in range(len(val_y3_duod)):
    val_y3_duod[i] = val_y3_duod[i] * std_value[-3:] + mean_value[-3:]
# true_y1 = test_y1 * std_value[-4:] + mean_value[-4:]
for i in range(len(val_y3_duod)):
    predict_y3[i] = predict_y3[i] * std_value[-3:] + mean_value[-3:]

for i in range(len(predict_y3)):
    plt.plot(predict_y3[i][:, 0])
    plt.plot(val_y3_duod[i][:, 0])
    plt.show()

predict_y3 = np.array(predict_y3)
print("验证集多点预测")
rmse_sum = 0
for i in range(len(val_y3_duod)):
    rmse_sum += np.sqrt(metrics.mean_squared_error(val_y3_duod[i], predict_y3[i]))
print(rmse_sum / len(val_y3_duod))

mae_sum = 0
for i in range(len(val_y3_duod)):
    mae_sum += metrics.mean_absolute_error(val_y3_duod[i], predict_y3[i])
print(mae_sum / len(val_y3_duod))

mape_sum = 0
for i in range(len(val_y3_duod)):
    mape_sum += metrics.mean_absolute_percentage_error(val_y3_duod[i], predict_y3[i])
print(mape_sum / len(val_y3_duod))


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