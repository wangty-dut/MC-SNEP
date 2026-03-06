import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import math
import xlwt
import datetime
from LSTMONE import Lstm1
from LSTMTWO import Lstm2
from LSTMTHREE import Lstm31,Lstm32,Lstm33
from MLP import mlp
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from Data_writer import data_writer

tf.compat.v1.disable_eager_execution()
starttime = datetime.datetime.now()
np.random.seed(5)
tf.set_random_seed(5)


batch_size=64
learningrate=0.001
x_input_width=8
y_input_width=8
z_input_width=8
m_input_width=8
state_width=64

output_width1 = 32
# train_size=300
# verification_size=100
# test_size=100
traintimes=100
space_weight=0.6
time_weight=0.4



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
    mape = sum / n *100
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
    median = (data[size//2]+data[size//2-1])/2
    data[0] = median
   if size % 2 == 1:
    median = data[(size-1)//2]
    data[0] = median
   return data[0]

def data_add(data,num):
    if num==1:
        data_compose = data
    elif num==2:
        data_compose = np.concatenate((data, data), axis=1)
    elif num>2:
        data_compose = np.concatenate((data, data), axis=1)
        for i in range(num-2):
            data_compose=np.concatenate((data_compose,data),axis=1)
    return data_compose

def data_fill(data,batchsize):
    a=np.zeros([batchsize-1, data.shape[1], data.shape[2]])
    b=np.concatenate((data,a),axis=0)
    return b

def data_write(file_path, datas, col):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # establish sheet
    # write data
    i=0
    for data in datas:
        sheet1.write(i,col,data)
        i=i+1
    f.save(file_path)  # save doc

df = pd.read_excel('./汇总.xlsx', header=0, usecols=[1, 2, 3, 4, 6, 10, 11, 12])
rawdata = df.values# X

state_data = pd.read_excel('./汇总.xlsx', header=0, usecols=[16]).values
print(np.shape(rawdata))

x_len = 60
pred_len = 60
output_width = 3
#train_data1 = rawdata[]
train_data = rawdata[:22400]
test_data = rawdata[22460:28000]
print(train_data)
max_value = np.max(train_data)
min_value = np.min(train_data)
print("max_value = ", max_value)
print("min_value=", min_value)
#mean_value = np.mean(train_data)
mean_value = train_data.mean(0)
print(mean_value)

std_value = train_data.std(0)
print(std_value)
#train_data = wavelet_denoising(train_data)
#test_data = wavelet_denoising(test_data)
train_data = (train_data-mean_value)/std_value

test_data = (test_data-mean_value)/std_value


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

train_size = int((len(train_data)-x_len-pred_len))

print(len(train_data))
for i in range(train_size):
    if((state_data[i+x_len]==7) or (state_data[i+x_len]==5)):
        train_x1.append(train_data[i: i+x_len, :])
        train_y1.append(train_data[i+x_len, -3:])
        y1_index.append(i+x_len)
    elif(state_data[i+x_len]==6 or state_data[i+x_len]==4):
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

def form_aux(y1_index, y2_index, y3_index, y4_index):
    train_x1_2=[]
    train_x1_3=[]
    train_x1_4=[]
    shunxu = []
    for index in y1_index:
        i2 = 0
        i3 = 0
        i4 = 0
        m = index
        while m >=0:
            if(m in y2_index):
                i2 = m
                train_x1_2.append(train_data[max(0, m-x_len): m])
                break
            m -= 1
        m = index
        while m >=0:
            if(m in y3_index):
                i3 = m
                train_x1_3.append(train_data[max(0, m-x_len): m])
                break
            m -= 1
        m = index
        while m >=0:
            if(m in y4_index):
                i4 = m
                train_x1_4.append(train_data[max(0, m-x_len): m])
                break
            m -= 1
        if(i2!=0 and i3!=0 and i4!=0):
            if(i2<i3 and i3<i4):
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

train_x1_2, train_x1_3, train_x1_4, shunxu1 = form_aux(y1_index, y2_index, y3_index, y4_index)

print(np.shape(train_x1_2))
shape = min(np.shape(train_x1_2)[0], np.shape(train_x1_3)[0], np.shape(train_x1_4)[0])
train_x1_2 = train_x1_2[np.shape(train_x1_2)[0]-shape:]
train_x1_3 = train_x1_3[np.shape(train_x1_3)[0]-shape:]
train_x1_4 = train_x1_4[np.shape(train_x1_4)[0]-shape:]
train_x1 = train_x1[np.shape(train_x1)[0]-shape:]
train_y1 = train_y1[np.shape(train_y1)[0]-shape:]

print(np.shape(train_x1))
print(np.shape(train_x1_2))
print(np.shape(train_x1_3))
print(np.shape(train_x1_4))
print(np.shape(train_y1))
print(shunxu1)

# train_x2_1, train_x2_3, train_x2_4, shunxu2 = form_aux(y2_index, y1_index, y3_index, y4_index)
#
# print(np.shape(train_x2_1))
# shape = min(np.shape(train_x2_1)[0], np.shape(train_x2_3)[0], np.shape(train_x2_4)[0])
# train_x2_1 = train_x2_1[np.shape(train_x2_1)[0]-shape:]
# train_x2_3 = train_x2_3[np.shape(train_x2_3)[0]-shape:]
# train_x2_4 = train_x2_4[np.shape(train_x2_4)[0]-shape:]
# train_x2 = train_x2[np.shape(train_x2)[0]-shape:]
# train_y2 = train_y2[np.shape(train_y2)[0]-shape:]
#
# print(np.shape(train_x2))
# print(np.shape(train_x2_1))
# print(np.shape(train_x2_3))
# print(np.shape(train_x2_4))
# print(np.shape(train_y2))
# print(shunxu2)


train_x1 = np.array(train_x1)
train_x1_2 = np.array(train_x1_2)
train_x1_3 = np.array(train_x1_3)
train_x1_4 = np.array(train_x1_4)
train_y1 = np.array(train_y1)
train_x2 = np.array(train_x2)
# train_x2_1 = np.array(train_x2_1)
# train_x2_3 = np.array(train_x2_3)
# train_x2_4 = np.array(train_x2_4)
train_y2 = np.array(train_y2)
train_x3 = np.array(train_x3)
train_y3 = np.array(train_y3)
train_x4 = np.array(train_x4)
train_y4 = np.array(train_y4)

def form_con(shunxu):
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

con = form_con(shunxu1)

print(con)
print(np.shape(train_x1))
print(np.shape(train_y1))
print(np.shape(train_x2))
print(np.shape(train_x3))
print(np.shape(train_x4))

train_len = len(train_y2)

test_x = []
test_y = []
test_state_data = []

for j in range(len(test_data)-x_len-pred_len):
    test_x.append(test_data[j:j+1*x_len, :])
    test_y.append(test_data[j+1*x_len, -3:])
    test_state_data.append(state_data[22460+j+x_len])

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
for i in range(len(test_data)-x_len-pred_len):
    if((test_state_data[i]==7) or (test_state_data[i]==5)):
        index1.append(i)
    elif(test_state_data[i]==6 or test_state_data[i]==4):
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
for i in range(len(test_data)-x_len-pred_len):
    if((test_state_data[i]==7 or test_state_data[i]==5) and (i==0 or (test_state_data[i-1]!=7 and test_state_data[i-1]!=5))):
        test_x1.append(test_x[i])
        t1_index.append(i)
        j=1
        test_y_proto = test_y[i]
        fuzhu_y = test_x[i + 1, -1, :]
        while(i+j < len(test_state_data)-1 and(test_state_data[i+j]==7 or test_state_data[i+j]==5)):
            test_y_proto = np.vstack((test_y_proto, test_y[i+j]))
            fuzhu_y = np.vstack((fuzhu_y, test_x[i+1+j, -1, :]))
            j = j+1
        if(np.shape(test_y_proto) == (3, )):
            test_y_proto = test_y_proto.reshape(1, 3)

        if (np.shape(fuzhu_y) == (8,)):
            fuzhu_y = fuzhu_y.reshape(1, 8)
        test_y1.append(test_y_proto.tolist())
        fuzhu_y1.append(fuzhu_y.tolist())

    elif((test_state_data[i]==6 or test_state_data[i]==4) and (i==0 or (test_state_data[i-1]!=6 and test_state_data[i-1]!=4))):
        test_x2.append(test_x[i])
        t2_index.append(i)
        j = 1
        test_y_proto = test_y[i]
        fuzhu_y = test_x[i + 1, -1, :]
        while (i+j < len(test_state_data)-1 and (test_state_data[i + j] == 6 or test_state_data[i + j] == 4)):
            test_y_proto = np.vstack((test_y_proto, test_y[i + j]))
            fuzhu_y = np.vstack((fuzhu_y, test_x[i + 1 + j, -1, :]))
            j = j + 1
        if (np.shape(test_y_proto) == (3, )):
            test_y_proto = test_y_proto.reshape(1, 3)
        if (np.shape(fuzhu_y) == (8,)):
            fuzhu_y = fuzhu_y.reshape(1, 8)
        test_y2.append(test_y_proto.tolist())
        fuzhu_y2.append(fuzhu_y.tolist())

    elif (test_state_data[i] == 3 and (i==0 or test_state_data[i-1]!=3)):
        test_x3.append(test_x[i])
        t3_index.append(i)
        j = 1
        test_y_proto = test_y[i]
        fuzhu_y = test_x[i + 1, -1, :]
        while (i+j < len(test_state_data)-1 and test_state_data[i + j] == 3):
            test_y_proto = np.vstack((test_y_proto, test_y[i + j]))
            fuzhu_y = np.vstack((fuzhu_y, test_x[i + 1 + j, -1, :]))
            j = j + 1
        if (np.shape(test_y_proto) == (3, )):
            test_y_proto = test_y_proto.reshape(1, 3)
        if (np.shape(fuzhu_y) == (8,)):
            fuzhu_y = fuzhu_y.reshape(1, 8)
        test_y3.append(test_y_proto.tolist())
        fuzhu_y3.append(fuzhu_y.tolist())

    elif (test_state_data[i] == 2 and (i==0 or test_state_data[i-1]!=2)):
        test_x4.append(test_x[i])
        t4_index.append(i)
        j = 1
        test_y_proto = test_y[i]
        fuzhu_y = test_x[i + 1, -1, :]
        while(i+j < len(test_state_data)-1 and test_state_data[i+j]==2):
            test_y_proto = np.vstack((test_y_proto, test_y[i+j]))
            fuzhu_y = np.vstack((fuzhu_y, test_x[i + 1 + j, -1, :]))
            j = j + 1
        if (np.shape(test_y_proto) == (3, )):
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
    test_x1_2=[]
    test_x1_3=[]
    test_x1_4=[]
    test1_shunxu = []
    for index in t1_index:
        i2 = 0
        i3 = 0
        i4 = 0
        m = index
        while m >= 0:
            if(m in index2):
                i2 = m
                test_x1_2.append(test_x[m])
                break
            m -= 1
        m = index
        while m >= 0:
            if(m in index3):
                i3 = m
                test_x1_3.append(test_x[m])
                break
            m -= 1
        m = index
        while m >= 0:
            if(m in index4):
                i4 = m
                test_x1_4.append(test_x[m])
                break
            m -= 1
    
        if(i2!=0 and i3!=0 and i4!=0):
            if(i2<i3 and i3<i4):
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

test_x1_2, test_x1_3, test_x1_4, test1_shunxu = form_test_aux(t1_index, index2, index3, index4)

print(test1_shunxu)
print(np.shape(test_x1))
print(np.shape(test_x1_2))
print(np.shape(test_x1_3))
print(np.shape(test_x1_4))

shape = min(np.shape(test_x1)[0], np.shape(test_x1_2)[0], np.shape(test_x1_3)[0], np.shape(test_x1_4)[0])
test_x1 = test_x1[np.shape(test_x1)[0]-shape:]
test_x1_2 = test_x1_2[np.shape(test_x1_2)[0]-shape:]
test_x1_3 = test_x1_3[np.shape(test_x1_3)[0]-shape:]
test_x1_4 = test_x1_4[np.shape(test_x1_4)[0]-shape:]
test_y1 = test_y1[np.shape(test_y1)[0]-shape:]
fuzhu_y1 = fuzhu_y1[np.shape(fuzhu_y1)[0]-shape:]
test1_con = form_con(test1_shunxu)

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
# test_x2 = test_x2[np.shape(test_x2)[0]-shape:]
# test_x2_1 = test_x2_1[np.shape(test_x2_1)[0]-shape:]
# test_x2_3 = test_x2_3[np.shape(test_x2_3)[0]-shape:]
# test_x2_4 = test_x2_4[np.shape(test_x2_4)[0]-shape:]
# test_y2 = test_y2[np.shape(test_y2)[0]-shape:]
# fuzhu_y2 = fuzhu_y2[np.shape(fuzhu_y2)[0]-shape:]
# test2_con = form_con(test2_shunxu)
# 
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

#1first layer
lstm11=Lstm1(z_input_width,state_width,batch_size,)
# lstm12=Lstm1(state_width,state_width,batch_size)
# lstm13=Lstm1(state_width,state_width,batch_size)
#1second layer
lstm21=Lstm2(y_input_width,state_width,batch_size,time_weight,space_weight)
# lstm22=Lstm2(state_width,state_width,batch_size,time_weight,space_weight)
# lstm23=Lstm2(state_width,state_width,batch_size,time_weight,space_weight)
#1third layer
lstm31=Lstm2(m_input_width,state_width,batch_size,time_weight,space_weight)
# lstm32=Lstm2(state_width,state_width,batch_size,time_weight,space_weight)
# lstm33=Lstm2(state_width,state_width,batch_size,time_weight,space_weight)
#forth layer
lstm41=Lstm31(x_input_width,state_width,batch_size,time_weight,space_weight)
# lstm42=Lstm32(state_width,state_width,batch_size,time_weight,space_weight)
# lstm43=Lstm33(state_width,state_width,batch_size,time_weight,space_weight)
#2third layer
lstm51=Lstm2(m_input_width,state_width,batch_size,time_weight,space_weight)
# lstm52=Lstm2(state_width,state_width,batch_size,time_weight,space_weight)
# lstm53=Lstm2(state_width,state_width,batch_size,time_weight,space_weight)
#2second layer
lstm61=Lstm2(z_input_width,state_width,batch_size,time_weight,space_weight)
# lstm62=Lstm2(state_width,state_width,batch_size,time_weight,space_weight)
# lstm63=Lstm2(state_width,state_width,batch_size,time_weight,space_weight)
#2first layer
lstm71=Lstm1(y_input_width,state_width,batch_size)
# lstm72=Lstm1(state_width,state_width,batch_size)
# lstm73=Lstm1(state_width,state_width,batch_size)

#3third layer
lstm81 = Lstm2(m_input_width,state_width,batch_size,time_weight,space_weight)
#3second layer
lstm91=Lstm2(z_input_width,state_width,batch_size,time_weight,space_weight)
#3first layer
lstm101=Lstm1(y_input_width,state_width,batch_size)

#4third layer
lstm111 = Lstm2(m_input_width,state_width,batch_size,time_weight,space_weight)
#4second layer
lstm121=Lstm2(z_input_width,state_width,batch_size,time_weight,space_weight)
#4first layer
lstm131=Lstm1(y_input_width,state_width,batch_size)

#5third layer
lstm141 = Lstm2(m_input_width,state_width,batch_size,time_weight,space_weight)
#5second layer
lstm151=Lstm2(z_input_width,state_width,batch_size,time_weight,space_weight)
#5first layer
lstm161=Lstm1(y_input_width,state_width,batch_size)

#6third layer
lstm171 = Lstm2(m_input_width,state_width,batch_size,time_weight,space_weight)
#6second layer
lstm181=Lstm2(z_input_width,state_width,batch_size,time_weight,space_weight)
#6first layer
lstm191=Lstm1(y_input_width,state_width,batch_size)

#MLP
mlp1=mlp(output_width,state_width, output_width1, batch_size)


# print(s==tf.Tensor(1))

x = tf.placeholder(tf.float32,[batch_size, x_len, x_input_width])
y = tf.placeholder(tf.float32,[batch_size, x_len, y_input_width])
m = tf.placeholder(tf.float32,[batch_size, x_len, m_input_width])
z = tf.placeholder(tf.float32,[batch_size, x_len, z_input_width])
control = tf.placeholder(tf.float32,[batch_size, 6, state_width])
label=tf.placeholder(tf.float32,[batch_size, 3])

h11=lstm11.forward(y)
# h12=lstm12.forward(h11)
# h13=lstm13.forward(h12)
h21=lstm21.forward(h11,z)
# h22=lstm22.forward(h12,h21)
# h23=lstm23.forward(h13,h22)
h31=lstm31.forward(h21, m)
h71=lstm71.forward(y)
# h52=lstm52.forward(h51)
# h53=lstm53.forward(h52)
h61=lstm61.forward(h71,m)
h51=lstm51.forward(h61,z)

h101=lstm101.forward(z)
h91=lstm91.forward(h101, y)
h81=lstm81.forward(h91, m)

h131=lstm131.forward(z)
h121=lstm121.forward(h131, m)
h111=lstm111.forward(h121, y)

h161=lstm161.forward(m)
h151=lstm151.forward(h161, y)
h141=lstm141.forward(h151, z)

h191=lstm191.forward(m)
h181=lstm181.forward(h161, z)
h171=lstm171.forward(h151, y)
# h42=lstm42.forward(h52,h41)
# h43=lstm43.forward(h53,h42)
h41, onet = lstm41.forward(h31,h51, h81, h111, h141, h171, x, control)
# h32=lstm32.forward(h42,h22,h31)
# h33=lstm33.forward(h43,h23,h32)
z1 = mlp1.forward(h41)



loss=tf.reduce_mean(tf.square(label-z1))
train_op = tf.train.AdamOptimizer(learningrate).minimize(loss)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    sess.run(init_op)
    index_rmse0=[]
    index_mape0=[]
    index_mae0=[]
    index_rmse1=[]
    index_mape1=[]
    index_mae1=[]
    index_rmse2=[]
    index_mape2=[]
    index_mae2=[]
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
            train_loss, con1, _ = sess.run([loss, onet, train_op], feed_dict={x: train_x1[j*batch_size:j*batch_size+batch_size, :, :],
                                          y: train_x1_2[j*batch_size:j*batch_size+batch_size, :, :],
                                          z: train_x1_3[j*batch_size:j*batch_size+batch_size, :, :],
                                          m: train_x1_4[j*batch_size:j*batch_size+batch_size, :, :],
                                          control: con[j*batch_size:j*batch_size+batch_size, :, :],
                                          label:train_y1[j*batch_size:j*batch_size+batch_size, :]})
            train_loss_array.append(train_loss)
        trsize = int(train_len * 0.8)
        train_loss = np.mean(train_loss_array)
        for j in range((train_len - trsize) // batch_size):
            val_loss = sess.run(loss, feed_dict={
                x: train_x1[trsize+j * batch_size:trsize+j * batch_size + batch_size, :, :],
                y: train_x1_2[trsize+j * batch_size:trsize+j * batch_size + batch_size, :, :],
                z: train_x1_3[trsize+j * batch_size:trsize+j * batch_size + batch_size, :, :],
                m: train_x1_4[trsize+j * batch_size:trsize+j * batch_size + batch_size, :, :],
                control: con[trsize+j * batch_size:trsize+j * batch_size + batch_size, :, :],
                label: train_y1[trsize+j * batch_size:trsize+j * batch_size + batch_size, :]
                })
            val_loss_array.append(val_loss)
        val_loss = np.mean(val_loss_array)
        print("train_loss:", train_loss, "val_loss:", val_loss)
        if val_loss <= minloss:
            saver.save(sess, './model_checkpoint/condition1_wc0.6_wt0.4/c1')
            minloss = val_loss
            ti = 0
        else:
            ti += 1
            if ti>earstop_patience:
                print("early_stopping at epoch", i)
                break
        trainloss.append(train_loss)
        valloss.append(val_loss)
    plt.plot(trainloss)
    plt.plot(valloss)
    plt.show()
    data_writer(trainloss, './loss保存/condition1/wc_0.6_wt_0.4/trloss.csv')
    data_writer(valloss, './loss保存/condition1/wc_0.6_wt_0.4/valloss.csv')

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

    saver.restore(sess, './model_checkpoint/condition1_wc0.6_wt0.4/c1')
    predict_y1 = []
    for i in range(20):
        print("i =", i)
        pred = []
        y1 = []
        test_x_in = test_x1[3 * i].reshape((-1, x_len, 8))
        for j in range(len(test_y1[i * 3])):
            pred_y1 = sess.run([z1], feed_dict={x: data_fill(test_x_in, batch_size),
                                      y: data_fill(test_x1_2[i * 3].reshape((-1, x_len, 8)), batch_size),
                                      z: data_fill(test_x1_3[i*3].reshape((-1, x_len, 8)), batch_size),
                                      m: data_fill(test_x1_4[i*3].reshape((-1, x_len, 8)), batch_size),
                                      control: data_fill(test1_con[i*3].reshape((-1, 6, state_width)), batch_size)})
            # print(np.shape(pred_y1))
            # print(np.shape(pred_y1[0:1]))
            test_x_in[:, :-1, :] = test_x_in[:, 1:, :]
            test_x_in[:, -1, -3:] = pred_y1[0][0:1]
            test_x_in[0, -1, :-3] = fuzhu_y1[i * 3][j][:-3]

            pred.append(pred_y1[0][0:1])
        predict_y1.append(np.array(pred).reshape(len(test_y1[i * 3]), 3).tolist())

    print(predict_y1)
    # predict_y3 = np.array(predict_y3)
    for i in range(len(test_y1)):
        test_y1[i] = test_y1[i] * std_value[-3:] + mean_value[-3:]
    # true_y1 = test_y1 * std_value[-3:] + mean_value[-3:]
    for i in range(20):
        predict_y1[i] = predict_y1[i] * std_value[-3:] + mean_value[-3:]

    for i in range(20):
        plt.plot(predict_y1[i][:, 0])
        plt.plot(test_y1[i * 3][:, 0])
        plt.show()

    predict_y1 = np.array(predict_y1)
    print("测试集多点预测")
    rmse_sum = 0
    for i in range(20):
        rmse_sum += np.sqrt(metrics.mean_squared_error(test_y1[i * 3], predict_y1[i]))
    print(rmse_sum / 20)

    mae_sum = 0
    for i in range(20):
        mae_sum += metrics.mean_absolute_error(test_y1[i * 3], predict_y1[i])
    print(mae_sum / 20)

    mape_sum = 0
    for i in range(20):
        mape_sum += metrics.mean_absolute_percentage_error(test_y1[i * 3], predict_y1[i])
    print(mape_sum / 20)
endtime = datetime.datetime.now()
print('TC',endtime - starttime)