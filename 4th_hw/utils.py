#coding:utf-8
import numpy as np
import matplotlib as plt
import os
#sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))
#数据加载函数
def load_datasets(data_path):
    #data_path = "/Users/xiaohu/Documents/Pycharm_server/152/Pattern-recognition-homework/4th_hw/data.txt"
    X = []
    Y = []

    with open(data_path,"r") as files:
        for line in files:
            data = []
            label = np.array([0,0,0])
            for data_str in line.rstrip().split(","):
                data.append(float(data_str))
            data = np.array(data)
            X.append(data[:-1])
            label[data[-1]-1] = 1
            Y.append(label)
    X = np.array(X).T
    Y = np.array(Y).T
    for i in range(0,3):
        X[:,10*i:10*(i+1)] = X[:,10*i:10*(i+1)] - np.mean(X[:,10*i:10*(i+1)], axis=0)
    return X,Y
#网络参数初始化函数
def para_init(X, Y, h_layer):
    n_input = X.shape[0]
    n_hidden = h_layer
    n_output = Y.shape[0]
    np.random.seed(1)
    W1 = np.random.randn(n_hidden, n_input)
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(n_output, n_hidden)
    b2 = np.zeros((n_output, 1))
    para = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    return para
#前向传播
def forward_propagation(x, para):
    W1 = para['W1']
    b1 = para['b1']
    W2 = para['W2']
    b2 = para['b2']

    net_h = np.dot(W1,x)+b1
    #print(np.dot(W1,x).shape)
    y_h = np.tanh(net_h)
    net_o = np.dot(W2,y_h)+b2
    y_o = sigmoid(net_o)

    cache = {"net_h": net_h,"y_h": y_h,"net_o": net_o,"y_o": y_o}

    return y_o,cache
#计算误差
def computer_cost(y_o, y):
    return sum((y_o-y)**2)/2
#反向传播
def backward_propagation(y, x, cache, para):
    net_h = cache['net_h']
    y_h = cache['y_h']
    net_o = cache['net_o']
    y_o = cache['y_o']

    W1 = para['W1']
    b1 = para['b1']
    W2 = para['W2']
    b2 = para['b2']

    dy_o = y_o-y
    dnet_o = dy_o*y_o*(1-y_o)
    #dnet_o = dy_o
    dW2 = np.dot(dnet_o, y_h.T)
    db2 = np.sum(dnet_o, axis=1, keepdims=True)
    dy_h = np.dot(W2.T,dnet_o)
    #print(dy_h.shape)
    dnet_h = dy_h*(1-y_h**2)
    #print(dnet_h.shape)
    dW1 = np.dot(dnet_h,x.T)
    db1 = np.sum(dnet_h, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads
#参数更新
def updata_para(para, grads, learning_rate):
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = para['W1']
    b1 = para['b1']
    W2 = para['W2']
    b2 = para['b2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    para = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return para
#测试函数
def test_model(x,para):
    W1 = para['W1']
    b1 = para['b1']
    W2 = para['W2']
    b2 = para['b2']

    net_h = np.dot(W1, x) + b1
    # print(np.dot(W1,x).shape)
    y_h = np.tanh(net_h)
    net_o = np.dot(W2, y_h) + b2
    y_o = sigmoid(net_o)

    cache = {"net_h": net_h, "y_h": y_h, "net_o": net_o, "y_o": y_o}

    return y_o, cache