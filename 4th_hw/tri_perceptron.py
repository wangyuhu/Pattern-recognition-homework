#coding:utf-8
import numpy as np
import random
import matplotlib.pyplot as plt

from utils import load_datasets,para_init,forward_propagation,computer_cost,backward_propagation,updata_para,test_model

def model(h_layer, batch_size=8, num_iters = 1000, print_cost=True, learning_rate=0.1,decay_rate=1,print_test=True):

#数据准备
    data_path = "/Users/xiaohu/Documents/Pycharm_server/152/Pattern-recognition-homework/4th_hw/data.txt"
    X, Y = load_datasets(data_path)

    # print({"X":X})
    # print({"Y":Y})
    # print(X.shape)
    # print(Y.shape)

    # train_X = np.hstack((X[:,2:10],X[:,12:20],X[:,22:30]))
    # train_Y = np.hstack((Y[:,2:10],Y[:,12:20],Y[:,22:30]))
    # test_X = np.hstack((X[:,0:2],X[:,10:12],X[:,20:22]))
    # test_Y = np.hstack((Y[:,0:2],Y[:,10:12],Y[:,20:22]))

    train_X = np.hstack((X[:,0:8],X[:,10:18],X[:,20:28]))
    train_Y = np.hstack((Y[:,0:8],Y[:,10:18],Y[:,20:28]))
    test_X = np.hstack((X[:,8:10],X[:,18:20],X[:,28:30]))
    test_Y = np.hstack((Y[:,8:10],Y[:,18:20],Y[:,28:30]))
    #print(train_Y)
    print(train_X.shape)
    print(test_X.shape)
    print(train_Y.shape)
    print(test_Y.shape)
#参数初始化
    para = para_init(X, Y, h_layer)
    print(para['W1'].shape)
    print(para['b1'].shape)
    print(para['W2'].shape)
    print(para['b2'].shape)
#模型训练
    cost_array = []
    max_right_num = [0,0]
    label = np.array([0.0, 0.333333333, 0.666666666])
    for i in range(0, num_iters):
        mean_cost = 0
        mean_grads = {"dW1": 0,"db1": 0,"dW2": 0,"db2": 0}
        train_index = random.sample(range(0,24), 24)
        train_index = np.reshape(np.array(train_index), [24, 1])

        #for j in train_index:
        for index, j in enumerate(train_index):
            x = np.reshape(train_X[:, j], [3, 1])
            # print(x.shape)
            y_o, cache = forward_propagation(x, para)
            cost = computer_cost(y_o, train_Y[:,j])
            mean_cost = mean_cost + cost
            grads = backward_propagation(train_Y[:,j], x, cache, para)
            #print(train_Y[:,j])
            for key in mean_grads:
                mean_grads[key] = mean_grads[key] + grads[key]
            if(not (index+1) % batch_size or (index+1) == train_X.shape[1]):
                batch_num = batch_size
                if((index+1) == train_X.shape[1]):
                    batch_num = batch_size - train_X.shape[1] % batch_size
                for key in mean_grads:
                    mean_grads[key] = mean_grads[key] / batch_num
                para = updata_para(para, mean_grads, learning_rate)
                mean_grads = {"dW1": 0, "db1": 0, "dW2": 0, "db2": 0}
        if (not i % (num_iters / 4)):
            learning_rate = learning_rate / decay_rate
        if (not i % 10):
            cost_array.append(mean_cost / train_X.shape[1])
            if (print_cost):
                print({'cost': mean_cost / train_X.shape[1]})

            right_num = 0
            error_num = 0
            for j in range(0,test_X.shape[1]):
                x = np.reshape(test_X[:, j], [3, 1])
                y_o, cache = forward_propagation(x, para)
                #print(y_o,test_Y[:,j])
                test_label = np.array([0,0,0])
                test_label[np.where(y_o == np.max(y_o))[0][0]]=1
                # print(test_label[0][0]==int(Y[j]*3))
                if ((test_label == test_Y[:,j]).all()):
                    right_num = right_num + 1
                else:
                    error_num = error_num + 1
        if (right_num > max_right_num[0]):
            #print(i)
            max_right_num[0] = right_num
            max_right_num[1] = i
            max_para = para

            # print(right_num)
#输出最优精度
    if (print_test):
        right_num = 0
        error_num = 0
        for j in range(0, test_X.shape[1]):
            x = np.reshape(test_X[:, j], [3, 1])
            y_o, cache = forward_propagation(x, max_para)
            test_label = np.array([0, 0, 0])
            test_label[np.where(y_o == np.max(y_o))[0][0]] = 1
            print(test_label, test_Y[:, j])
            # test_label = np.where(abs(label - y_o[0][0]) == np.min(abs(label - y_o[0][0])))
            # # print(test_label[0][0]==int(Y[j]*3))
            # if (test_label[0][0] == int((test_Y[j] - 0.05) * 3)):
            #     right_num = right_num + 1
            # else:
            #     error_num = error_num + 1
        print({"accuracy": float(max_right_num[0]) / 6})
        #print({"best iters": max_right_num[1]})
#绘图并保存
    cost_array = np.array(cost_array)
    cost_array = np.reshape(cost_array, [cost_array.shape[0], 1])
    # print(cost_array)
    plt.figure()
    plt.plot(range(0, cost_array.shape[0]), cost_array, c='blue', label='batch-sample model Loss')
    # plt.figure()
    plt.title('Model Loss Function batch_size=%d'%batch_size)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.text(cost_array.shape[0] / 2, cost_array[0] / 2, "accuracy: %f %%" % (float(max_right_num[0]) * 100 / 6))
    # plt.scatter(cost_array,range(0,cost_array.shape[0]))
    # plt.show()
    plt.legend(loc='best')
    plt.savefig('hl=%d'%h_layer+'lr=%f'%learning_rate+'bz=%d.png'%batch_size)

#返回最优精度对应的模型参数
    return max_para

model(h_layer=5,batch_size=1,num_iters=1000,learning_rate=0.3,decay_rate=1,print_cost=False,print_test=True)
model(h_layer=5,batch_size=24,num_iters=1000,learning_rate=1,decay_rate=1,print_cost=False,print_test=True)
#model(h_layer=5,batch_size=24,num_iters=1000,learning_rate=20,decay_rate=1,print_cost=False,print_test=True)



