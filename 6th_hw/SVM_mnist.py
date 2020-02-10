# coding:utf-8
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import numpy as np
import struct
import matplotlib.pyplot as plt
import os
##加载svm模型
from sklearn import svm
###用于做数据预处理
from sklearn import preprocessing
import time

path = ''
DEBUG = False
def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels
def load_mnist_test(path, kind='t10k'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def main():
    train_images,train_labels=load_mnist_train(path)
    test_images,test_labels=load_mnist_test(path)
    if(DEBUG):
        print(train_images.shape,train_labels.shape)
        print(test_images.shape,test_labels.shape)
        #print(train_labels[0:10])
        #print(train_labels[0].type)
    # 标准化
    X = preprocessing.StandardScaler().fit_transform(train_images)
    # X_train = X[0:60000]
    # y_train = train_labels[0:60000]
    X_train = X[np.where(train_labels < 2)]
    y_train = train_labels[np.where(train_labels < 2)]
    if(DEBUG):
        print(X_train.shape,y_train.shape)
        print(y_train[0:10])

    # 模型训练
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    model_svc = svm.SVC()
    model_svc.fit(X_train, y_train)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))

    # 评分并预测
    x = preprocessing.StandardScaler().fit_transform(test_images)
    # x_test = x[0:10000]
    # y_pred = test_labels[0:10000]
    x_test = x[np.where(test_labels < 2)]
    y_pred = test_labels[np.where(test_labels < 2)]
    print('test score:')
    print(model_svc.score(x_test, y_pred))
    # y = model_svc.predict(x_test)


if __name__ == '__main__':
    main()