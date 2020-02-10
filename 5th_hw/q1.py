#coding:utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io as scio
from utils import K_means
import matplotlib.patches as mpatches

if __name__ == '__main__':
    #载入数据
    data_path = 'Kmeans_data.mat'
    data_array = scio.loadmat(data_path)['Kdata']
    mu_realarray = scio.loadmat(data_path)['Krealmu']
    #print(data_array)
    plt.clf()
    plt.scatter(data_array[:,0], data_array[:,1], marker='+', color='red', s=40, label='All data')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('All data')
    #plt.show()
    plt.savefig('kmeans_data.png')
    #进行K-Means聚类
    class_index,mu_array = K_means(data_array,5)
    #print(mu_array)
    #绘制分类结果
    classified_data = {'0':[], '1':[], '2':[], '3':[], '4':[]}
    for index,value in enumerate(class_index):
        classified_data[str(int(value))].append(data_array[index])
    #classified_data = np.array(classified_data)
    #print(classified_data['0'][0])
    # color_d = {'0':'r', '1':'g', '2':'b', '3':'c', '4':'y'}
    color = ['lightpink', 'green', 'lightblue', 'cyan','y']
    # labels = ['0','1','2','3','4']
    # patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
    # color_class = [color_d[str(int(i))] for i in class_index]
    plt.clf()
    for i in range(5):
        plt.scatter(np.array(classified_data[str(i)])[:,0], np.array(classified_data[str(i)])[:,1], marker='+', color=color[i], s=40,label=str(i))
    plt.scatter(mu_array[:,0], mu_array[:,1], marker='o',color='r')
    for mu in mu_array:
        plt.text(mu[0], mu[1], '['+str(round(mu[0],2))+','+str(round(mu[1],2))+']', ha='center', va='bottom', fontsize=10)
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('k_means_result')
    plt.savefig('k_means_result.png')

    #print('每一类中的样本个数为：')
    for i in range(5):
        error = 10
        for j in range(5):
            error_tmp = np.linalg.norm(mu_realarray[j] - mu_array[i])
            #print(error_tmp)
            if (error > error_tmp):
                error = error_tmp
        print('第'+str(i)+'类聚类中心为：'+'['+str(round(mu_array[i,0],2))+','+str(round(mu_array[i,1],2))+']'
              +'   类内个数：'+str(len(classified_data[str(i)]))
              +'   聚类中心均方误差为：'+str(error))

    #print(mu_realarray)

