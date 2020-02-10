#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io as scio
from utils import Spectral_clustering

def main():
    #载入数据
    data_path = 'Sdata.mat'
    data_array = np.array(scio.loadmat(data_path)['Sdata'])
    #mu_realarray = scio.loadmat(data_path)['Krealmu']
    #print(data_array.shape)
    plt.clf()
    plt.scatter(data_array[:,0], data_array[:,1], marker='+', color='red', s=40, label='All data')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('All data')
    #plt.show()
    plt.savefig('spectral_data.png')
    #调用函数进行谱聚类
    class_index, mu_array = Spectral_clustering(data_array,sigma=1,w_k=6,l_k=2,m_k=2)
    #获取分类准确度
    real_class_index_1 = np.zeros(class_index.shape[0])
    real_class_index_1[0:real_class_index_1.shape[0]/2] = 1
    real_class_index_2 = np.zeros(class_index.shape[0])
    real_class_index_2[real_class_index_2.shape[0]/2:] = 1
    #print(real_class_index_1)
    #print(real_class_index_2)
    error = np.min([np.linalg.norm(class_index-real_class_index_1),np.linalg.norm(class_index-real_class_index_2)])
    accuracy = (class_index.shape[0]-error)/class_index.shape[0]
    #画出分类结果
    while(max(class_index) !=1 or min(class_index) !=0):
        class_index, mu_array = Spectral_clustering(data_array, sigma=1, w_k=6, l_k=2, m_k=2)
        #print('Error occured in clustering random init! Run the Prj again,thanks')
        #return 0
    #print(class_index)

    classified_data = {'0': [], '1': []}
    for index, value in enumerate(class_index):
        classified_data[str(int(value))].append(data_array[index])
    # classified_data = np.array(classified_data)
    # print(classified_data['0'][0])
    # color_d = {'0':'r', '1':'g', '2':'b', '3':'c', '4':'y'}
    color = ['r', 'g', 'b', 'c', 'y']
    # labels = ['0','1','2','3','4']
    # patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
    # color_class = [color_d[str(int(i))] for i in class_index]
    plt.clf()
    for i in range(2):
        plt.scatter(np.array(classified_data[str(i)])[:, 0], np.array(classified_data[str(i)])[:, 1], marker='+',
                    color=color[i], s=40, label=str(i))
    plt.text(1.25,-1.25, 'accuracy:'+str(accuracy), ha='center', va='bottom',
             fontsize=10)
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Spectral_clustering_result')
    plt.savefig('Spectral_clustering_result.png')
    #分析sigma对准确度的影响，并画出关系曲线
    sigma_accuracy = {'sigma':[], 'accuracy':[]}
    for i in range(1,10):
        #print('sigma:'+str(i*0.5))
        sigma_accuracy['sigma'].append(i*0.5)
        class_index, mu_array = Spectral_clustering(data_array, sigma=i*0.5, w_k=6, l_k=2, m_k=2)
        error = np.min(
            [np.linalg.norm(class_index - real_class_index_1), np.linalg.norm(class_index - real_class_index_2)])
        accuracy = (class_index.shape[0] - error) / class_index.shape[0]
        sigma_accuracy['accuracy'].append(accuracy)
    plt.clf()
    plt.plot(sigma_accuracy['sigma'],sigma_accuracy['accuracy'])
    plt.xlabel('sigma')
    plt.ylabel('accuracy')
    plt.title('accuracy-sigma relationship')
    plt.savefig('accuracy_sigma_relationship.png')
    #分析w_k对准确度的影响，并画出关系曲线
    wk_accuracy = {'wk': [], 'accuracy': []}
    for i in range(1,30):
        #print('wk:'+str(i))
        wk_accuracy['wk'].append(i)
        class_index, mu_array = Spectral_clustering(data_array, sigma=1, w_k=i, l_k=2, m_k=2)
        error = np.min(
            [np.linalg.norm(class_index - real_class_index_1), np.linalg.norm(class_index - real_class_index_2)])
        accuracy = (class_index.shape[0] - error) / class_index.shape[0]
        wk_accuracy['accuracy'].append(accuracy)
    plt.clf()
    plt.plot(wk_accuracy['wk'], wk_accuracy['accuracy'])
    plt.xlabel('w_k')
    plt.ylabel('accuracy')
    plt.title('accuracy-w_k relationship')
    plt.savefig('accuracy_w_k_relationship.png')

if __name__ == '__main__':
    main()