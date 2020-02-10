#单样本三层神经网络模型
def onesample_model(h_layer,num_iters = 1000, print_cost=True, learning_rate=0.1,decay_rate=1,print_test=True):
    data_path = "/Users/xiaohu/Documents/Pycharm_server/152/Pattern-recognition-homework/4th_hw/data.txt"
    X, Y = load_datasets(data_path)
    print({"X":X})
    print({"Y":Y})
    print(X.shape)
    print(Y.shape)

    para = para_init(X, Y, h_layer)
    print(para['W1'].shape)
    print(para['b1'].shape)
    print(para['W2'].shape)
    print(para['b2'].shape)

    cost_array = []
    max_right_num = [0,0]
    for i in range(0,num_iters):
        mean_cost = 0
        #print(X.shape[1])
        # train_index=[]
        # for k in range(0,3):
        #     train_index.append(random.sample(range(k*10,k*10+10),8))
        # train_index = np.reshape(np.array(train_index),[24,1])
        #print(train_index)
        for j in range(0,X.shape[1]):
            if (j%10==0 or j%10==1):
                continue
            x = np.reshape(X[:, j], [3, 1])
            # print(x.shape)
            y_o, cache = forward_propagation(x, para)
            cost = computer_cost(y_o, Y[j])
            mean_cost = mean_cost + cost
            grads = backward_propagation(Y[j], x, cache, para)
            para = updata_para(para,grads,learning_rate)
        if (not i % (num_iters / 4)):
            learning_rate = learning_rate / decay_rate
        if(not i%1 ):

            cost_array.append(mean_cost/(X.shape[1]-6))
            if (print_cost):
                print({'cost': mean_cost/(X.shape[1]-6)})


            label = np.array([Y[0],Y[10],Y[20]])
            right_num = 0
            error_num = 0
            for k in range(0, X.shape[1]):
                if (k%10==0 or k%10==1):
                #if(True):
                    x = np.reshape(X[:, k], [3, 1])
                    y_o, cache = forward_propagation(x,para)
                    #print(y_o,Y[j])
                    test_label = np.where(abs(label - y_o[0][0]) == np.min(abs(label - y_o[0][0])))
                    #print(test_label[0][0]==int(Y[j]*3))
                    if (test_label[0][0] == int(Y[k]*3)):
                        right_num = right_num + 1
                    else:
                        error_num = error_num + 1
            if(right_num > max_right_num[0]):
                print(i)
                #print(max_right_num[0])
                max_right_num[0] = right_num
                max_right_num[1] = i
                max_para = para
    if (print_test):
        for k in range(0, X.shape[1]):
            if (k % 10 == 0 or k % 10 == 1):
                # if(True):
                x = np.reshape(X[:, k], [3, 1])
                y_o, cache = forward_propagation(x, para)
                print(y_o,Y[k])
        print({"accuracy": float(max_right_num[0]) / 6})
        print({"best iters": max_right_num[1]})
    cost_array = np.array(cost_array)
    cost_array = np.reshape(cost_array,[cost_array.shape[0],1])
    #print(cost_array)
    plt.figure()
    plt.plot(range(0,cost_array.shape[0]),cost_array,c='red',label = 'one-sample model Loss')
    #plt.figure()
    plt.title('One-sample Model Loss Function')
    #plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('Loss')
    plt.xlabel('Step')
    #plt.scatter(cost_array,range(0,cost_array.shape[0]))
    #plt.show()
    plt.text(cost_array.shape[0] / 2, cost_array[0] / 2, "accuracy: %f %%" % (float(max_right_num[0])*100 / 6))
    plt.legend(loc='best')
    plt.savefig('one-sample-model.png')
    return max_para
#批处理三层神经网络模型
def batch_model(h_layer, num_iters = 1000, print_cost=True, learning_rate=0.1,decay_rate=1,print_test=True):
    data_path = "/Users/xiaohu/Documents/Pycharm_server/152/Pattern-recognition-homework/4th_hw/data.txt"
    X, Y = load_datasets(data_path)

    # print({"X":X})
    # print({"Y":Y})
    print(X.shape)
    print(Y.shape)

    para = para_init(X, Y, h_layer)
    print(para['W1'].shape)
    print(para['b1'].shape)
    print(para['W2'].shape)
    print(para['b2'].shape)

    cost_array = []
    max_right_num = [0,0]
    for i in range(0, num_iters):
        mean_cost = 0
        # train_index = []
        # for k in range(0, 3):
        #     train_index.append(random.sample(range(k * 10, k * 10 + 10), 8))
        # train_index = np.reshape(np.array(train_index), [24, 1])
        mean_grads = {"dW1": 0,"db1": 0,"dW2": 0,"db2": 0}
        #for j in train_index:
        for j in range(0,X.shape[1]):
            if (j%10==0 or j%10==1):
                continue
            x = np.reshape(X[:, j], [3, 1])
            # print(x.shape)
            y_o, cache = forward_propagation(x, para)
            cost = computer_cost(y_o, Y[j])
            mean_cost = mean_cost + cost
            grads = backward_propagation(Y[j], x, cache, para)
            for key in mean_grads:
                mean_grads[key] = mean_grads[key] + grads[key]
            # para = updata_para(para, grads, learning_rate)
        for key in mean_grads:
            mean_grads[key] = mean_grads[key]/(X.shape[1] - 6)
        para = updata_para(para, mean_grads, learning_rate)
        if (not i % (num_iters / 3)):
            learning_rate = learning_rate / decay_rate
        if (not i % 1):

            cost_array.append(mean_cost / (X.shape[1] - 6))
            if (print_cost):
                print({'cost': mean_cost / (X.shape[1]-6)})

            label = np.array([0.0, 0.333333333, 0.666666666])
            right_num = 0
            error_num = 0
            for j in range(0, X.shape[1]):
                if (j%10==0 or j%10==1):
                #if (True):
                    x = np.reshape(X[:, j], [3, 1])
                    y_o, cache = forward_propagation(x, para)
                    #print(x,y_o,Y[j])
                    test_label = np.where(abs(label - y_o[0][0]) == np.min(abs(label - y_o[0][0])))
                    # print(test_label[0][0]==int(Y[j]*3))
                    if (test_label[0][0] == int(Y[j] * 3)):
                        right_num = right_num + 1
                    else:
                        error_num = error_num + 1
        if(right_num > max_right_num[0]):
            print(i)
            max_right_num[0] = right_num
            max_right_num[1] = i
            max_para = para

            # print(right_num)
    if (print_test):
        for j in range(0, X.shape[1]):
            if (j % 10 == 0 or j % 10 == 1):
                # if(True):
                x = np.reshape(X[:, j], [3, 1])
                y_o, cache = forward_propagation(x, max_para)
                print(y_o,Y[j])
        print({"accuracy": float(max_right_num[0]) / 6})
        print({"best iters": max_right_num[1]})
    cost_array = np.array(cost_array)
    cost_array = np.reshape(cost_array,[cost_array.shape[0],1])
    #print(cost_array)
    plt.figure()
    plt.plot(range(0,cost_array.shape[0]),cost_array,c='blue',label = 'batch-sample model Loss')
    #plt.figure()
    plt.title('Batch-sample Model Loss Function')
    #plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.text( cost_array.shape[0]/2,cost_array[0]/2,"accuracy: %f %%" % (float(max_right_num[0]) *100/ 6))
    #plt.scatter(cost_array,range(0,cost_array.shape[0]))
    #plt.show()
    plt.legend(loc='best')
    plt.savefig('batch-sample-model.png')

    return max_para