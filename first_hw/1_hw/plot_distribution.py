import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
matplotlib.use('Agg')

DEBUG = 1
# 生成n个在[xl,xu]之间的数，返回数组形式
def generate_n(xl,xu,n):
	return np.random.randint(xl,xu,n)
# 生成xl,xu,n
def generate_Hyperparameter():
	para = []
	x1 = np.random.randint(-100,100,1)[0]
	x2 = np.random.randint(-100,100,1)[0]
	para.append(min(x1,x2))
	para.append(max(x1,x2))
	para.append(np.random.randint(1,1000,1)[0])
	return para

if __name__ == '__main__':
	total_number_set = 1000000    #设置数组最终的大小
	total_number = 0
	total_result = []
	for i in range(total_number_set):    #产生一系列的数组，并合并这些数组
		para = generate_Hyperparameter()
		if(DEBUG):
			print(para)
		if(para[0] >= para[1]):
			continue
		result = generate_n(para[0],para[1],para[2])
		total_result = np.concatenate((total_result, result))
		total_number = total_number + para[2]
		if(total_number > total_number_set):
			break
	if(DEBUG):
		print(len(total_result))
	result_mean = np.mean(total_result)    #统计并画图
	result_std = np.std(total_result)
	print(result_mean)
	print(result_std)
	n, bins, patches = plt.hist(total_result, bins=201, range=(-100,100))
	x = np.arange(-100,100,0.1)
	y = total_number_set * np.exp(-((x - result_mean)**2)/(2*result_std**2)) / (result_std * np.sqrt(2*np.pi))
	plt.plot(x,y)
	plt.xlabel('x')
	plt.ylabel('Total_number')
	plt.title('Distribution')
	plt.savefig('result.jpg')