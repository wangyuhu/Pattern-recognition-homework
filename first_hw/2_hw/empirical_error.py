import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from numpy.linalg import cholesky
matplotlib.use('Agg')
# 生成d维高斯分布
def generate_d_Gaussian(mu,sigma,total_num):
	Mu = np.array(mu)
	Sigma = np.array(sigma)
	R = cholesky(Sigma)
	y = np.dot(np.random.randn(total_num,Mu.shape[0]), R) + Mu
	return y

if __name__ == '__main__':
	total_number = 100

	# 生成第一组数据
	mu1 = [1,0]
	sigma1 = [[1,0],[0,1]]
	Gaussian_y1 = generate_d_Gaussian(mu1, sigma1, int(total_number/2))
	plt.plot(Gaussian_y1[:,0], Gaussian_y1[:,1], '+', color='green')
	# 生成第二组数据
	mu2 = [-1,0]
	sigma2 = [[1,0],[0,1]]
	Gaussian_y2 = generate_d_Gaussian(mu2, sigma2, int(total_number/2))
	plt.plot(Gaussian_y2[:,0], Gaussian_y2[:,1], '+', color='red')
    # 计算错误率并绘制图形
	Perror1 = 0
	Perror2 = 2
	for i in range(int(total_number/2)):
		if(Gaussian_y1[i,0] < 0):
			Perror1 = Perror1 +1
		if(Gaussian_y2[i,0] > 0):
			Perror2 = Perror2 +1
	Perror = (Perror1 + Perror2) / total_number
	print(Perror)
	plt.plot([0,0],[-3,3])
	plt.savefig('result.jpg')