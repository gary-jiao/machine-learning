#coding=utf-8

from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])		#X0, X1, X2, 为方便计算，将X0的值设为1.0
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

'''梯度上升算法：dataMatIn 是一个100*3的矩阵，包含了两个特征X1和X2，再加上第0维特征X0。每行是一个训练样本，每列代表每个不同的特征值'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #类别标签，1*100的行向量，为便于计算，将其转换为列向量（即转置）
    m,n = shape(dataMatrix)					#获取矩阵的大小  m=100, n=3
    alpha = 0.001							#向目标移动的步长
    maxCycles = 500							#迭代次数
    weights = ones((n,1))					#返回n行1列的，全为1的矩阵，此处为3*1
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #dataMatrix*weights（初始为全为1的列向量，即权重相等）也就是之前的z，输入为100*1的列向量，计算得到h
        error = (labelMat - h)              #与实际类别进行比较，得到误差值error,100*1的列向量
        weights = weights + alpha * dataMatrix.transpose()* error #根据差值error的方向更新weights
    return weights    						#返回训练好的回归系数，也即当前最佳权重系数or最佳回归系数


'''画出数据集和Logistic回归最佳拟合直线的函数'''
def plotBestFit(wei):
	import matplotlib.pyplot as plt
	weights = wei
	dataMat, labelMat = loadDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
		else:
			xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s = 30, c = 'red')
	ax.scatter(xcord2, ycord2, s = 30, c = 'green')
	x = arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, y)
	plt.xlabel('X1'); plt.ylabel('X2')
	plt.show()

'''随机梯度上升算法'''
def stocGradAscent0(dataMatrix, classLabels):
	m, n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	for i in range(m):
		h = sigmoid(dataMatrix[i] * weights)
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return weights

'''改进的梯度上升算法'''
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
	m, n = shape(dataMatrix)
	weights = ones(n)
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			#1：alpha每次迭代时需要调整
			alpha = 4 / (1.0 + j + i) + 0.01

			#2：随机选取样本进行更新
			randIndex = int(random.uniform(0, len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex] * weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights