#coding=utf-8

from math import log
import operator

def createDataSet():
	dataSet = [
		[1, 1, 'yes'],
		[1, 1, 'yes'],
		[1, 0, 'no'],
		[0, 1, 'no'],
		[0, 1, 'no']
	]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

#计算给定数据集的香农熵(ShannonEntropy)
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)		#获得样本数
	labelCounts = {}				#为每个label（特征）包含的样本数进行计数
	
	#为所有可能分类创建字典
	for vec in dataSet:
		currentLabel = vec[-1]		#获取当前样本的标签，这里也就是最后一列
		if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1

	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob * log(prob, 2)		#以2为底 求对数

	return shannonEnt

#按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		#如果指定位置的特征值等于指定的value，则将数据放置到新的数据里
		if featVec[axis] == value:
			#将符合特征的数据抽取出来，并去掉用来划分数据集的特征
			reducedFeatVec = featVec[:axis]		
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)

	return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1			#特征的数量
	baseEntropy = calcShannonEnt(dataSet)		#计算初始数据集的熵
	bestInfoGain = 0.0; bestFeature = -1;		#定义best信息增益，best特征

	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]		#i对应的特征的所有样本取值的列表
		uniqueVals = set(featList)
		newEntropy = 0.0

		for value in uniqueVals:
			#计算每种划分方式的信息熵
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)

		infoGain = baseEntropy - newEntropy
		if (infoGain > bestInfoGain):
			#计算最好的信息增益
			bestInfoGain = infoGain
			bestFeature = i

	return bestFeature

#得到出现次数最多的分类名称
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys(): classCount[vote] = 0
		classCount[vote] += 1

	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

#创建树的函数代码
def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]		#获取所有样本的类列表

	#如果类别classList[0]的数目和classList的列表长度一样大，即数据集中的类别完全相同 -> 停止继续划分，返回类型
	if classList.count(classList[0]) == len(classList):	return classList[0]
	#数据集长度为1，也即所有特征划分均已进行，列表中只剩下"类别"，此时使用多数表决的方法决定该叶子节点的分类
	if len(dataSet[0]) == 1: return majorityCnt(classList)

	bestFeat = chooseBestFeatureToSplit(dataSet)	#当前数据集选取的最好特征
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel: {}}		#存储树的信息

	'''得到列表包含的 所有属性值 '''
	del(labels[bestFeat])		#移除上面划分所用的特征
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)

	'''遍历当前选择特征包含的 所有属性值 ，在每个数据集划分上递归调用函数createTree(),
	 	得到的返回值将被插入到字典变量myTree中，因此函数终止执行时，字典中将会嵌套很多代表叶子节点信息的字典数据'''
	for value in uniqueVals:
		subLables = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLables)

	return myTree


def classify(inputTree, featLables, testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLables.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ = 'dict':
				classLabel = classify(secondDict[key], featLables, testVec)
			else:
				classLabel = secondDict[key]
	return classLabel
