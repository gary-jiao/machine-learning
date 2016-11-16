#coding=utf-8

from numpy import *

def loadDataSet():
	postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
				 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]    #1 代表侮辱性文字，0 代表正常言论
	return postingList, classVec

'''创建词汇表'''
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)		#创建两个集合的并集，并且去除重复的词
	return list(vocabSet)

'''设置词向量'''
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)			#创建一个其中所含元素都为0的向量
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print "the word: %s is not in my vocabulary!" % word
	return returnVec

'''朴素贝叶斯词袋模型'''
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


'''朴素贝叶斯分类器训练函数'''
'''trainMatrix:文档矩阵'''
'''trainCategory:文档类别标签构成的向量'''
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)  # 总文档数
	numWords = len(trainMatrix[0])  # 文档1的长度
	pAbusive = sum(trainCategory) / float(numTrainDocs)  # 训练文档中,属于侮辱类型的概率
	p0Num = ones(numWords); p1Num = ones(numWords)     # 初始化概率
	p0Denom = 2.0; p1Denom = 2.0
	
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			#向量相加
			p1Num += trainMatrix[i]         #类别1的分子，是个词向量，最后值为：在侮辱类文档，词表中各个词出现的次数
			p1Denom += sum(trainMatrix[i])  #类别1的分母，是个常数，表示每个侮辱类文档中出现的词数的总和
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])

	#对每个元素作除法
	p1Vect = log(p1Num / p1Denom)      #属于侮辱类文档时，每个词出现的概率，即p(w0|c1),p(w1|c1),p(w2|c1)···
	p0Vect = log(p0Num / p0Denom)      #属于非侮辱类文档时，每个词出现的概率，即p（w0|c0）,p(w1|c1),p(w2|c1)···
	return p0Vect, p1Vect, pAbusive

'''朴素贝叶斯分类函数'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0: return 1
	return 0

'''测试函数'''
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)


def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 


def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText