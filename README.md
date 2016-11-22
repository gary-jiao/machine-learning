# 跟着《机器学习实践》的学习过程

书本对应的源代码可以在[这里](https://github.com/longyinzaitian/MLInActionCode)找到。
![如何选择合适的算法](images/algorithm.png)

## kNN的近邻算法

- 优点：精度高、对异常值不敏感、无数据输入假定
- 缺点：计算复杂度高、空间复杂度高
- 适用数据范围：数值型和标称型 (标称型：就是离散型数据，变量的结果只在有限目标集中取值)

**k-近邻算法的一般流程**  

1. 收集数据
2. 准备数据：距离计算所需要的数值，最好是结构化的数据格式
3. 分析数据
4. 训练算法：本算法不需要此步骤
5. 测试算法：计算错误率
6. 使用算法：首先需要输入样本数据和结构化的输出结果，然后运行算法判定输入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理

k-近邻算法是基于实例的学习，必须有接近实际数据的训练样本数据。计算时必须保存全部数据集，可能会需要使用大量的存储空间。此外，因为必须对数据集中的每个数据计算距离值，实际使用可能非常耗时。另一个缺陷是，无法给出任何数据的基础结构信息，无法知晓平均实例样本和典型实例样本具有什么特征。

根据k-近邻算法的公式，某些数据偏差非常大的数据，会对结果千万非常大的影响。例如每年的飞行里程数，可以从0到几万，如果有这样的数据，在计算的时候，飞行里程对结果的影响非常严重。这种情况下，需要对数值进行归一化处理，也就是将取值范围处理为0~1，或者-1~1之间。可能使用以下公式将任意取值范围的特征值转化为 0 ~ 1 的区间。  
` newValue = (oldValue - min) / (max - min) `

## 决策树

- 优点：计算复杂度不高、输入结果易于理解、对中间值的缺失不敏感、可以处理不相关特征数据
- 缺点：可能会产生过度匹配(过度拟合，overfitting)问题
- 适用数据类型：数值型和标称型

**overfitting** 是这样一种现象：一个假设在训练数据上能够获得比其他假设更好的拟合，但是在训练数据外的数据集上却不能很好的拟合数据。此时我们就叫这个假设出现了overfitting 的现象。出现这种现象的主要原因是训练数据中存在噪音或者训练数据太少。而解决 overfitting 的方法主要有两种：提前停止树的增长或者对已经生成的树按照一定的规则进行后剪枝。


**决策树的一般流程**

1. 收集数据
2. 准备数据：树构造算法只适用于标称型数据，因此数据值数据必须离散化
3. 分析数据：可以使用任何方法，构造树完成之后，应该检查图形是否符合预期
4. 训练算法：构造树的数据结构
5. 测试算法：使用经验树计算错误率
6. 使用算法：此步骤可以适用于任何监督学习算法，而使用决策树可以更好地理解数据的内在含义

**递归构建决策树工作原理：**
得到原始数据集，然后基于最好的属性值划分数据集，由于特征值可能多于两个，因此可能存在大于两个分支的数据集划分。第一次划分之后，数据将被向下传递到树分支的下一个节点，在这个节点上，我们可以再次划分数据。因此我们可以采用递归的原则处理数据集。

**递归的结束条件是**  程序遍历完素有划分数据集的属性，或者每个分支下的所有实例都具有相同的分类。如果所有实例具有相同的分类，则得到一个叶子节点或者终止块。任何到达叶子节点的数据必然属于叶子节点的分类。

决策树分类器就像带有终止块的流程图，终止块表示分类结果。开始处理数据集时，我们首先需要测量集合中数据的不一致性，也不是熵，然后寻找最优方案划分数据集，直到数据集中的所有数据属于同一分类。ID3算法可以用于划分标称型数据集。通过采用递归的方法将数据转化为决策树。


## 基于概率论的朴素贝叶斯

- 优点：在数据较少的情况下仍然有效，可以处理多类别问题
- 缺点：对于输入数据的准备方式较为敏感
- 适用数据类型：标称型数据

**贝叶斯决策理论的核心思想，选择具有最高概率的决策。**  

一个重要的概念：**条件概率**

例如：针对某个物体，我们用 p1(x) 表示其属于 类别1 的概率，用 p2(x) 表示其属于 类别2 的概率，我们可以用下面的规则来判断它的类别：
	- 如果p1(x) > p2(x)， 那么类别为1。
	- 如果p2(x) > p1(x)， 那么类别为2。

**朴素贝叶斯的一般流程**

1. 收集数据
2. 准备数据：需要数值型或者布尔型数据
3. 分析数据：有大量特征时，绘制特征作用不大，此时使用直方图效果更好
4. 训练算法：计算不同的独立特征的条件概率
5. 测试算法：计算错误率
6. 使用算法：一个常见的相互贝叶斯应用是文档分类。可以在任意的分类场景中使用朴素贝叶斯分类器，不一定非要是文本

假设词汇表中有1000个单词。要得到好的概率分布，就需要足够的数据样本，假定样本数为N。由统计学知，如果每个特征需要N个样本，那么对于10个特征将需要 N^10 个样本，对于包含1000个特征的词汇表将需要 N^1000 个样本。可以看到，所需要的样本数会随着特征数目增大而迅速增长。

如果特征间相互独立，那么样本数就可以从 N^1000 减少到 1000xN。所谓 独立（independence）指的是统计意义上的独立，即一个特征或者单词出现的可能性与它和其他单词相邻没有关系。

**训练算法：从词向量计算概率**

伪代码如下：
```
计算每个类别中的文档数目
对每篇训练文档：
	对每个类别：
		如果词条出现文档中 --> 增加该词条的计数值
		增加所有词条的计数值
	对每个类别：
		对每个词条：
			将该词条的数目除以总词条数目得到条件概率
	返回每个类别的条件概率
```

为避免发生下溢出（太多很小的数相乘，得到0），一种解决办法是**对乘积取自然对数**。在代数中有ln(a*b) = ln(a) + ln(b)。

**文档词袋模型：**  
我们将每个词的出现与否作为一个特征，这可以被描述为**词集模型（set-of-words model）**。如果一个词在文档中不止一次出现，那它可能还包含着除了此词是否在文档中出现以外的一系列信息，这种方法被称为**词袋模型（bag-of-words model）**。在词袋模型中，每个单词可以出现多次，而在词集中，每个词只能出现一次。

**小结：**  
对于分类而言，使用概率有时要比使用硬规则更为有效。贝叶斯概率及贝叶斯准则提供了一种利用已知值来估计未知概率的有效方法。

可以通过特征之间的条件独立性假设，降低对数据量的需求。独立性假设是指一个词的出现概率并不依赖于文档中的其他词。当然，这个假设过于简单。这就是称其为朴素贝叶斯的原因。尽管条件独立性假设并不正确，但是朴素贝叶斯仍然是一种有效的分类器。

利用现代编程语言来实现朴素贝叶斯时需要考虑很多实际因素。下溢出就是其中一个问题，它可以通过对概率取对数来解决。词袋模型在解决文档分类问题上比词集模型有所提高。还有其他一些方面的改进，比如说移除停用词，当然也可以话大量时间对切分器进行优化。


## Logistic回归

**什么是回归**： 假设有一些数据点，用一条直线对这些点进行拟合（该线称为最佳拟合直线），这个拟合过程就称作回归。  
**回归分类的主要思想**： 根据现有数据对分类边界线建立回归公式，以此进行分类。

**Logistic回归的一般流程**

1. 收集数据
2. 准备数据：由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳
3. 分析数据：采用做生意方法对数据进行分析
4. 训练算法：大部分时间将用于训练，训练的目的是为了找到最佳的分类回归系数
5. 测试算法：一旦训练步骤完成，分类将会很快
6. 使用算法：
	1. 需要输入一些数据，并将其转换成对应的结构化数值；
	2. 基于训练好的回归系数就可以对这些数值进行简单的回归计算，判断它们属于哪个类别；
	3. 我们就可以在输出的类别上做一些其他分析工作

- 优点：计算代价不高，易于理解和实现
- 缺点：容易欠拟合，分类精度可能不高
- 适用数据类型：数值型和标称型数据

梯度上升算法的伪代码如下：   
```
每个回归系统初始化为1
重复R次：
	计算整个数据集的梯度
	使用 alpha x gradient 更新回归系数的向量
	返回回归系统
```

**随机梯度上升算法**

梯度上升算法在每次更新回归系数时都需要遍历整个数据集，该方法在处理100个左右的数据集时尚可，但如果有数十亿样本和成千上万的特征，那么该方法的计算复杂度就太高了。一种改进方法是一次仅用一个样本点来更新回归系数，该方法称为随机梯度上升算法。由于可以在新样本到来时对分类器进行增量式更新，因而随机梯度上升算法是一个在线学习算法。与“在线学习”相对应，一次处理所有数据被称作是“批处理”。

随机梯度上升算法的伪代码：
```
所有回归系数初始化为1 
对数据集中每个样本 
	计算该样本的梯度 
    使用alpha x gradient更新回归系数 
返回回归系数值
```

随机梯度上升算法与梯度上升算法很相似，但是也有一些区别：
- 后者的变量h和误差error都是向量，而前者则全是数值。
- 前者没有矩阵的转换过程，所有变量的数据类型都是NumPy数组。

Logistic回归的目的是寻找一个非线性函数Sigmoid的最佳拟合参数，求解过程可以由最优化算法来完成。在最优化算法中，最常用的就是梯度上升算法，而梯度上升算法又可以简化为随机梯度上升算法。

随机梯度上升算法与梯度上升算法的效果相当，但占用更少的计算资源。此外，随机梯度上升算法是一个在线算法，它可以在新数据到来时就完成参数更新，而不需要重新读取整个数据集来进行批处理运算。



## 支持向量机

**支持向量(support vector)**  就是离分隔超平面最近的那些点。

- 优点：泛化错误率低，计算开销不大，结果易解释
- 缺点：对参数调节和核函数的选择敏感，原始分类器不加修改仅适用于处理二类问题
- 适用数据类型：数值型和标称型数据

**SVM的一般流程**

1. 收集数据
2. 准备数据：需要数值型数据
3. 分析数据：有助于可视化分隔超平面
4. 训练算法：SVN的大部分时间都源自训练，该过程主要实现两个参数的调优
5. 测试算法：十分简单的计算过程就可以实现
6. 使用算法：几乎所有分类问题都可以使用SVM，值得一提的是，SVN本身是一个二类分类器，对多类问题应用SVM需要对代码做一些修改

**SMO高效优化算法**

SMO表示序列最小优化(Sequential Minimal Optimization)。SMO算法将大优化问题分解为多个小优化问题来求解的。这些小优化问题往往很容易求解，并且对它们进行顺序求解的结果与将它们作为整体来求解的结果是完全一致的。在结果完全相同的同时，SMO算法的求解时间短很多。

**SMO算法的工作原理**： 每次循环中选择两个alpha进行优化处理。一旦找到一对合适的alpha，那么就增大其中一个同时减小另一个。这里所谓的“合适”是指两个alpha必须要符合一定的条件，条件之一就是这两个alpha必须要在间隔边界之外，而第二个条件就是这两个alpha还没有进行过区间化处理或者不在边界上。


## 利用AdaBoost元算法提高分类性能

当做出重要决定时，大家可能都会考虑吸取多个专家而不只是一个人的意见。这就是元算法(meta-algorithm)背后的思路。元算法是对其他算法进行组合的一种方式。AdaBoost是最流行的元算法。全称是Adaptive boosting(自适应boosting)。

- 优点：泛化错误率低，易编码，可以应用在大部分分类器上，无参数调整
- 缺点：对离群点敏感
- 适用数据类型：数值型和标称型数据

**AdaBoost的一般流程**
1. 收集数据
2. 准备数据：依赖于所使用的弱分类器类型，本章使用的是单层决策树，这种分类器可以处理任何数据类型。当然也可以使用做任意分类器作为弱分类器，前面几章中的任一分类器都可以充当弱分类器。作为弱分类器，简单分类器的效果更好
3. 分析数据：任意方法
4. 训练算法：AdaBoost大分部时间都用在训练上，分类器将多次在同一数据集上训练弱分类器
5. 测试算法：计算分类的错误率
6. 使用算法：同SVM一样，AdaBoost预测两个类别中的一个。如果想把它应用到多具类别的场合，那么就要像多类SVM中的做法一样对AdaBoost进行修改

完整AdaBoost算法的伪代码：
```
对每次迭代：
	利用buildStump()函数找到最佳的单层决策树
	将最佳单层决策树加入到单层决策树数组
	计算alpha
	计算新的权重向量D
	更新累计类别估计值
	如果错误率等于0.0，退出循环
```

## 预测数值型数据：回归

用线性回归找到最佳拟合直线
- 优点：结果易于理解，计算上不复杂
- 缺点：对非线性的数据拟合不好
- 适用数据类型：数值型和标称型数据

**回归的目的**  是预测数值型的目标值。

**线性回归的一般流程**
1. 收集数据
2. 准备数据：回归需要数值型数据，标称型数据将被转成二值型数据
3. 分析数据：绘出数据的可视化二维图将有助于对数据做出理解和分析，在采用缩减法求得新回归系数之后，可以将新拟合线会在图片作为对比
4. 训练算法：找到回归系数
5. 测试算法：使用R2或者预测值和数据的拟合度，来分析模型的效果
6. 使用算法：使用回归，可以在给定输入的时候预测出一个数值，这是对分类方法的提升，因为这样可以预测连续型数据而不是仅仅是离散的类别标签

线性回归的一个问题是可能出现欠拟合现象，因为它求的是具有最小均方误差的无偏估计。所有如果模型欠拟合将不能取到最好的预测效果。所以有些方法允许在估计中引入一些偏差，从而降低预测的均方误差。其中一个方法就是局部加权线性回归(Locally Weighted Linear Regression, LWLR)。

缩减系数来“理解”数据
- 岭回归 (Ridge Regression)
- lasso
- 前向逐步线性回归


=====================================

# 目前明显发现，对于数学底子已经差的一塌糊涂的情况下，要去搞懂机器学习那些算法的原理，真的是相当难，很多内容看完之后，知其然不其知所以然，这样学习了几章，感觉收获颇小。决定后面的重点还是往实际应用上偏，多了解每种算法可以大致用在什么场合就可以了，至于其原理，还是留给别人去搞吧。