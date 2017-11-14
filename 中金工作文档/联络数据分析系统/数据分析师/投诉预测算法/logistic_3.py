# python -version 3.5+
import numpy as np
from numpy import *

def sigmoid(x):
    return .5 * (1 + np.tanh(.5 * x))

def trainLogRegres(train_x, train_y, opts):
    numSamples, numFeatures = shape(train_x)
    alpha = opts['alpha']
    maxIter = opts['maxIter']
    weights = ones((numFeatures, 1))

    for k in range(maxIter):
        if opts['optimizeType'] == 'gradDescent':
            output = sigmoid(train_x * weights)
            error = train_y - output
            weights = weights + alpha * train_x.transpose() * error
        elif opts['optimizeType'] == 'stocGradDescent':
            for i in range(numSamples):
                output = sigmoid(train_x[i, :] * weights)
                error = train_y[i, 0] - output
                weights = weights + alpha * train_x[i, :].transpose() * error
        elif opts['optimizeType'] == 'smoothStocGradDescent':
            dataIndex = list(range(numSamples))
            for i in range(numSamples):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int(random.uniform(0, len(dataIndex)))
                output = sigmoid(train_x[randIndex, :] * weights)
                error = train_y[randIndex, 0] - output
                weights = weights + alpha * train_x[randIndex, :].transpose() * error
                del (dataIndex[randIndex])
        else:
            raise NameError('Not support optimize method type!')
    return weights

def preLogRegres(weights ,pre_x):
    numSamples, numFeatures = shape(pre_x)
    outfile = open('E://test1.txt', 'w')
    for i in range(numSamples):
        predict = sigmoid(pre_x[i, :] * weights)[0, 0]
        outfile.write(str(predict)+'\n')
    outfile.close()

## step 1: 模型训练
def loadData():
    train_x = []
    train_y = []
    fileIn = open('E://data.txt')
    for line in fileIn.readlines():
        lineArr = line.split(',')
        train_x.append([1.0, float(lineArr[2]), float(lineArr[3]), float(lineArr[4]), int(lineArr[5]), float(lineArr[6]), float(lineArr[7]), float(lineArr[8])])
        train_y.append(float(lineArr[-1]))
    return mat(train_x), mat(train_y).transpose()
    fileIn.close()

train_x, train_y = loadData()
opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
optimalWeights = trainLogRegres(train_x, train_y, opts)

## step 2: 模型预测
def loadData1():
    pre_x = []
    fileIn = open('E://test.txt')
    for line in fileIn.readlines():
        lineArr = line.split(',')
        pre_x.append([1.0, float(lineArr[2]), float(lineArr[3]), float(lineArr[4]), int(lineArr[5]), float(lineArr[6]), float(lineArr[7]), float(lineArr[8])])
    return mat(pre_x)
    fileIn.close()

pre_x = loadData1()
accuracy = preLogRegres(optimalWeights, pre_x)

# step 3: 数据集输出
a = open('E://test.txt', 'r')
b = open('E://test1.txt', 'r')
list1 = []
for i in a:
    m = i.split(',')
    list1.append(m[:2])
list2 = []
for i in b:
    n = i.split()
    list2.append(n)
for (x, y) in zip(list1, list2):
    print((x[0] + ',' + x[1] + ',' + ''.join(y)), file=open('E://predict.txt', 'a+'))
a.close()
b.close()
