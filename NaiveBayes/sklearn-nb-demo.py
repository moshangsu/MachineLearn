#!/usr/bin/python
# coding:utf8

"""
Created on 2019-03-28
Updated on 2019-03-28
NaiveBayes：朴素贝叶斯
Author: zhangjilong
GitHub: https://github.com/apachecn/AiLearning
"""
from __future__ import print_function


# GaussianNB_高斯朴素贝叶斯
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


"""
收集数据: 可以使用任何方法。
准备数据: 需要数值型或者布尔型数据。
分析数据: 有大量特征时，绘制特征作用不大，此时使用直方图效果更好。
训练算法: 计算不同的独立特征的条件概率。
测试算法: 计算错误率。
使用算法: 一个常见的朴素贝叶斯应用是文档分类。可以在任意的分类场景中使用朴素贝叶斯分类器，不一定非要是文本。
"""
def load_data(path='data.txt'):

    data = pd.read_csv(path, sep=' ')
    X = data[['high', 'weight']]
    y = data[['somatotype']]

    return X, y

def fit_model(X, y):

    model = GaussianNB()
    model.fit(X, y)

    return model

if __name__ == '__main__':
    X, y = load_data()
    model = fit_model(X, y)
    print(model.predict(pd.DataFrame([[1.7, 63], [1.71, 65]], columns=['high', 'weight'])))

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y = np.array([1, 1, 1, 2, 2, 2])
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(X, Y)
# print(clf.predict([[-0.8, -1]]))
# clf_pf = GaussianNB()
# clf_pf.partial_fit(X, Y, np.unique(Y))
# print(clf_pf.predict([[-0.8, -1]]))

# MultinomialNB_多项朴素贝叶斯
'''
import numpy as np
X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, y)
print clf.predict(X[2:3])
'''

# BernoulliNB_伯努利朴素贝叶斯
'''
import numpy as np
X = np.random.randint(2, size=(6, 100))
Y = np.array([1, 2, 3, 4, 4, 5])
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X, Y)
print clf.predict(X[2:3])
'''