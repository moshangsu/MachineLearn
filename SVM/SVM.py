#!/usr/bin/env python
# coding: utf-8
"""
date: 2019-04-27
author: 张继龙
"""

import pandas as pd
from sklearn import svm

def load_data(path='testSetRBF.txt'):

    data = pd.read_csv(path, sep=' ')
    X = data[['x', 'y']]
    y = data[['catagory']]

    return X, y

def fit_model(X, y):

    # 解决回归类型的问题(这里的y是浮点类型数据)
    # model = tree.DecisionTreeRegressor()
    model = svm.LinearSVC()
    model.fit(X, y)

    return model

def plt_show(X):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.title('散点图示例')
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    plt.scatter(X['x'], X['y'], s=20, c="#ff1212", marker='o')
    plt.show()

if __name__ == '__main__':
    X, y = load_data()
    model = fit_model(X, y)
    print(model.predict(pd.DataFrame([[0.1, 0.1], [1.71, 1.3]], columns=['x', 'y'])))

    plt_show(X)