#!/usr/bin/env python
# coding: utf-8
"""
date: 2019-04-27
author: 张继龙
"""

import statsmodels.api as sm
import pandas as pd

def load_data(path='dataSet.txt'):

    data = pd.read_csv(path, sep=' ')
    X = data[['x']]
    y = data[['y']]

    return X, y

def fit_model(X, y):

    # 解决回归类型的问题(这里的y是浮点类型数据)
    X = sm.add_constant(X)
    model = sm.OLS(X, y).fit()
    print(model)
    # model = sm.WLS(X, y).fit()

    return model

def plt_show(X, model):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.title('散点图示例')
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    plt.plot(X, model.fittedvalues)
    plt.scatter(X, y, s=20, c="#ff1212", marker='o')
    plt.show()

if __name__ == '__main__':
    X, y = load_data()
    model = fit_model(X, y)
    print(model.predict(pd.DataFrame([[0.1], [1.71]], columns=['x'])))

    plt_show(X, model)