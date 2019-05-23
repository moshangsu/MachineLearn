#!/usr/bin/env python
# coding: utf-8
"""
date: 2019-04-27
author: 张继龙
"""

from sklearn.linear_model import Ridge
import pandas as pd

def load_data(path='dataSet.txt'):

    data = pd.read_csv(path, sep=' ')
    X = data[['x']]
    y = data[['y']]

    return X, y

def fit_model(X, y):
    """
    alpha：正则化项的系数
    copy_X：是否对X数组进行复制，默认为True，如果选False的话会覆盖原有X数组
    fit_intercept：是否需要计算截距
    max_iter：最大的迭代次数，对于sparse_cg和lsqr而言，默认次数取决于scipy.sparse.linalg，对于sag而言，则默认为1000次。
    normalize：标准化X的开关，默认为False
    solver：在计算过程中选择的解决器
    auto：自动选择
    svd：奇异值分解法，比cholesky更适合计算奇异矩阵
    cholesky：使用标准的scipy.linalg.solve方法
    sparse_cg：共轭梯度法，scipy.sparse.linalg.cg,适合大数据的计算
    lsqr：最小二乘法，scipy.sparse.linalg.lsqr
    sag：随机平均梯度下降法，在大数据下表现良好。
    注：后四个方法都支持稀疏和密集数据，而sag仅在fit_intercept为True时支持密集数据。
    tol：精度
    random_state：sag的伪随机种子
    """
    model = Ridge(alpha=5).fit(X, y)
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
    # plt.plot(X, model.fittedvalues)
    plt.scatter(X, y, s=20, c="#ff1212", marker='o')
    plt.show()

if __name__ == '__main__':
    X, y = load_data()
    model = fit_model(X, y)
    print(model.predict(pd.DataFrame([[0.1], [1.71]], columns=['x'])))

    plt_show(X, model)