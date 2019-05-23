#!/usr/bin/env python
# coding: utf-8
"""
date: 2019-04-27
author: 张继龙
"""

from sklearn import tree
import numpy as np
import pandas as pd
"""
收集数据：可以使用任何方法。
准备数据：树构造算法 (这里使用的是ID3算法，只适用于标称型数据，这就是为什么数值型数据必须离散化。 还有其他的树构造算法，比如CART)
分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期。
训练算法：构造树的数据结构。
测试算法：使用训练好的树计算错误率。
使用算法：此步骤可以适用于任何监督学习任务，而使用决策树可以更好地理解数据的内在含义。
"""

def load_data(path='lenses.txt'):

    data = pd.read_csv(path, sep=' ')
    X = data[['high', 'weight']]
    y = data[['somatotype']]

    return X, y

def fit_model(X, y):

    # 解决回归类型的问题(这里的y是浮点类型数据)
    # model = tree.DecisionTreeRegressor()
    model = tree.DecisionTreeClassifier()
    model.fit(X, y)

    return model

if __name__ == '__main__':
    X, y = load_data()
    model = fit_model(X, y)
    print(model.predict(pd.DataFrame([[1.7, 63], [1.71, 82]], columns=['high', 'weight'])))

