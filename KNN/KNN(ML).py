#!/usr/bin/env python
# coding: utf-8
"""
date: 2019-04-27
author: 张继龙
"""

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


def load_train_data(path='traindata.txt'):

    X = np.loadtxt(path, usecols=[0, 1, 2]).reshape(-1, 3)
    y = np.loadtxt(path, usecols=[3]).reshape(-1, 1)

    return X, y

def fit_model(X, y):

    model = KNeighborsClassifier()
    model.fit(X, y)

    return model

if __name__ == '__main__':
    X, y = load_train_data()
    model = fit_model(X, y.ravel())

    print(model.predict([[40920, 10.141740, 0.191283], [10000, 0.1, 10]]))          # 预测出分类
    print(model.predict_proba([[40920, 10.141740, 0.191283], [10000, 0.1, 10]]))    # 预测出对应分类的概率