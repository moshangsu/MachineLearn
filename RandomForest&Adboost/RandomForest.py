#!/usr/bin/env python
# coding: utf-8
"""
date: 2019-04-27
author: 张继龙
"""

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def load_data(path='train.csv'):

    data = pd.read_csv(open(path, 'rb'))
    x = data[['Pclass', 'Age', 'Fare', 'title_Miss', 'title_Mr', 'title_Mrs', 'title_Other', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Family', 'Alone']]      # 选取作为划分特征的数据列
    y = data[['Survived']]                  # 预测结果

    x['Age'].fillna(x['Age'].mean(), inplace=True)

    return x, y

if __name__ == '__main__':
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    dt = DictVectorizer(sparse=False)
    dtc = DecisionTreeClassifier()

    x_train = dt.fit_transform(x_train.to_dict(orient="record"))

    x_test = dt.fit_transform(x_test.to_dict(orient="record"))

    dtc.fit(x_train, y_train)
    dt_predict = dtc.predict(x_test)
    print(dtc.score(x_test, y_test))
    print(classification_report(y_test, dt_predict, target_names=["died", "survived"]))

    rfc = RandomForestClassifier()

    rfc.fit(x_train, y_train)

    rfc_y_predict = rfc.predict(x_test)

    print(rfc.score(x_test, y_test))

    print(classification_report(y_test, rfc_y_predict, target_names=["died", "survived"]))

# 填充缺失值



# dt = DictVectorizer(sparse=False)
#
# print(x_train.to_dict(orient="record"))
#
# # 按行，样本名字为键，列名也为键，[{"1":1,"2":2,"3":3}]
# x_train = dt.fit_transform(x_train.to_dict(orient="record"))
#
# x_test = dt.fit_transform(x_test.to_dict(orient="record"))
#
# # 使用决策树
# dtc = DecisionTreeClassifier()
#
# dtc.fit(x_train, y_train)
#
# dt_predict = dtc.predict(x_test)
#
# print(dtc.score(x_test, y_test))
#
# print(classification_report(y_test, dt_predict, target_names=["died", "survived"]))
#
# # 使用随机森林
#
# rfc = RandomForestClassifier()
#
# rfc.fit(x_train, y_train)
#
# rfc_y_predict = rfc.predict(x_test)
#
# print(rfc.score(x_test, y_test))
#
# print(classification_report(y_test, rfc_y_predict, target_names=["died", "survived"]))
