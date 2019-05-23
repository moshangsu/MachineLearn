import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def load_data(path='train.csv'):

    data = pd.read_csv(open(path, 'rb'))
    x = data[['Pclass', 'Age', 'Fare', 'title_Miss', 'title_Mr', 'title_Mrs', 'title_Other', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Family', 'Alone']]      # 选取作为划分特征的数据列
    y = data[['Survived']]

    x['Age'].fillna(x['Age'].mean(), inplace=True)

    return x, y


def choice_alt(X_train, X_test, y_train, y_test):
    classifiers = [
        KNeighborsClassifier(3), SVC(probability=True), DecisionTreeClassifier(), RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(), GaussianNB(), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(),
        LogisticRegression(),
        XGBClassifier()
    ]

    log_cols = ["算法", "得分"]
    log = pd.DataFrame(columns=log_cols)
    acc_dict = {}

    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train.values.ravel())
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

    for clf in acc_dict:
        acc_dict[clf] = acc_dict[clf]
        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
        log = log.append(log_entry)

    return log

if __name__ == '__main__':
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    log = choice_alt(x_train, x_test, y_train, y_test)
    print(log)