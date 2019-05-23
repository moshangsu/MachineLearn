

import pandas as pd
from sklearn import linear_model


def load_data(path='TestSet.txt'):

    data = pd.read_csv(path, sep=' ')
    X = data[['x']]
    y = data[['y']]

    return X, y

def fit_model(X, y):

    # 解决回归类型的问题(这里的y是浮点类型数据)
    # model = tree.DecisionTreeRegressor()
    model = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    model.fit(X, y.astype('int'))

    return model

def plt_show(X, y):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.title('散点图示例')
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    plt.scatter(X, y, s=20, c="#ff1212", marker='o')
    plt.show()

if __name__ == '__main__':
    X, y = load_data()
    model = fit_model(X, y)
    print(model.predict(pd.DataFrame([[1.7], [1.71]], columns=['x'])))

    plt_show(X, y)