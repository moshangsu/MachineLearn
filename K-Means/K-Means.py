
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def loadDataSet(fileName):    # 通用函数，用来解析以 tab 键分隔的 floats（浮点数），例如: 1.658985	4.285136
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        print(curLine)
        fltLine = map(float, curLine)    # 映射所有的元素为 float（浮点数）类型
        dataMat.append(fltLine)
    return dataMat


if __name__ == '__main__':
    X = loadDataSet('testSet.txt')
    y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()

