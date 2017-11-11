import numpy
from knn_solver import *
from sklearn import svm


if __name__ == '__main__':
    train_data, train_label = loadTrainData()
    test_data = loadTestData()
    print('begin')
    linear_svc = svm.SVC(kernel='rbf')
    linear_svc.fit(train_data, train_label)

    predict = linear_svc.predict(test_data)
    getResult('./svm_result.csv', predict)