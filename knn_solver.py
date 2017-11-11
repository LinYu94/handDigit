from numpy import *
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier

def toInt(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] == '0':
                data[i][j] = 0
            else:
                data[i][j] = 1
    return data


def loadTrainData():
    l=[]
    with open('./train.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line) #42001*785
    l.remove(l[0])

    label = [int(x[0]) for x in l]
    data = [x[1:] for x in l]
    data = toInt(data)
    return data, label

def loadTestData():
    l = []
    with open('./test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001*785
    l.remove(l[0])
    data = toInt(l)
    return data

def getResult(filename, predict):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['ImageId', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(predict)):
            writer.writerow({'ImageId': i+1, 'Label': predict[i]})

if __name__ == '__main__':
    train_data, train_label = loadTrainData()
    test_data = loadTestData()
    print('get the data')
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(train_data, train_label)

    print("begin...")
    test_predict = []

    predict = neigh.predict(test_data)

    getResult('./knn_result.csv', predict)

    print('xx')