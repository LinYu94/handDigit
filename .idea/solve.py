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

def toInt2(array):
    array = mat(array)
    m, n = shape(array)
    newArray = zeros((m, n))
    for i in range(m):
        for j in range(n):
            if array[i,j] != '0':
                newArray[i, j] = 1
            else:
                newArray[i, j] = 0
    return newArray

def nomalizing(array):
    m,n=shape(array)
    for i in range(m):
        for j in range(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

def loadTrainData():
    l=[]
    with open('../train.csv') as file:
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
    with open('../test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001*785
    l.remove(l[0])
    data = toInt(l)
    return data

train_data, train_label = loadTrainData()
test_data = loadTrainData()

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_data, train_label)

test_predict = []
for x in test_data:
    predict = neigh.predict(x)
    print(predict)
    test_predict.append(predict)

print('xx')