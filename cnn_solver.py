import numpy as np
from knn_solver import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPooling2D



if __name__ == '__main__':
    model = Sequential()
    model.add(Conv2D(25, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(50, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 3)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])


    train_data, train_label = loadTrainData()
    test_data = loadTestData()
    # train_data = train_data[:100]
    # train_label = train_label[:100]
    # test_data = test_data[:10]

    train_data = np.array(train_data)
    t_label = np.zeros(len(train_label)*10).reshape(len(train_label),10)
    for i in range(len(train_label)):
        t_label[i][train_label[i]] = 1

    t_data = []
    for i in range(len(train_data)):
        t_data.append(train_data[i].reshape(28, 28, 1))
    t_data = np.array(t_data)

    print('begin train')
    model.fit(t_data, t_label, batch_size=32, epochs=20)

    test_data = np.array(test_data)
    te_data = []
    for i in range(len(test_data)):
        te_data.append(test_data[i].reshape(28, 28, 1))
    te_data = np.array(te_data)

    print('begin test')
    try:
        res = model.predict(te_data)
    except Exception as e:
        print(e.__str__())
    predict = []
    for item in res:
        index = 0
        max = item[0]
        for i in range(len(item)):
            if item[i] > max:
                max = item[i]
                index = i
        predict.append(index)
    print(predict)
    getResult('./cnn_result.csv', predict)
    print('begin')
