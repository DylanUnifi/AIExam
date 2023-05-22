import numpy as np
import MNISTReader
import matplotlib.pyplot as plt

from MultiLayerPerceptron import MultiLayerPerceptron
from utils import accuracy_score, to_categorical, normalize


X_train, y_train = MNISTReader.load_mnist('data/fashion', kind='train')
X_test, y_test = MNISTReader.load_mnist('data/fashion', kind='t10k')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


if __name__ == '__main__':
    clf = MultiLayerPerceptron(n_hidden=196, n_iterations=1000, learning_rate=0.01)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    y_pred = np.argmax(predict, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>")

