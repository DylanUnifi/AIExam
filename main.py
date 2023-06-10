import numpy as np
import MNISTReader

from MultiLayerPerceptron import MultiLayerPerceptron
from utils import accuracy_score, to_categorical, iterate_minibatches

X_train, y_train = MNISTReader.load_mnist('data/fashion', kind='train')
X_test, y_test = MNISTReader.load_mnist('data/fashion', kind='t10k')

X_train = X_train / 255.
X_train = X_train.reshape([-1, 28*28])
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int32)

X_test = X_test / 255.
X_test = X_test.reshape([-1, 28*28])
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.int32)

# normalize(X_train)
# normalize(X_test)


selected_classes = [0, 1]
train_index = np.where(np.isin(y_train, selected_classes))[0]
test_index = np.where(np.isin(y_test, selected_classes))[0]
X_train = X_train[train_index]
y_train = y_train[train_index]
X_test = X_test[test_index]
y_test = y_train[test_index]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)




if __name__ == '__main__':
    clf = MultiLayerPerceptron(n_hidden=196, n_iterations=10, batch_size=12000, learning_rate=0.01)
    clf.fit(X_train, y_train, X_test, y_test)
    predict = clf.predict(X_test)
    y_pred = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>")
