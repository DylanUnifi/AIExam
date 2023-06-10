import math
import random

import numpy as np

from utils import Sigmoid, Softmax, CrossEntropy, iterate_minibatches, accuracy_score


class MultiLayerPerceptron:
    def __init__(self, n_hidden, n_iterations=3000, batch_size=1, learning_rate=0.01):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()
        self.batch_size = batch_size
        self.batch = []

    def _initialize_weights(self, X, y):
        n_samples, n_features = X.shape
        _, n_outputs = y.shape

        # Hidden layer
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.w0 = np.zeros((1, self.n_hidden))

        # Output layer
        limit = 1 / math.sqrt(self.n_hidden)
        self.V = np.random.uniform(-limit, limit, (self.n_hidden, n_outputs))
        self.v0 = np.zeros((1, n_outputs))

        for i in iterate_minibatches(X, y, self.batch_size, shuffle=True):
            self.batch.append(i)

    def fit(self, X, y, X_test, y_test):
        self._initialize_weights(X, y)
        y_test = np.argmax(y_test, axis=1)
        for n in range(self.n_iterations):
            X_bacth, y_batch = self.batch[random.randint(0, len(self.batch) - 1)]
            # ..............
            #  Forward Pass
            # ..............

            # HIDDEN LAYER
            hidden_input = X_bacth.dot(self.W) + self.w0
            #hidden_input = X.dot(self.W) + self.w0
            hidden_output = self.hidden_activation(hidden_input)
            # OUTPUT LAYER
            output_layer_input = hidden_output.dot(self.V) + self.v0
            y_pred = self.output_activation(output_layer_input)
            ############################################################

            # ...............
            #  Backward Pass
            # ...............

            # OUTPUT LAYER
            # Grad. with respect to input of output layer
            grad_wrt_out_l_input = self.loss.gradient(y_batch, y_pred) * self.output_activation.gradient(output_layer_input)
            #grad_wrt_out_l_input = self.loss.gradient(y, y_pred) * self.output_activation.gradient(output_layer_input)
            grad_v = hidden_output.T.dot(grad_wrt_out_l_input)
            grad_v0 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)

            # HIDDEN LAYER
            # Grad. with respect to input of hidden layer
            grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(self.V.T) * self.hidden_activation.gradient(hidden_input)
            grad_w = X_bacth.T.dot(grad_wrt_hidden_l_input)
            #grad_w = X.T.dot(grad_wrt_hidden_l_input)
            grad_w0 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)

            ############################################################

            # Update weights (by gradient descent)
            # Move against the gradient to minimize loss
            self.V -= self.learning_rate * grad_v
            self.v0 -= self.learning_rate * grad_v0
            self.W -= self.learning_rate * grad_w
            self.w0 -= self.learning_rate * grad_w0

            pred = self.predict(X_test)
            y_pred_test = np.argmax(pred, axis=1)
            accuracy = accuracy_score(y_test, y_pred_test)
            print(accuracy)



    def predict(self, X):
        hidden_input = X.dot(self.W) + self.w0
        hidden_output = self.hidden_activation(hidden_input)
        output_layer_input = hidden_output.dot(self.V) + self.v0
        y_pred = self.output_activation(output_layer_input)
        return y_pred

