import random
import numpy as np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    def stochastic_gradient_descent(self, training_data, batch_size, num_of_epochs, learn_rate, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(num_of_epochs):
            random.shuffle(training_data)
            mini_batches = []

            for k in range(0, n, batch_size):
                mini_batches.append(training_data[k:k+batch_size])

            for batch in mini_batches:
                self.update_batch(batch, learn_rate)

            if test_data:
                print("Epoch {0}: {1} / {2} ".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_batch(self, training_batch, learn_rate):
        m = len(training_batch)
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in training_batch:
            delta_grad_b, delta_grad_w = self.backprop(x,y)
            grad_b = [gb+dgb for gb, dgb in zip(grad_b, delta_grad_b)]
            grad_w = [gw+dgw for gw, dgw in zip(grad_w, delta_grad_w)]

        self.weights = [w - (learn_rate/m)*gw for w, gw in zip(self.weights, grad_w)]
        self.biases = [b - (learn_rate/m)*gb for b, gb in zip(self.weights, grad_b)]

    def backprop(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        current_activation = x
        all_activations = [x]
        all_zs = []

        # feed forward
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, current_activation) + b
            all_zs.append(z)
            current_activation = self.sigmoid(z)
            all_activations.append(current_activation)

        # initialize last row for back prop
        delta = self.cost_derivative(all_activations[-1], y) * self.sigmoid_prime(all_zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, all_activations[-2].transpose())

        # iterate through the rest of the net backwards
        for l in range(2, self.num_layers):
            z = all_zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, all_activations[-l+1].transpose())

        return grad_b, grad_w

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_prime(self, z):
        return (self.sigmoid(z))*(1-self.sigmoid(z))

    def cost_derivative(self, output_activations, actual):
        return output_activations - actual