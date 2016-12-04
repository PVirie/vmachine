import tensorflow as tf
import numpy as np
import util


class Matter:

    def __init__(self, layers, activation=tf.sigmoid, rbm=False):
        self.f = activation
        self.Ws = []
        self.Bs = []
        self.input_bias = tf.Variable(np.zeros((layers[0])), dtype=tf.float32)
        for i in xrange(1, len(layers)):
            self.Ws.append(tf.Variable(util.random_uniform(layers[i - 1], layers[i]), dtype=tf.float32))
            self.Bs.append(tf.Variable(np.zeros((layers[i])), dtype=tf.float32))

    def forward(self, input):
        output = input
        for i in xrange(0, len(self.Ws)):
            output = self.f(tf.matmul(output, self.Ws[i]) + self.Bs[i])
        return output

    def backward(self, output):
        input = output
        for i in xrange(len(self.Ws) - 1, 0, -1):
            input = self.f(tf.matmul(input, self.Ws[i], transpose_b=True) + self.Bs[i - 1])
        return self.f(tf.matmul(input, self.Ws[0], transpose_b=True) + self.input_bias)


class BeliefNet:

    def __init__(self, unit_size, activation=tf.sigmoid, depth=20):
        self.depth = depth
        self.f = activation
        self.W = tf.Variable(util.random_uniform(unit_size, unit_size), dtype=tf.float32)
        self.B = tf.Variable(np.zeros((unit_size)), dtype=tf.float32)
        self.input_bias = tf.Variable(np.zeros((unit_size)), dtype=tf.float32)

    def forward(self, input):
        output = input
        for i in xrange(0, self.depth):
            output = self.f(tf.matmul(self.f(tf.matmul(output, self.W) + self.B), self.W, transpose_b=True) + self.input_bias)
        return output
