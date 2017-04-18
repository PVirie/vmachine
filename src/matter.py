import tensorflow as tf
import numpy as np
import util


class Matter:

    def __init__(self, layers, activation=tf.sigmoid):
        self.f = activation
        self.Ws = []
        self.Bs = []
        self.input_bias = tf.Variable(np.zeros((layers[0])), dtype=tf.float32)
        for i in xrange(1, len(layers)):
            self.Ws.append(tf.Variable(util.random_uniform(layers[i - 1], layers[i]), dtype=tf.float32))
            self.Bs.append(tf.Variable(np.zeros((layers[i])), dtype=tf.float32))

        self.reset_ops = []
        ops = []
        for i in xrange(1, len(layers)):
            ops.append(tf.assign(self.Ws[i - 1], util.random_uniform(layers[i - 1], layers[i])))
            ops.append(tf.assign(self.Bs[i - 1], np.zeros((layers[i]))))
        ops.append(tf.assign(self.input_bias, np.zeros((layers[0]))))

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

    def get_reset_operation(self):
        return self.reset_ops


if __name__ == '__main__':
    sess = tf.Session()

    input_size = 100
