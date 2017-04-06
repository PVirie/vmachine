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
        self.unit_size = unit_size
        self.W = tf.Variable(util.random_uniform(self.unit_size, self.unit_size), dtype=tf.float32)
        self.B = tf.Variable(np.zeros((self.unit_size)), dtype=tf.float32)
        self.input_bias = tf.Variable(np.zeros((unit_size)), dtype=tf.float32)

    def forward(self, input):
        output = input
        for i in xrange(0, self.depth):
            output = self.f(tf.matmul(self.f(tf.matmul(output, self.W) + self.B), self.W, transpose_b=True) + self.input_bias)
        return output

    def get_reset_operation(self):
        return (tf.assign(self.W, util.random_uniform(self.unit_size, self.unit_size)), tf.assign(self.B, np.zeros((self.unit_size))))

    def get_weights(self):
        return self.W, self.B


if __name__ == '__main__':
    sess = tf.Session()
    inputs = tf.constant(np.random.rand(5, 10), dtype=tf.float32)
    bnet = BeliefNet(10)
    outputs = bnet.forward(inputs)
    ops = util.cross_entropy(outputs, inputs, tf.trainable_variables())
    sess.run(tf.global_variables_initializer())

    for i in xrange(100):
        sess.run(ops)
    print sess.run(bnet.get_weights())
    sess.run(bnet.get_reset_operation())
    print sess.run(bnet.get_weights())
