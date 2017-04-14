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
            h = self.f(tf.matmul(output, self.W) + self.B)
            output = self.f(tf.matmul(h, self.W, transpose_b=True) + self.input_bias)
        return output

    def gradients(self, input):

        grads = []

        v = input
        h = self.f(tf.matmul(v, self.W) + self.B)

        v_ = self.forward(input)
        h_ = self.f(tf.matmul(v_, self.W) + self.B)

        grads.append((- tf.matmul(v, h, transpose_a=True) + tf.matmul(v_, h_, transpose_a=True), self.W))
        grads.append((- tf.reduce_sum(h, 0) + tf.reduce_sum(h_, 0), self.B))
        grads.append((- tf.reduce_sum(v, 0) + tf.reduce_sum(v_, 0), self.input_bias))

        return grads, tf.reduce_sum((v - v_)**2)

    def get_reset_operation(self):
        return (tf.assign(self.W, util.random_uniform(self.unit_size, self.unit_size)), tf.assign(self.B, np.zeros((self.unit_size))))

    def get_weights(self):
        return self.W, self.B


if __name__ == '__main__':
    sess = tf.Session()

    input_size = 100

    bnet = BeliefNet(input_size, tf.sigmoid, depth=10)

    inputs = []
    outputs = []
    ops = []
    for i in xrange(10):
        input = tf.constant(np.random.rand(1, input_size), dtype=tf.float32)
        inputs.append(input)
        outputs.append(bnet.forward(input))
        grads, delta = bnet.gradients(input)
        ops.append(util.apply_gradients(grads, delta, 0.01))

    sess.run(tf.global_variables_initializer())

    for i in xrange(10):
        for j in xrange(1000):
            print sess.run(ops[i])

    # sess.run(bnet.get_reset_operation())

    for i in xrange(10):
        print sess.run(tf.reduce_sum(tf.squared_difference(outputs[i], inputs[i])))
