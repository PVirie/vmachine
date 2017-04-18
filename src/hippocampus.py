import tensorflow as tf
import numpy as np
import util


class BeliefNet:

    def __init__(self, unit_size, activation=tf.sigmoid, keep_prob=0.05):
        self.f = activation
        self.unit_size = unit_size
        self.keep_prob = keep_prob
        self.W = tf.Variable(np.zeros((self.unit_size, self.unit_size), dtype=np.float32), dtype=tf.float32)

        self.seed = tf.Variable(np.zeros((1, self.unit_size), dtype=np.float32), dtype=tf.float32)

        self.reset_ops = []
        self.reset_ops.append(tf.assign(self.W, np.zeros((self.unit_size, self.unit_size), dtype=np.float32)))

        self.reseed_ops = tf.assign(self.seed, util.tf_random_binomial(tf.ones([1, self.unit_size]) * keep_prob))

    def forward(self, input):
        h = tf.sigmoid(tf.matmul(input, self.W))
        h_ = util.tf_ones_or_zeros(h >= tf.reduce_max(h, 1, keep_dims=True))
        output = self.backward(h_)
        return output

    def backward(self, h):
        v_ = tf.matmul(h, self.W, transpose_b=True)
        return v_ / tf.reduce_sum(h, 1, keep_dims=True)

    def gradients(self, input):

        grads = []

        h = tf.tile(self.seed, [tf.shape(input)[0], 1])

        v = input
        v_ = self.backward(h)

        grads.append((- tf.matmul(v, h, transpose_a=True) + tf.matmul(v_, h, transpose_a=True), self.W))

        return grads, tf.reduce_sum((v - v_)**2)

    def get_reset_operation(self):
        return self.reset_ops

    def get_reseed_operation(self):
        return self.reseed_ops


if __name__ == '__main__':
    sess = tf.Session()

    input_size = 100

    bnet = BeliefNet(input_size, tf.sigmoid)

    inputs = []
    outputs = []
    ops = []
    for i in xrange(10):
        input = util.tf_random_binomial(tf.constant(np.random.rand(1, input_size), dtype=tf.float32))
        inputs.append(input)
        outputs.append(bnet.forward(input))
        grads, delta = bnet.gradients(input)
        ops.append(util.apply_gradients(grads, delta, 0.01))

    sess.run(tf.global_variables_initializer())

    for i in xrange(10):
        print sess.run(bnet.get_reseed_operation())
        for j in xrange(100):
            print sess.run(ops[i])

    # sess.run(bnet.get_reset_operation())

    for i in xrange(10):
        print sess.run(tf.reduce_sum(tf.squared_difference(outputs[i], inputs[i])))
