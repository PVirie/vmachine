import tensorflow as tf
import numpy as np
import util


def build_cpu_shift_mat(size):
    r12340 = np.arange(1, size + 1, 1, dtype=np.int32)
    r12340[size - 1] = 0
    cpu_shift = np.identity(size)[:, r12340]
    return cpu_shift


class BeliefNet:

    def __init__(self, unit_size):
        self.unit_size = unit_size
        self.W = tf.Variable(util.random_uniform(unit_size, unit_size), dtype=tf.float32)

        s = np.zeros((1, unit_size), dtype=np.float32)
        s[0, 0] = 1
        shifter = tf.constant(build_cpu_shift_mat(unit_size), dtype=tf.float32)
        self.seed = tf.Variable(s, dtype=tf.float32)
        self.reseed_ops = tf.assign(self.seed, tf.matmul(self.seed, shifter))

        self.reset_ops = []
        self.reset_ops.append(tf.assign(self.W, util.random_uniform(unit_size, unit_size)))

    def forward(self, input):
        h = tf.matmul(input, self.W)
        h_ = tf.nn.softmax(h, dim=-1)
        return h_

    def backward(self, h):
        v_ = tf.matmul(h, self.W, transpose_b=True)
        return v_

    def gradients(self, input):

        grads = []

        v = input
        h = tf.tile(self.seed, [tf.shape(input)[0], 1])
        v_ = self.backward(h)

        grads.append((tf.matmul(-v + v_, h, transpose_a=True), self.W))

        return grads, tf.reduce_sum((v - v_)**2)

    def get_reset_operation(self):
        return self.reset_ops

    def get_reseed_operation(self):
        return self.reseed_ops


if __name__ == '__main__':
    sess = tf.Session()

    input_size = 100

    bnet = BeliefNet(input_size)

    inputs = []
    outputs = []
    ops = []
    for i in xrange(10):
        input = tf.constant(np.random.rand(1, input_size), dtype=tf.float32)
        output = bnet.backward(bnet.forward(tf.nn.dropout(input, 0.5)))
        grads, delta = bnet.gradients(input)
        # this memory learning rate can be huge (convex).
        ops.append(util.apply_gradients(grads, delta, 1.0))
        inputs.append(input)
        outputs.append(output)

    sess.run(tf.global_variables_initializer())

    for i in xrange(10):
        # shift memory anchor
        print sess.run(bnet.get_reseed_operation())
        for j in xrange(100):
            print sess.run(ops[i])

    # sess.run(bnet.get_reset_operation())

    for i in xrange(10):
        print sess.run(tf.reduce_sum(tf.squared_difference(outputs[i], inputs[i])))
