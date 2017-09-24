import tensorflow as tf
import numpy as np
import util


class Transceducer:

    def __init__(self, memory_size, context_size, recall_size, threshold):
        self.threshold = threshold
        self.unit_sizes = (memory_size, context_size, recall_size)
        self.C = tf.Variable(util.random_uniform(context_size, memory_size), dtype=tf.float32)
        self.R = tf.Variable(util.random_uniform(recall_size, memory_size), dtype=tf.float32)

        s = np.zeros((1, memory_size), dtype=np.float32)
        s[0, 0] = 1
        shifter = tf.constant(util.build_cpu_shift_mat(memory_size), dtype=tf.float32)
        self.seed = tf.Variable(s, dtype=tf.float32)
        self.reseed_ops = tf.assign(self.seed, tf.matmul(self.seed, shifter))

        self.reset_ops = []
        self.reset_ops.append(tf.assign(self.C, util.random_uniform(context_size, memory_size)))
        self.reset_ops.append(tf.assign(self.R, util.random_uniform(recall_size, memory_size)))

    def _extract_winner(self, h):
        val = tf.reduce_max(h, axis=1, keep_dims=True)
        loc = tf.one_hot(tf.argmax(h, axis=1), self.unit_sizes[0], 1.0, 0.0)
        return loc, val

    def infer(self, context):
        loc, val = self._extract_winner(tf.matmul(context, self.C))
        h_ = tf.multiply(loc, tf.where(val > self.threshold, val, tf.zeros(tf.shape(val))))
        v_ = tf.matmul(tf.clip_by_value(h_, 0.0, 1.0), self.R, transpose_b=True)
        return v_

    def gradients(self, context, recall):

        loc, val = self._extract_winner(tf.nn.softmax(tf.matmul(context, self.C), dim=-1))
        h = tf.where(
            tf.tile(val, [1, self.unit_sizes[0]]) > self.threshold,
            loc,
            tf.tile(self.seed, [tf.shape(context)[0], 1]))

        context_ = tf.matmul(h, self.C, transpose_b=True)
        recall_ = tf.matmul(h, self.R, transpose_b=True)

        grads = []
        grads.append((tf.matmul(-context + context_, h, transpose_a=True), self.C))
        grads.append((tf.matmul(-recall + recall_, h, transpose_a=True), self.R))

        return grads, tf.reduce_sum((recall - recall_)**2)

    def get_reset_operation(self):
        return self.reset_ops

    def get_reseed_operation(self):
        return self.reseed_ops


if __name__ == '__main__':

    sess = tf.Session()

    memory_size = 10
    context_size = 20
    recall_size = 10

    bnet = Transceducer(memory_size, context_size, recall_size, 0.5)

    contexts = []
    inputs = []
    outputs = []
    ops = []
    for i in xrange(memory_size):
        context = tf.constant(np.random.rand(1, context_size) - 0.5, dtype=tf.float32)
        input = tf.constant(np.random.rand(1, recall_size) - 0.5, dtype=tf.float32)
        output = bnet.infer(tf.nn.dropout(context, 0.5))
        grads, delta = bnet.gradients(context, input)
        # this memory learning rate can be huge (convex).
        ops.append(util.apply_gradients(grads, delta, 1.0))

        contexts.append(context)
        inputs.append(input)
        outputs.append(output)

    sess.run(tf.global_variables_initializer())

    reseed = bnet.get_reseed_operation()
    for i in xrange(memory_size):
        # shift memory anchor
        for j in xrange(100):
            print sess.run(ops[i])
        print sess.run(reseed)

    # sess.run(bnet.get_reset_operation())

    for i in xrange(memory_size):
        print sess.run(tf.reduce_sum(tf.squared_difference(outputs[i], inputs[i])))
