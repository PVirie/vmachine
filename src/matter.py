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

    matter = Matter([input_size, input_size, input_size], tf.nn.elu)

    gens = []
    outputs = []
    ops = []
    for i in xrange(10):
        input = tf.constant(np.random.rand(1, input_size), dtype=tf.float32)
        output = tf.reshape(tf.one_hot(i, input_size, 1.0, 0.0, axis=-1, dtype=tf.float32), [1, -1])
        gen = matter.forward(input)
        ops.append(util.l2_loss(gen, output, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), 0.001))
        outputs.append(output)
        gens.append(gen)

    sess.run(tf.global_variables_initializer())

    for i in xrange(10):
        # print sess.run(bnet.get_reseed_operation())
        for j in xrange(100):
            print sess.run(ops[i])

    for i in xrange(10):
        print sess.run(tf.reduce_sum(tf.squared_difference(outputs[i], gens[i])))
