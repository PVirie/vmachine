import tensorflow as tf
import numpy as np
import util


class Matter:
    # implement overcoming catastrophic forgetting based on https://arxiv.org/pdf/1612.00796.pdf

    def __init__(self, layers, activation=tf.sigmoid, penalty=0.1, mov=0.999):
        self.f = activation
        self.mov = mov
        self.penalty = penalty
        self.Ws = []
        self.Bs = []
        self.input_bias = tf.Variable(np.zeros((layers[0])), dtype=tf.float32)

        # each = (Fisher information, previous optimal parameters)
        self.vWs = []
        self.vBs = []
        self.vBias = (tf.Variable(np.zeros((layers[0])), dtype=tf.float32), tf.Variable(np.zeros((layers[0])), dtype=tf.float32))

        for i in xrange(1, len(layers)):
            self.Ws.append(tf.Variable(util.random_uniform(layers[i - 1], layers[i]), dtype=tf.float32))
            self.vWs.append((tf.Variable(np.zeros((layers[i - 1], layers[i])), dtype=tf.float32),
                             tf.Variable(np.zeros((layers[i - 1], layers[i])), dtype=tf.float32)))
            self.Bs.append(tf.Variable(np.zeros((layers[i])), dtype=tf.float32))
            self.vBs.append((tf.Variable(np.zeros((layers[i])), dtype=tf.float32),
                             tf.Variable(np.zeros((layers[i])), dtype=tf.float32)))

    def debug(self):
        return [x[0] for x in self.vWs]

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

    def gradients(self, cost):

        all_vars = self.Ws + self.Bs + [self.input_bias]
        all_Fs = self.vWs + self.vBs + [self.vBias]
        penalty_term = tf.constant(0, dtype=tf.float32)
        for i in xrange(len(all_vars)):
            penalty_term = penalty_term + tf.reduce_sum(tf.stop_gradient(all_Fs[i][0]) * tf.square(all_vars[i] - tf.stop_gradient(all_Fs[i][1])))
        grads = tf.gradients(cost + penalty_term * self.penalty, all_vars)

        return zip(grads, all_vars)

    def get_reseed_operation(self, cost):

        all_vars = self.Ws + self.Bs + [self.input_bias]
        all_Fs = self.vWs + self.vBs + [self.vBias]
        grads = tf.gradients(cost, all_vars)
        self.reseed_ops = []
        for i in xrange(len(grads)):
            if(grads[i] is None):
                continue
            self.reseed_ops.append(tf.assign(all_Fs[i][0], all_Fs[i][0] * self.mov + tf.square(grads[i]) * (1 - self.mov)))
            self.reseed_ops.append(tf.assign(all_Fs[i][1], all_vars[i]))

        return self.reseed_ops


if __name__ == '__main__':
    sess = tf.Session()

    input_size = 100
    matter = Matter([input_size, input_size, input_size], tf.nn.elu, penalty=1e8, mov=0.99)
    input = tf.placeholder(tf.float32, [1, input_size])
    output = tf.placeholder(tf.float32, [1, input_size])
    gen = matter.forward(input)
    cost = tf.reduce_sum(tf.squared_difference(gen, output))
    grads = util.apply_gradients(matter.gradients(cost), cost, 0.01)
    reseed = matter.get_reseed_operation(cost)
    # grads = util.l2_loss(output, gen, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), 0.01)

    sess.run(tf.global_variables_initializer())

    inputs = []
    outputs = []
    for i in xrange(10):
        inp = np.random.rand(1, input_size)
        out = np.multiply(inp, inp)
        inputs.append(inp)
        outputs.append(out)
        for j in xrange(500):
            print sess.run(grads, feed_dict={input: inp, output: out})
        sess.run(reseed, feed_dict={input: inp, output: out})
    sess.run(matter.debug())

    for i in xrange(10):
        print sess.run(tf.reduce_sum(tf.squared_difference(output, gen)), feed_dict={input: inputs[i], output: outputs[i]})
