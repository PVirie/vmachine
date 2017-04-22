import tensorflow as tf
import numpy as np
import util
import lobe
import temporal_lobe


class Machine:

    def __init__(self, sess, input_size, total_pasts, num_components, component_size, belief_depth):
        self.sess = sess
        self.input_initializer = tf.placeholder(tf.float32, [1, input_size])
        self.pasts_initializer = tf.placeholder(tf.float32, [total_pasts, input_size])

        self.input = tf.Variable(self.input_initializer, trainable=False, collections=[])
        self.pasts = tf.Variable(self.pasts_initializer, trainable=False, collections=[])

        self.generated_thoughts = tf.zeros([1, input_size], dtype=tf.float32)
        self.backward_thoughts = tf.zeros([1, input_size], dtype=tf.float32)
        self.improve_focus_operations = []
        self.memorize_operations = []
        self.reset_memory_operations = []
        self.reseed_memory_operations = []
        self.components = []

        self.components.append(temporal_lobe.Temporal_Component(component_size, input_size, total_pasts, 3, "C"))
        for i in xrange(1, num_components):
            self.components.append(lobe.Component(component_size, input_size, total_pasts, belief_depth, "C" + str(i)))

        for i in xrange(num_components):
            u, v = self.components[i].build_graphs(self.input, tf.reshape(self.pasts, [-1, total_pasts * input_size]))
            self.generated_thoughts = self.generated_thoughts + u
            self.backward_thoughts = self.backward_thoughts + u
            self.improve_focus_operations.append(self.components[i].get_improve_focus_operation())
            self.memorize_operations.append(self.components[i].get_memorize_operation())
            self.reset_memory_operations.append(self.components[i].get_reset_memory_operation())
            self.reseed_memory_operations.append(self.components[i].get_reseed_memory_operation())

        variables = [var for var in tf.global_variables() if "content" in var.name]
        # print [var.name for var in variables]
        self.learn_content_operation = util.l2_loss(self.backward_thoughts, self.input, variables, rate=0.01)

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    def learn(self, input, pasts, max_iteration=100, session_name=None):
        self.sess.run((self.input.initializer, self.pasts.initializer),
                      feed_dict={self.input_initializer: input, self.pasts_initializer: pasts})
        for step in xrange(max_iteration):
            v_, m_, f_ = self.sess.run((self.learn_content_operation, self.memorize_operations, self.improve_focus_operations))
        print "Content: ", v_
        print "Focus: ", f_
        print "Memory: ", m_
        print "-----------"
        self.sess.run(self.reseed_memory_operations)
        if session_name is not None:
            self.saver.save(self.sess, session_name)

    def generate_thought(self, pasts):
        self.sess.run((self.pasts.initializer),
                      feed_dict={self.pasts_initializer: pasts})
        return self.sess.run(self.generated_thoughts)

    def reset_memory(self):
        return self.sess.run(self.reset_memory_operations)

    def load_session(self, session_name):
        print "loading from last save..."
        self.saver.restore(self.sess, session_name)

    def load_last(self, directory):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(directory))


if __name__ == '__main__':
    sess = tf.Session()
    inputs = np.where(np.random.rand(10, 100) > 0.5, np.ones((10, 100), dtype=np.float32), 0)
    machine = Machine(sess, 100, 4, 3, 20, 5)
    sess.run(tf.global_variables_initializer())

    for i in xrange(4, inputs.shape[0]):
        pasts = inputs[i - 4:i, :]
        input_data = inputs[i:i + 1, :]
        print machine.generate_thought(pasts)
        print "-----------"
        # when the generated thought is far from the example
        machine.learn(input_data, pasts, 20)
