import tensorflow as tf
import util
import lobe


class Machine:

    def __init__(self, input_size, total_pasts, num_components):
        self.sess = tf.Session()
        self.input = tf.placeholder(tf.float32, [1, input_size])
        self.pasts = tf.placeholder(tf.float32, [total_pasts, input_size])

        self.generated_thoughts = tf.zeros([1, input_size], dtype=tf.float32)
        self.backward_thoughts = tf.zeros([1, input_size], dtype=tf.float32)
        self.improve_thinking_operations = []
        self.learn_operations = []
        self.components = []
        for i in xrange(num_components):
            self.components.append(lobe.Component(input_size / num_components, input_size, total_pasts, "C" + str(i)))
            u, v = self.components[i].build_graphs(self.input, tf.reshape(self.pasts, [-1, total_pasts * input_size]))
            self.generated_thoughts = self.generated_thoughts + u
            self.backward_thoughts = self.backward_thoughts + v
            self.improve_thinking_operations.append(self.components[i].get_improve_thinking_operation())
            self.learn_operations.append(self.components[i].get_learn_operation())

        variables = [var for var in tf.global_variables() if "content" in var.name]
        self.learn_content_operation = util.l2_loss(self.backward_thoughts, self.input, variables)

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        # Before starting, initialize the variables.  We will 'run' this first.
        init = tf.global_variables_initializer()

        # Launch the graph.
        self.sess.run(init)

    def improve_thinking(self, pasts, session_name, max_iteration=100):
        for step in xrange(max_iteration):
            clist = self.sess.run(self.improve_thinking_operations, feed_dict={self.pasts: pasts})
            print clist
            print "-----------"
            if step % 100 == 0:
                self.saver.save(self.sess, session_name)

    def learn(self, input, pasts, session_name, max_iteration=100):
        for step in xrange(max_iteration):
            v_, clist = self.sess.run((self.learn_content_operation, self.learn_operations), feed_dict={self.input: input, self.pasts: pasts})
            print v_
            print clist
            print "-----------"
            if step % 100 == 0:
                self.saver.save(self.sess, session_name)

    def generate_thought(self, pasts):
        return self.sess.run(self.generated_thoughts, feed_dict={self.pasts: pasts})

    def load_session(self, session_name):
        print "loading from last save..."
        self.saver.restore(self.sess, session_name)

    def load_last(self, directory):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(directory))
