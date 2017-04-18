import tensorflow as tf
import matter
import hippocampus
import util
import lobe
import numpy as np


class Temporal_Component(lobe.Component):

    def __init__(self, component_size, input_size, total_past_steps, belief_depth, scope_name):
        self.sizes = {'input_size': input_size, 'component_size': component_size, 'scope_name': scope_name}
        self.temporal_memory_size = 10

        with tf.variable_scope(scope_name):
            with tf.variable_scope("content") as content_scope:
                print content_scope.name
                self.content_scope = content_scope
                self.Gw = matter.Matter([total_past_steps * input_size, component_size, input_size * component_size])
                self.Ww = matter.Matter([component_size, component_size, component_size])
            with tf.variable_scope("selectivefocus") as selectivefocus_scope:
                print selectivefocus_scope.name
                self.selectivefocus_scope = selectivefocus_scope
                self.Sw = matter.Matter([total_past_steps * input_size, self.temporal_memory_size, self.temporal_memory_size], activation=tf.nn.relu)
            with tf.variable_scope("memory")as memory_scope:
                print memory_scope.name
                self.memory_scope = memory_scope
                self.Mw = hippocampus.BeliefNet(component_size + self.temporal_memory_size)

    def update_counter(self, time):
        return tf.ones([1, self.temporal_memory_size], dtype=tf.float32) * time

    # not a true conditional query, should use conditional belief net instead.
    def retrieve_memory(self, s):
        query = tf.concat([tf.zeros([1, self.sizes['component_size']], dtype=tf.float32), s], 1)
        return tf.slice(lobe.Component.retrieve_memory(self, query), [0, 0], [-1, self.sizes['component_size']])

    # not a true conditional query, should use conditional belief net instead.
    def retrieve_time(self, h):
        query = tf.concat([h, tf.zeros([1, self.temporal_memory_size], dtype=tf.float32)], 1)
        return tf.slice(lobe.Component.retrieve_memory(self, query), [0, self.sizes['component_size']], [-1, -1])

    def build_graphs(self, input, pasts, time):

        s = self.selective_focus(pasts)
        G = self.generative_focus(pasts)
        h = self.forward(input, G)
        v = self.backward(h, G)

        t = self.retrieve_time(h)
        m = self.retrieve_memory(s)
        u = self.backward(m, G)

        # memorize the new memory,time tuple
        grads, delta = self.Mw.gradients(tf.concat([h, self.update_counter(time)], 1))
        self.memorize_operation = util.apply_gradients(grads, delta, 1.0)
        # tie the focus to best time
        self.improve_focus_operation = util.cross_entropy(s, t, self.get_selective_focus_variables())
        self.reset_memory_operation = self.Mw.get_reset_operation()
        self.reseed_memory_operation = self.Mw.get_reseed_operation()

        return u, v
