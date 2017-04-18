import tensorflow as tf
import matter
import hippocampus
import util


class Component:

    def __init__(self, component_size, input_size, total_past_steps, belief_depth, scope_name):
        self.sizes = {'input_size': input_size, 'component_size': component_size, 'scope_name': scope_name}

        with tf.variable_scope(scope_name):
            with tf.variable_scope("content") as content_scope:
                print content_scope.name
                self.content_scope = content_scope
                self.Gw = matter.Matter([total_past_steps * input_size, component_size, input_size * component_size])
                self.Ww = matter.Matter([component_size, component_size, component_size])
            with tf.variable_scope("selectivefocus") as selectivefocus_scope:
                print selectivefocus_scope.name
                self.selectivefocus_scope = selectivefocus_scope
                self.Sw = matter.Matter([total_past_steps * input_size, component_size, component_size])
            with tf.variable_scope("memory")as memory_scope:
                print memory_scope.name
                self.memory_scope = memory_scope
                self.Mw = hippocampus.BeliefNet(component_size)

    def generative_focus(self, pasts):
        return tf.nn.sigmoid(tf.reshape(self.Gw.forward(pasts), [self.sizes['input_size'], self.sizes['component_size']]))

    def selective_focus(self, pasts):
        """Focus as a part of the state"""
        return self.Sw.forward(pasts)

    def retrieve_memory(self, s):
        return self.Mw.backward(self.Mw.forward(s))

    def forward(self, v, G):
        x = tf.matmul(v, G)
        return self.Ww.forward(x)

    def backward(self, h, G):
        x = self.Ww.backward(h)
        return tf.matmul(x, G, transpose_b=True)

    def build_graphs(self, input, pasts, time):
        """component content h, selective focus s, memory m"""
        """return generated thought u, generated content v"""
        """v -> input; only when receiving an external input"""
        """m -> h; only when receiving an external input"""
        """s -> h; only when receiving an external input"""
        """s -> m; when perform thinking"""
        s = self.selective_focus(pasts)
        G = self.generative_focus(pasts)
        h = self.forward(input, G)
        v = self.backward(h, G)

        m = self.retrieve_memory(s)
        u = self.backward(m, G)

        grads, delta = self.Mw.gradients(h)
        self.memorize_operation = util.apply_gradients(grads, delta, 1.0)
        self.improve_focus_operation = util.cross_entropy(s, h, self.get_selective_focus_variables())
        self.reset_memory_operation = self.Mw.get_reset_operation()
        self.reseed_memory_operation = self.Mw.get_reseed_operation()

        return u, v

    def get_content_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.content_scope.name)

    def get_selective_focus_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.selectivefocus_scope.name)

    def get_memory_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.memory_scope.name)

    def get_memorize_operation(self):
        return self.memorize_operation

    def get_improve_focus_operation(self):
        return self.improve_focus_operation

    def get_reset_memory_operation(self):
        return self.reset_memory_operation

    def get_reseed_memory_operation(self):
        return self.reseed_memory_operation
