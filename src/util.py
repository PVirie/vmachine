import numpy as np
import tensorflow as tf


def random_uniform(rows, cols):
    return (np.random.rand(rows, cols) - 0.5) * 0.001


def generate_ones(rows, cols):
    return np.ones((rows, cols), dtype=np.float32)


def cross_entropy(y, z, variables):
    cost = tf.reduce_sum(tf.mul(z, -tf.log(y)) + tf.mul((1 - z), -tf.log(1 - y)))
    training_op = tf.train.AdamOptimizer(0.001).minimize(cost, var_list=variables)
    return {"op": training_op, "cost": cost}


def l2_loss(y, z, variables):
    cost = tf.reduce_sum((z - y)**2)
    training_op = tf.train.AdamOptimizer(0.001).minimize(cost, var_list=variables)
    return {"op": training_op, "cost": cost}
