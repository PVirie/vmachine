import tensorflow as tf
import src.thalamus as core
import experiments.map as world
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cont", help="continue mode", action="store_true")
parser.add_argument("--rate", help="learning rate", type=float)
args = parser.parse_args()

if __name__ == '__main__':

    world_size = (60, 60)
    past_steps = 4
    components = 3

    sess = tf.Session()
    machine = core.Machine(sess, world_size[0] * world_size[1], past_steps, components)
    sess.run(tf.global_variables_initializer())

    if args.cont:
        machine.load_session("./artifacts/demo")

    # for i in xrange(4, inputs.shape[0]):
    #     pasts = inputs[i - 4:i, :]
    #     input_data = inputs[i:i + 1, :]
    #     print machine.generate_thought(pasts)
    #     print "-----------"
    #     # when the generated thought is far from the example
    #     machine.learn(input_data, pasts, "./artifacts/demo", 5)
    #     # otherwise memorize own thought for later use
    #     machine.improve_thinking(pasts, "./artifacts/demo", 5)
