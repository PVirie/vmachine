import tensorflow as tf
import src.thalamus as core
import experiments.map as world
import argparse
import src.util as util
import os

parser = argparse.ArgumentParser()
parser.add_argument("--reset", help="reset weight", action="store_true")
parser.add_argument("--rate", help="learning rate", type=float)
parser.add_argument("--total", help="total maps", type=int)
args = parser.parse_args()

if __name__ == '__main__':

    world_size = (20, 20)
    past_steps = 2
    components = 2
    belief_depth = 5
    total_maps = 5 if not args.total else args.total

    print "-----------------------"
    print "world size: " + world_size
    print "past steps: " + past_steps
    print "components: " + components
    print "memory depth: " + belief_depth
    print "total maps: " + total_maps
    print "-----------------------"

    sess = tf.Session()
    machine = core.Machine(sess, world_size[0] * world_size[1], past_steps, components, belief_depth)
    sess.run(tf.global_variables_initializer())

    if not args.reset:
        print "loading..."
        machine.load_session("./artifacts/demo")

    for m in xrange(total_maps):
        frames = world.get_valid_data(world_size, 2, map_complexity=6, length_modifier=0.2)
        machine.reset_memory()

        generated_frames = []
        for i in xrange(1, frames.shape[0]):
            pasts = util.prepare_data(frames, i - past_steps, i)
            input_data = util.prepare_data(frames, i, i + 1)
            generated_frames.append(machine.generate_thought(pasts))
            print "-----------"
            # when the generated thought is far from the example
            machine.learn(input_data, pasts, "./artifacts/demo", 5)
            # otherwise memorize own thought for later use
            machine.improve_thinking(pasts, "./artifacts/demo", 5)

    artifact_path = os.path.dirname(os.path.abspath(__file__)) + "/artifacts/"
    world.toGif(world.to_numpy(generated_frames, world_size), artifact_path + "sample_path.gif")
