import numpy as np
import src.thalamus as core

inputs = np.where(np.random.rand(20, 100) > 0.5, np.ones((20, 100), dtype=np.float32), 0)
machine = core.Machine(100, 4, 3)

for i in xrange(4, inputs.shape[0]):
    pasts = inputs[i - 4:i, :]
    input_data = inputs[i:i + 1, :]
    print machine.generate_thought(pasts)
    print "-----------"
    # when the generated thought is far from the example
    machine.learn(input_data, pasts, "./artifacts/demo", 5)
    # otherwise memorize own thought for later use
    machine.improve_thinking(pasts, "./artifacts/demo", 5)
