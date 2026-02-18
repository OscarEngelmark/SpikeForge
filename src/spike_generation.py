from typing import List, Tuple, Dict
from collections import defaultdict

def linear_movement(num_neurons: int, simulation_time: float, speed: float) -> Dict[int, List[float]]:
    """
    Generates input neuron spikes linearly distributed in time.
    :param num_neurons: Number of neurons in the series.
    :param simulation_time: Total simulation time.
    :param speed: Pulse movement speed [neurons/s].
    :return: Lists of input neuron spikes and the corresponding timestamps.
    """

    min_id = 0
    max_id = num_neurons - 1

    current_id = min_id
    step_dir = 1
    time_step = 1.0 / speed

    t = 0.0

    spike_times = defaultdict(list)

    # Generate and plot spikes from receptor neurons
    while t <= simulation_time:

        spike_times[current_id].append(t)

        current_id += step_dir
        t += time_step

        if current_id < min_id or current_id > max_id:
            step_dir *= -1
            current_id += 2 * step_dir

    return spike_times