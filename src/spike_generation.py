from typing import List, Tuple

def linear_movement(input_ids: List[int], simulation_time: float, speed: float) -> Tuple[List[float], List[int]]:
    """
    Generates input neuron spikes linearly distributed in time.
    :param input_ids: Input neuron ids.
    :param simulation_time: Total simulation time.
    :param speed: Pulse movement speed [neurons/s].
    :return: Lists of input neuron spikes and the corresponding timestamps.
    """

    timestamps = []
    spikes = []

    min_id = input_ids[0]
    max_id = input_ids[-1]

    current_id = min_id
    step_dir = 1
    time_step = 1.0 / speed

    t = 0.0

    # Generate and plot spikes from receptor neurons
    while t <= simulation_time:

        timestamps.append(t)
        spikes.append(current_id)

        current_id += step_dir
        t += time_step

        if current_id < min_id or current_id > max_id:
            step_dir *= -1
            current_id += 2 * step_dir

    return timestamps, spikes