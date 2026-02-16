import numpy as np

def generate_spike_trains(
        inputs: int,
        direction: str = 'left',
        num_steps: int = 5000,
        pulse_start: int = 500,
        pulse_spacing: int = 400,
        pulse_width: int = 100
) -> np.ndarray:
    spike_trains = np.zeros((inputs, num_steps), dtype=bool)  # shape: (5 inputs, time)

    order = list(range(inputs))

    if direction == 'left':
        order.reverse()

    for i, neuron_idx in enumerate(order):
        start = pulse_start + i * pulse_spacing
        end   = start + pulse_width
        spike_trains[neuron_idx, start:end] = True

    return spike_trains  # binary (T, 5)