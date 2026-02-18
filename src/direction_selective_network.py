from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from SpikingNeurons import ScheduledSpikeSource, DynamicNeuron, SpikingNetwork
from spike_generation import linear_movement

def main():
    """
    This function implements the direction-selective network from Neuronify.
    """

    # Simulation parameters
    dt = 0.1e-3  # 0.1 ms
    num_steps = 20000  # total steps
    sim_time = num_steps * dt

    # Neuron parameters
    kwargs = {
        'u_rest': -70e-3,
        'u_reset': -80e-3,
        'u_thres': -55e-3,
        'R': 100e6,
        'tau_m': 20e-3
    }
    tau_syn = 39e-3
    ex_weight = 350e-12
    in_weight = -250e-12

    # Create network instance
    snn = SpikingNetwork()

    # Generate spike times
    input_spike_times = linear_movement(num_neurons=5, simulation_time=sim_time, speed=18)

    # Define 5 input sources
    input_sources = [ScheduledSpikeSource(spike_times=input_spike_times[i]) for i in range(5)]
    input_ids = snn.add_sources(input_sources)

    # Define inhibitory neurons
    inhib_neurons = [DynamicNeuron(n_synapses=1, tau_syn=tau_syn, **kwargs) for _ in range(4)]
    inhib_ids = snn.add_neurons(inhib_neurons)

    # Define relay neurons
    relay_neurons = [DynamicNeuron(n_synapses=2, tau_syn=tau_syn, **kwargs) for _ in range(4)]
    relay_ids = snn.add_neurons(relay_neurons)

    # Define output neuron
    output_neuron = DynamicNeuron(n_synapses=4, tau_syn=tau_syn, **kwargs)
    output_id = snn.add_neuron(output_neuron)

    # Define connections
    # Connect sources to inhibitory neurons
    for idx, ID in enumerate(input_ids[:-1]):
        snn.connect(pre=ID, post=inhib_ids[idx], syn_idx=0)

    # Connect inhibitory neurons to relay neurons
    for idx, ID in enumerate(inhib_ids):
        snn.connect(pre=ID, post=relay_ids[idx], syn_idx=0)

    # Connect sources to relay neurons
    for idx, ID in enumerate(input_ids[1:]):
        snn.connect(pre=ID, post=relay_ids[idx], syn_idx=1)

    # Connect relay neurons to output neuron
    for idx, ID in enumerate(relay_ids):
        snn.connect(pre=ID, post=output_id, syn_idx=idx)

    # Set weights
    for idx in range(4):
        inhib_neurons[idx].synapses[0].weight = ex_weight
        relay_neurons[idx].synapses[0].weight = in_weight
        relay_neurons[idx].synapses[1].weight = ex_weight
        output_neuron.synapses[idx].weight = ex_weight

    # Run
    timestamps, potentials, spike_times, spike_ids = snn.simulate(
        dt=dt,
        num_steps=num_steps,
        tracked_neurons=[output_id] + relay_ids
    )

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, height_ratios=[2, .8])

    # 1. Raster plot (all spikes)
    axs[0].scatter(spike_times, spike_ids, marker='|', s=20, c='k', lw=1.8)
    axs[0].set_ylabel("Neuron ID")
    axs[0].yaxis.set_major_locator(MultipleLocator(1))
    axs[0].grid(True, alpha=0.5)

    # 2. Output voltage
    axs[1].plot(timestamps, potentials[output_id], label="Output", c='darkblue')
    axs[1].axhline(y=output_neuron.u_thres, color='r', ls='--', lw=0.9, label="threshold")
    axs[1].set_ylabel("u_output [V]")
    axs[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()






