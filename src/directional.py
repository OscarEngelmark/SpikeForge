from matplotlib import pyplot as plt
from SpikingNeurons import InputNeuron, DynamicNeuron, SpikingNetwork
from spike_generation import generate_spike_trains

def main():
    # Implementation of the direction-selective network from Neuronify

    # Simulation parameters
    direction = 'right'
    dt = 0.1e-3  # 0.1 ms
    num_steps = 3000  # total steps -> 600 ms

    # Spike train parameters
    pulse_start = int(50e-3 / dt)
    pulse_spacing = int(25e-3 / dt)
    pulse_width = 1

    # Neuron parameters
    kwargs = {
        'u_rest': -70e-3,
        'u_reset': -80e-3,
        'u_thres': -55e-3,
        'R': 100e6,
        'tau_m': 20e-3
    }
    tau_syn = 50e-3
    ex_weight = 300e-12
    in_weight = -250e-12

    # Create network instance
    snn = SpikingNetwork()

    # Define 5 input (forced spike) neurons
    input_neurons = [InputNeuron(**kwargs) for _ in range(5)]
    input_ids = snn.add_neurons(input_neurons)

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
    # Connect input neurons to inhibitory neurons
    for idx, ID in enumerate(input_ids[:-1]):
        snn.connect(pre=ID, post=inhib_ids[idx], syn_idx=0)

    # Connect inhibitory neurons to relay neurons
    for idx, ID in enumerate(inhib_ids):
        snn.connect(pre=ID, post=relay_ids[idx], syn_idx=0)

    # Connect input neurons to relay neurons
    for idx, ID in enumerate(input_ids[1:]):
        snn.connect(pre=ID, post=relay_ids[idx], syn_idx=1)

    # Connect relay neurons to output neuron
    for idx, ID in enumerate(relay_ids):
        snn.connect(pre=ID, post=output_id, syn_idx=idx)


    for idx in range(4):
        # Set weights for input -> inhib connections (excitatory)
        inhib_neurons[idx].synapses[0].weight = ex_weight  # Adjust value as needed

        # Set weights for inhib -> relay connections (inhibitory)
        relay_neurons[idx].synapses[0].weight = in_weight

        # Set weights for input -> relay connections (excitatory)
        relay_neurons[idx].synapses[1].weight = ex_weight

        # Set weights for relay -> output connections (excitatory)
        output_neuron.synapses[idx].weight = ex_weight

    # Generate input pattern matching exactly n_steps
    spike_trains = generate_spike_trains(
        inputs=5,
        direction=direction,
        num_steps=num_steps,
        pulse_start=pulse_start,
        pulse_spacing=pulse_spacing,
        pulse_width=pulse_width
    )

    # Run
    timestamps, potentials, t_spike, n_spike = snn.simulate(
        dt=dt,
        num_steps=num_steps,
        tracked_neurons=[output_id] + relay_ids,
        input_spike_trains=spike_trains,
        input_neuron_ids=input_ids
    )

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                            height_ratios=[2, 3, 1.5])

    # 1. Raster plot (all spikes)
    axs[0].scatter(t_spike, n_spike, marker='|', s=20, c='k', lw=1.2)
    axs[0].set_ylabel("Neuron index")

    # 2. Output voltage
    axs[1].plot(timestamps, potentials[output_id], label="Output", c='darkblue')
    axs[1].axhline(y=output_neuron.u_thres, color='r', ls='--', lw=0.9, label="threshold")
    axs[1].set_ylabel("u_output [V]")
    axs[1].legend(loc='upper right')

    # 3. Optional: example relay voltages (one or two)
    for rid in relay_ids[:2]:  # show first two relays as example
        axs[2].plot(timestamps, potentials[rid], label=f"Relay {rid}", lw=1.1)
    axs[2].set_ylabel("Relay u [V]")
    axs[2].set_xlabel("Time [s]")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()






