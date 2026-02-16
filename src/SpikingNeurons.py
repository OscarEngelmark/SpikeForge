import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from matplotlib import pyplot as plt
from plotting_tools import rasterplot

@dataclass
class Synapse:
    weight: float           # [A] or [nA], sign encodes excitation/inhibition
    tau: float = 5e-3       # [s] — can be per-synapse
    i_syn: float = 0.0      # current state [A]

class Neuron:
    """Abstract base class"""
    def __init__(self, name: str ='Neuron', u_rest=-65e-3, u_reset=-65e-3, u_thres=-50e-3, R=95e6, tau_m=30e-3):
        self.name    = name
        self.u_rest  = u_rest
        self.u_reset = u_reset
        self.u_thres = u_thres
        self.R       = R
        self.tau_m   = tau_m
        self.u       = u_reset

    def integrate(self, dt: float):
        raise NotImplementedError

    def receive_spike(self, syn_idx: int, weight: Optional[float] = None):
        raise NotImplementedError

    def reset_if_spiked(self) -> bool:
        if self.u >= self.u_thres:
            self.u = self.u_reset
            return True
        return False

class InputNeuron(Neuron):
    """Simple spike source neuron that can be forced to spike at specified times.
    Does not integrate membrane potential; only exists to emit spikes on command.
    """
    def __init__(self, name: str = "InputNeuron", **kwargs):
        super().__init__(name=name, **kwargs)
        self.force_spike = False

    def set_force_spike(self, force: bool):
        self.force_spike = force

    def reset_if_spiked(self) -> bool:
        if self.force_spike:
            self.force_spike = False
            return True
        return False

    def integrate(self, dt: float):
        # No voltage dynamics, this neuron does not have a membrane
        pass

    def receive_spike(self, syn_idx: int, weight: Optional[float] = None):
        # Input neurons don't receive synapses
        pass

class CICNeuron(Neuron):
    """Constant (DC) current input"""
    def __init__(self, I_syn: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.I_syn = I_syn

    def integrate(self, dt: float):
        dudt = (self.u_rest - self.u + self.R * self.I_syn) / self.tau_m
        self.u += dt * dudt

class DynamicNeuron(Neuron):
    """Current-based exponential synapses"""
    def __init__(self, n_synapses: int, tau_syn: float = 50e-3, **kwargs):
        super().__init__(**kwargs)
        self.synapses: List[Synapse] = [Synapse(weight=0.0, tau=tau_syn) for _ in range(n_synapses)]
        self.force_spike = False

    def set_force_spike(self, force: bool):
        self.force_spike = force

    def reset_if_spiked(self) -> bool:
        if self.force_spike:
            self.u = self.u_reset
            self.force_spike = False
            return True

        if self.u >= self.u_thres:
            self.u = self.u_reset
            return True
        return False

    def receive_spike(self, syn_idx: int, weight: Optional[float] = None):
        if weight is not None:
            self.synapses[syn_idx].weight = weight
        self.synapses[syn_idx].i_syn += self.synapses[syn_idx].weight

    def integrate(self, dt: float):
        # Decay synaptic currents
        for syn in self.synapses:
            syn.i_syn -= dt * syn.i_syn / syn.tau

        # Total current
        I_total = sum(syn.i_syn for syn in self.synapses)
        dudt = (self.u_rest - self.u + self.R * I_total) / self.tau_m
        self.u += dt * dudt

@dataclass
class Connection:
    pre_idx: int
    post_idx: int
    syn_idx: int   # which synapse on the post neuron

class SpikingNetwork:
    def __init__(self):
        self.neurons: List[Neuron] = []
        self.connections: List[Connection] = []
        self.spikes: List[float] = []

    def add_neuron(self, neuron: Neuron) -> int:
        idx = len(self.neurons)
        self.neurons.append(neuron)
        return idx

    def add_neurons(self, neurons: List[Neuron]) -> List[int]:
        indices = []
        for neuron in neurons:
            idx = self.add_neuron(neuron)
            indices.append(idx)
        return indices

    def connect(self, pre: int, post: int, syn_idx: int):
        self.connections.append(Connection(pre, post, syn_idx))

    def step(self, dt: float) -> List[int]:
        spikes: List[int] = []

        # 1. Check which neurons spikes & reset (non-linear part)
        for i, n in enumerate(self.neurons):
            if n.reset_if_spiked():
                spikes.append(i)

        # 2. Propagate spikes → PSPs
        for c in self.connections:
            if c.pre_idx in spikes:
                post_n = self.neurons[c.post_idx]
                post_n.receive_spike(c.syn_idx)

        # 3. Integrate all neurons
        for n in self.neurons:
            n.integrate(dt)

        return spikes

    def simulate(
            self,
            dt: float,
            num_steps: int,
            tracked_neurons: List[int],
            input_spike_trains: Optional[np.ndarray] = None,
            input_neuron_ids: Optional[List[int]] = None
    ) -> Tuple[List[float], Dict[int, List[float]], List[int], List[int]]:

        t = 0.0

        # Store initial values
        timestamps: List[float] = [t]
        potentials: Dict[int, List[float]] = {idx: [self.neurons[idx].u] for idx in tracked_neurons}
        t_spike = []
        n_spike = []

        for step in range(num_steps):

            if input_spike_trains is not None:
                for row_idx, neuron_id in enumerate(input_neuron_ids):
                    if input_spike_trains[row_idx, step]:
                        self.neurons[neuron_id].set_force_spike(True)

            # Update the network
            spikes = self.step(dt)

            # Timestep completed
            t += dt
            timestamps.append(t)

            # Store membrane potentials of tracked neurons
            for neuron_id in tracked_neurons:
                potentials[neuron_id].append(self.neurons[neuron_id].u)

            # Record spikes
            for neuron_idx in spikes:
                t_spike.append(t)
                n_spike.append(neuron_idx)

        return timestamps, potentials, t_spike, n_spike


def main():
    # This implements Task 7 of Exercise SNN1

    n0 = CICNeuron(200e-12) # LIF neuron with constant input current of 200 pA

    n1 = DynamicNeuron(1) # LIF neuron with one dynamic synapse
    n2 = DynamicNeuron(2) # LIF neuron with two dynamic synapses

    snn = SpikingNetwork()

    neurons = [n0, n1, n2]  # A population of three neurons

    snn.add_neurons(neurons)

    t = 0  # Reset the simulation time

    snn.connect(pre=0, post=1, syn_idx=0) # n1 receives spikes from n0 on synapse 0
    snn.connect(pre=0, post=2, syn_idx=0) # n2 receives spikes from n0 on synapse 0
    snn.connect(pre=1, post=2, syn_idx=1) # n2 also receives spikes from n1 on synapse 1

    n1.synapses[0].weight = 300e-12 # Excitatory synapse with weight 300 pA
    n2.synapses[0].weight = 300e-12 # Excitatory synapse with weight 300 pA
    n2.synapses[1].weight = -100e-12

    dt = 1e-5
    T = 1

    t, U, t_spike, n_spike = snn.simulate(
        dt=dt,
        num_steps=int(T/dt),
        tracked_neurons=[0, 1, 2]
    )

    plt.rcParams['figure.figsize'] = [10, 10]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(t, U[1])
    ax1.set_ylabel('Membrane potential u1(t)')
    ax2.plot(t, U[2])
    ax2.set_ylabel('Membrane potential u2(t)')
    rasterplot(ax3, t_spike, n_spike, 'Time [s]', 'Neuron index')

    plt.show()

if __name__ == "__main__":
    main()