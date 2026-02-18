from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from matplotlib import pyplot as plt
from plotting_tools import rasterplot

@dataclass
class Synapse:
    weight: float           # [A] (sign encodes excitation/inhibition)
    tau: float = 5e-3       # [s] (can be per-synapse)
    i_syn: float = 0.0      # [A] (current state)

@dataclass
class Connection:
    pre_id: int
    post_id: int
    syn_id: int # which synapse on the post neuron

class SpikeSource:
    """Base for anything that can emit spikes without voltage dynamics."""

    def __init__(self, name: str = "SpikeSource"):
        self.name = name

    def wants_to_spike(self, t_start: float, t_end: float) -> bool:
        raise NotImplementedError

    def consume_emitted_spike(self):
        pass   # default: nothing (for forced/one-shot sources)

class ScheduledSpikeSource(SpikeSource):
    """
    Spike source that emits at pre-scheduled times (sparse, per-source list).
    Replaces ForcedSpikeSource for this example.
    """
    def __init__(self, spike_times: Optional[List[float]] = None, name: str = "ScheduledSource"):
        super().__init__(name)
        self.spike_times = sorted(spike_times or [])
        self._idx = 0

    def wants_to_spike(self, t_start: float, t_end: float) -> bool:
        if self._idx >= len(self.spike_times):
            return False
        next_t = self.spike_times[self._idx]
        return t_start <= next_t < t_end

    def consume_emitted_spike(self):
        self._idx += 1

class Neuron:
    """Abstract base class"""
    def __init__(self, name: str ='Neuron', u_rest=-65e-3, u_reset=-65e-3, u_thres=-50e-3, r=95e6, tau_m=30e-3):
        self.name    = name
        self.u_rest  = u_rest
        self.u_reset = u_reset
        self.u_thres = u_thres
        self.R       = r
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

class CICNeuron(Neuron):
    """Constant (DC) current input"""
    def __init__(self, i_syn: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.i_syn = i_syn

    def integrate(self, dt: float):
        dudt = (self.u_rest - self.u + self.R * self.i_syn) / self.tau_m
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

class SpikingNetwork:
    def __init__(self):
        self.neurons: List[Neuron] = []
        self.sources: List[SpikeSource] = []
        self.connections: List[Connection] = []
        self.t: float = 0.0

    @staticmethod
    def _negate_index(idx: int) -> int:
        return -(1 + idx)

    def add_neuron(self, neuron: Neuron) -> int:
        idx = len(self.neurons)
        self.neurons.append(neuron)
        return idx

    def add_neurons(self, neurons: List[Neuron]) -> List[int]:
        return [self.add_neuron(n) for n in neurons]

    def add_source(self, source: SpikeSource) -> int:
        idx = len(self.sources)
        self.sources.append(source)
        return self._negate_index(idx) # negative, descending

    def add_sources(self, sources: List[SpikeSource]) -> List[int]:
        return [self.add_source(s) for s in sources]

    def connect(self, pre: int, post: int, syn_idx: int):
        self.connections.append(Connection(pre, post, syn_idx))

    def step(self, dt: float) -> Tuple[List[int], List[int]]:
        neuron_spikes: List[int] = []
        source_spikes: List[int] = []

        t_start = self.t
        t_end = self.t + dt

        # 1. Collect source spikes in this interval
        for i, s in enumerate(self.sources):
            if s.wants_to_spike(t_start, t_end):
                source_spikes.append(self._negate_index(i))
                s.consume_emitted_spike()

        # 2. Collect neuron spikes and reset
        for i, n in enumerate(self.neurons):
            if n.reset_if_spiked():
                neuron_spikes.append(i)

        # 3. Deliver all spikes (sources + neurons)
        for c in self.connections:
            spikes = source_spikes if c.pre_id < 0 else neuron_spikes
            if c.pre_id in spikes:
                self.neurons[c.post_id].receive_spike(c.syn_id)

        # 4. Integrate all neurons
        for n in self.neurons:
            n.integrate(dt)

        self.t = t_end

        return source_spikes, neuron_spikes

    def simulate(
            self,
            dt: float,
            num_steps: int,
            tracked_neurons: List[int]
    ) -> Tuple[List[float], Dict[int, List[float]], List[float], List[int]]:

        # Store initial values
        timestamps: List[float] = [self.t]
        potentials: Dict[int, List[float]] = {idx: [self.neurons[idx].u] for idx in tracked_neurons}
        spike_times: List[float] = []
        spike_ids: List[int] = []

        for _ in range(num_steps):

            source_spikes, neuron_spikes = self.step(dt)

            # Record updated state
            timestamps.append(self.t)
            for neuron_id in tracked_neurons:
                potentials[neuron_id].append(self.neurons[neuron_id].u)

            # Record source spikes
            for source_id in source_spikes:
                spike_times.append(self.t)
                spike_ids.append(source_id) # negative

            # Record neuron spikes
            for neuron_id in neuron_spikes:
                spike_times.append(self.t)
                spike_ids.append(neuron_id)

        return timestamps, potentials, spike_times, spike_ids


def main():
    # This implements Task 7 of Exercise SNN1

    n0 = CICNeuron(200e-12) # LIF neuron with constant input current of 200 pA

    n1 = DynamicNeuron(1) # LIF neuron with one dynamic synapse
    n2 = DynamicNeuron(2) # LIF neuron with two dynamic synapses

    snn = SpikingNetwork()

    neurons = [n0, n1, n2]  # A population of three neurons

    snn.add_neurons(neurons)

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