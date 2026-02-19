# Lightweight Spiking Neural Network Framework in Python

A minimal **spiking neural network (SNN)** simulator written in Python.

The framework provides:
- Leaky Integrate-and-Fire (LIF) neurons
- Current-based exponential synaptic dynamics
- Scheduled (pre-defined) spike sources
- Flexible network construction via explicit connections
- Simple forward Euler time-stepping simulation
- Easy recording of membrane potentials

Designed to be **easy to read, modify and extend** — ideal for teaching SNN basics, prototyping small biologically-inspired circuits, or understanding core SNN mechanics without heavy abstractions or ML-framework overhead.

## Core Features

- Two neuron types:
  - `CICNeuron`: LIF with constant (DC) current injection
  - `DynamicNeuron`: LIF with multiple current-based exponential synapses (PSP decay)
- `ScheduledSpikeSource`: emits spikes at user-provided times
- Explicit `connect(pre, post, syn_idx)` syntax — very transparent wiring
- Time-step simulation with `step(dt)` or `simulate(dt, num_steps, tracked_neurons=…)`
- Generates timestamps, membrane potentials, and spike times/IDs for raster plotting
- Helper utilities for generating moving-bar-like spike patterns

## Files

- `SpikingNeurons.py`  
  Core simulation framework containing neuron and network classes

- `spike_generation.py`  
  Utilities to generate input spike data

- `plotting_tools.py`  
  Helper functions for plotting

- `direction_selective_network.py`  
  Builds and runs a direction-selective model inspired by [Neuronify](https://ovilab.net/neuronify/)