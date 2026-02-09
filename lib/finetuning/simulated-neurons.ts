/**
 * Simulated Neurons
 * Biologically-inspired neural simulation with spiking dynamics
 */

export interface NeuronConfig {
  numNeurons: number;
  threshold: number; // Firing threshold
  refractoryPeriod: number; // Time after firing before neuron can fire again
  leakRate: number; // Membrane potential leak rate
  synapticStrength: number; // Initial synaptic weight strength
  plasticityRate: number; // Hebbian learning rate
  enableSTDP: boolean; // Spike-timing-dependent plasticity
}

export interface Neuron {
  id: number;
  potential: number; // Membrane potential
  lastSpikeTime: number;
  isRefractory: boolean;
  connections: Map<number, number>; // target neuron -> weight
  spikeHistory: number[]; // Recent spike times
}

/**
 * Simulated Neural Network with biological dynamics
 */
export class SimulatedNeuralNetwork {
  private neurons: Neuron[] = [];
  private config: NeuronConfig;
  private timeStep: number = 0;
  private globalActivity: number = 0;

  constructor(config: NeuronConfig) {
    this.config = config;
    this.initializeNeurons();
  }

  /**
   * Initialize neurons with random connections
   */
  private initializeNeurons(): void {
    // Create neurons
    for (let i = 0; i < this.config.numNeurons; i++) {
      this.neurons.push({
        id: i,
        potential: Math.random() * 0.1 - 0.05,
        lastSpikeTime: -Infinity,
        isRefractory: false,
        connections: new Map(),
        spikeHistory: [],
      });
    }

    // Create random connections (sparse connectivity like in brain)
    const connectionProbability = 0.1; // 10% connectivity
    for (let i = 0; i < this.config.numNeurons; i++) {
      for (let j = 0; j < this.config.numNeurons; j++) {
        if (i !== j && Math.random() < connectionProbability) {
          const weight = (Math.random() - 0.5) * this.config.synapticStrength;
          this.neurons[i].connections.set(j, weight);
        }
      }
    }
  }

  /**
   * Process input through the neural network
   */
  forward(input: Float32Array): Float32Array {
    const output = new Float32Array(input.length);

    // Apply input to first neurons
    const neuronsPerInput = Math.floor(this.config.numNeurons / input.length);
    
    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < neuronsPerInput; j++) {
        const neuronIdx = i * neuronsPerInput + j;
        if (neuronIdx < this.neurons.length) {
          this.neurons[neuronIdx].potential += input[i];
        }
      }
    }

    // Simulate neural dynamics for several time steps
    const simulationSteps = 10;
    for (let step = 0; step < simulationSteps; step++) {
      this.updateNeurons();
    }

    // Read output from last neurons
    for (let i = 0; i < input.length; i++) {
      let sum = 0;
      let count = 0;
      
      for (let j = 0; j < neuronsPerInput; j++) {
        const neuronIdx = this.config.numNeurons - (i * neuronsPerInput + j) - 1;
        if (neuronIdx >= 0) {
          sum += this.neurons[neuronIdx].potential;
          count++;
        }
      }
      
      output[i] = count > 0 ? sum / count : 0;
    }

    return output;
  }

  /**
   * Update all neurons for one time step
   */
  private updateNeurons(): void {
    this.timeStep++;
    const spikes: number[] = [];

    // Update each neuron
    for (const neuron of this.neurons) {
      // Check refractory period
      if (this.timeStep - neuron.lastSpikeTime < this.config.refractoryPeriod) {
        neuron.isRefractory = true;
        continue;
      } else {
        neuron.isRefractory = false;
      }

      // Leak membrane potential
      neuron.potential *= (1 - this.config.leakRate);

      // Check if neuron fires
      if (neuron.potential >= this.config.threshold && !neuron.isRefractory) {
        spikes.push(neuron.id);
        neuron.lastSpikeTime = this.timeStep;
        neuron.potential = 0; // Reset after spike
        neuron.spikeHistory.push(this.timeStep);
        
        // Keep only recent spike history
        if (neuron.spikeHistory.length > 100) {
          neuron.spikeHistory.shift();
        }
      }
    }

    // Propagate spikes through connections
    for (const sourceId of spikes) {
      const source = this.neurons[sourceId];
      
      for (const [targetId, weight] of source.connections) {
        this.neurons[targetId].potential += weight;
      }
    }

    // Apply Hebbian learning (neurons that fire together, wire together)
    if (this.config.enableSTDP) {
      this.applySTDP(spikes);
    }

    // Track global activity
    this.globalActivity = spikes.length / this.config.numNeurons;
  }

  /**
   * Apply Spike-Timing-Dependent Plasticity (STDP)
   * Strengthens connections between neurons that fire close in time
   */
  private applySTDP(recentSpikes: number[]): void {
    const stdpWindow = 20; // Time window for STDP

    for (const sourceId of recentSpikes) {
      const source = this.neurons[sourceId];
      const sourceTime = source.lastSpikeTime;

      for (const [targetId, weight] of source.connections) {
        const target = this.neurons[targetId];
        const targetTime = target.lastSpikeTime;
        
        if (targetTime > -Infinity) {
          const timeDiff = targetTime - sourceTime;
          
          if (Math.abs(timeDiff) < stdpWindow) {
            // Potentiation (strengthen) if target fires shortly after source
            // Depression (weaken) if target fires before source
            let deltaWeight = 0;
            
            if (timeDiff > 0) {
              // Target fired after source - strengthen connection
              deltaWeight = this.config.plasticityRate * Math.exp(-timeDiff / 10);
            } else {
              // Target fired before source - weaken connection
              deltaWeight = -this.config.plasticityRate * Math.exp(timeDiff / 10);
            }
            
            const newWeight = weight + deltaWeight;
            // Limit weight magnitude
            const clampedWeight = Math.max(-1.0, Math.min(1.0, newWeight));
            source.connections.set(targetId, clampedWeight);
          }
        }
      }
    }
  }

  /**
   * Apply Hebbian learning rule for fast adaptation
   * "Neurons that fire together, wire together"
   */
  hebbianLearning(input: Float32Array, target: Float32Array): void {
    // Find neurons that were active for input
    const activeInput = new Set<number>();
    const neuronsPerInput = Math.floor(this.config.numNeurons / input.length);
    
    for (let i = 0; i < input.length; i++) {
      if (input[i] > 0.5) {
        for (let j = 0; j < neuronsPerInput; j++) {
          activeInput.add(i * neuronsPerInput + j);
        }
      }
    }

    // Find neurons that should be active for target
    const activeTarget = new Set<number>();
    for (let i = 0; i < target.length; i++) {
      if (target[i] > 0.5) {
        for (let j = 0; j < neuronsPerInput; j++) {
          const neuronIdx = this.config.numNeurons - (i * neuronsPerInput + j) - 1;
          if (neuronIdx >= 0) {
            activeTarget.add(neuronIdx);
          }
        }
      }
    }

    // Strengthen connections between active neurons
    for (const sourceId of activeInput) {
      const source = this.neurons[sourceId];
      
      for (const targetId of activeTarget) {
        const currentWeight = source.connections.get(targetId) || 0;
        const newWeight = currentWeight + this.config.plasticityRate;
        source.connections.set(targetId, Math.min(1.0, newWeight));
      }
    }
  }

  /**
   * Get network statistics
   */
  getStats(): {
    averagePotential: number;
    firingRate: number;
    globalActivity: number;
    totalConnections: number;
    averageConnectionStrength: number;
  } {
    let totalPotential = 0;
    let recentFires = 0;
    let totalConnections = 0;
    let totalStrength = 0;

    for (const neuron of this.neurons) {
      totalPotential += Math.abs(neuron.potential);
      
      // Count recent fires (last 100 time steps)
      const recentSpikes = neuron.spikeHistory.filter(
        (time) => this.timeStep - time < 100
      );
      recentFires += recentSpikes.length;

      totalConnections += neuron.connections.size;
      for (const weight of neuron.connections.values()) {
        totalStrength += Math.abs(weight);
      }
    }

    return {
      averagePotential: totalPotential / this.config.numNeurons,
      firingRate: recentFires / (this.config.numNeurons * 100),
      globalActivity: this.globalActivity,
      totalConnections,
      averageConnectionStrength: totalConnections > 0 ? totalStrength / totalConnections : 0,
    };
  }

  /**
   * Reset neuron states
   */
  reset(): void {
    for (const neuron of this.neurons) {
      neuron.potential = 0;
      neuron.lastSpikeTime = -Infinity;
      neuron.isRefractory = false;
      neuron.spikeHistory = [];
    }
    this.timeStep = 0;
  }

  /**
   * Prune weak connections (like synaptic pruning in brain)
   */
  pruneConnections(threshold: number = 0.1): void {
    for (const neuron of this.neurons) {
      const toDelete: number[] = [];
      
      for (const [targetId, weight] of neuron.connections) {
        if (Math.abs(weight) < threshold) {
          toDelete.push(targetId);
        }
      }
      
      for (const targetId of toDelete) {
        neuron.connections.delete(targetId);
      }
    }
  }

  /**
   * Save network state
   */
  saveState(): any {
    return {
      neurons: this.neurons.map((n) => ({
        id: n.id,
        potential: n.potential,
        connections: Array.from(n.connections.entries()),
      })),
      timeStep: this.timeStep,
    };
  }

  /**
   * Load network state
   */
  loadState(state: any): void {
    this.timeStep = state.timeStep;
    
    for (let i = 0; i < state.neurons.length; i++) {
      this.neurons[i].potential = state.neurons[i].potential;
      this.neurons[i].connections = new Map(state.neurons[i].connections);
    }
  }
}
