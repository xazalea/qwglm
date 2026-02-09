/**
 * Fast Learner
 * Human-like learning with meta-learning and episodic memory
 */

import { SimulatedNeuralNetwork, type NeuronConfig } from './simulated-neurons';
import type { TrainingExample } from './training-queue';

export interface EpisodicMemory {
  id: string;
  input: Float32Array;
  output: Float32Array;
  context: string;
  timestamp: number;
  accessCount: number;
  lastAccessed: number;
  importance: number;
}

export interface FastLearnerConfig {
  episodicMemorySize: number;
  similarityThreshold: number;
  metaLearningRate: number;
  adaptiveThreshold: number;
  neuronConfig: NeuronConfig;
}

/**
 * Fast Learner with human-like learning capabilities
 */
export class FastLearner {
  private episodicMemory: EpisodicMemory[] = [];
  private neuralNetwork: SimulatedNeuralNetwork;
  private config: FastLearnerConfig;
  private learningHistory: { error: number; time: number }[] = [];

  constructor(config: FastLearnerConfig) {
    this.config = config;
    this.neuralNetwork = new SimulatedNeuralNetwork(config.neuronConfig);
  }

  /**
   * Learn from example with human-like speed (one-shot learning)
   */
  learnFast(example: TrainingExample): { learned: boolean; recall: boolean } {
    // Encode example as neural patterns
    const input = this.encodeText(example.input);
    const target = this.encodeText(example.output);

    // Check episodic memory for similar examples
    const similar = this.findSimilar(input);
    
    if (similar) {
      // Already learned something similar - quick recall
      similar.accessCount++;
      similar.lastAccessed = Date.now();
      similar.importance *= 1.1; // Increase importance
      return { learned: false, recall: true };
    }

    // New experience - store in episodic memory
    const memory: EpisodicMemory = {
      id: example.id,
      input,
      output: target,
      context: example.input,
      timestamp: Date.now(),
      accessCount: 1,
      lastAccessed: Date.now(),
      importance: 1.0,
    };

    this.episodicMemory.push(memory);

    // Apply Hebbian learning for instant connection strengthening
    this.neuralNetwork.hebbianLearning(input, target);

    // Manage memory size
    if (this.episodicMemory.length > this.config.episodicMemorySize) {
      this.consolidateMemory();
    }

    return { learned: true, recall: false };
  }

  /**
   * Meta-learning: Learn how to learn faster
   */
  metaLearn(examples: TrainingExample[]): void {
    // Track prediction errors across examples
    const errors: number[] = [];

    for (const example of examples) {
      const input = this.encodeText(example.input);
      const target = this.encodeText(example.output);
      
      // Predict using neural network
      const prediction = this.neuralNetwork.forward(input);
      
      // Compute error
      const error = this.computeError(prediction, target);
      errors.push(error);

      // Fast adaptation based on error
      if (error > this.config.adaptiveThreshold) {
        // High error - learn quickly
        const adaptiveLearningRate = this.config.metaLearningRate * (1 + error);
        this.neuralNetwork.hebbianLearning(input, target);
      }
    }

    // Update learning history
    this.learningHistory.push({
      error: errors.reduce((a, b) => a + b, 0) / errors.length,
      time: Date.now(),
    });

    // Adapt learning strategy based on history
    this.adaptLearningStrategy();
  }

  /**
   * Recall from episodic memory (instant retrieval like human memory)
   */
  recall(query: string): string | null {
    const queryEncoding = this.encodeText(query);
    const similar = this.findSimilar(queryEncoding);

    if (similar) {
      similar.accessCount++;
      similar.lastAccessed = Date.now();
      return similar.context;
    }

    return null;
  }

  /**
   * Predict output using neural network and episodic memory
   */
  predict(input: string): Float32Array {
    const inputEncoding = this.encodeText(input);
    
    // Check episodic memory first (fast recall)
    const similar = this.findSimilar(inputEncoding);
    if (similar) {
      similar.accessCount++;
      similar.lastAccessed = Date.now();
      return similar.output;
    }

    // Use neural network for novel inputs
    return this.neuralNetwork.forward(inputEncoding);
  }

  /**
   * Find similar memory using cosine similarity
   */
  private findSimilar(input: Float32Array): EpisodicMemory | null {
    let bestMatch: EpisodicMemory | null = null;
    let bestSimilarity = 0;

    for (const memory of this.episodicMemory) {
      const similarity = this.cosineSimilarity(input, memory.input);
      
      if (similarity > this.config.similarityThreshold && similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestMatch = memory;
      }
    }

    return bestMatch;
  }

  /**
   * Cosine similarity between two vectors
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) return 0;

    return dotProduct / (normA * normB);
  }

  /**
   * Encode text as neural activation pattern
   */
  private encodeText(text: string): Float32Array {
    // Simple encoding: character frequencies and bigrams
    const encoding = new Float32Array(256);
    
    for (let i = 0; i < text.length; i++) {
      const charCode = text.charCodeAt(i) % 256;
      encoding[charCode] += 1.0 / text.length;
    }

    // Normalize
    let sum = 0;
    for (let i = 0; i < encoding.length; i++) {
      sum += encoding[i];
    }
    
    if (sum > 0) {
      for (let i = 0; i < encoding.length; i++) {
        encoding[i] /= sum;
      }
    }

    return encoding;
  }

  /**
   * Compute prediction error
   */
  private computeError(prediction: Float32Array, target: Float32Array): number {
    let error = 0;
    const length = Math.min(prediction.length, target.length);

    for (let i = 0; i < length; i++) {
      const diff = prediction[i] - target[i];
      error += diff * diff;
    }

    return Math.sqrt(error / length);
  }

  /**
   * Consolidate memory (like sleep in humans)
   * Keeps important memories, discards less important ones
   */
  private consolidateMemory(): void {
    // Sort by importance (combination of access count and recency)
    this.episodicMemory.sort((a, b) => {
      const scoreA = a.importance * Math.log(a.accessCount + 1) * 
                     (1.0 / (Date.now() - a.lastAccessed + 1));
      const scoreB = b.importance * Math.log(b.accessCount + 1) * 
                     (1.0 / (Date.now() - b.lastAccessed + 1));
      return scoreB - scoreA;
    });

    // Keep top memories
    this.episodicMemory = this.episodicMemory.slice(0, this.config.episodicMemorySize);
  }

  /**
   * Adapt learning strategy based on performance
   */
  private adaptLearningStrategy(): void {
    if (this.learningHistory.length < 10) return;

    const recent = this.learningHistory.slice(-10);
    const avgError = recent.reduce((sum, h) => sum + h.error, 0) / recent.length;

    // If error is decreasing, we're learning well
    const trend = recent[recent.length - 1].error - recent[0].error;

    if (trend < 0) {
      // Learning well - can be more aggressive
      this.config.metaLearningRate *= 1.05;
    } else {
      // Not learning well - be more conservative
      this.config.metaLearningRate *= 0.95;
    }

    // Clamp learning rate
    this.config.metaLearningRate = Math.max(0.001, Math.min(0.1, this.config.metaLearningRate));
  }

  /**
   * Get learning statistics
   */
  getStats(): {
    episodicMemorySize: number;
    averageAccessCount: number;
    learningRate: number;
    recentError: number;
    neuralStats: ReturnType<SimulatedNeuralNetwork['getStats']>;
  } {
    const totalAccess = this.episodicMemory.reduce((sum, m) => sum + m.accessCount, 0);
    const recentError = this.learningHistory.length > 0 
      ? this.learningHistory[this.learningHistory.length - 1].error 
      : 0;

    return {
      episodicMemorySize: this.episodicMemory.length,
      averageAccessCount: this.episodicMemory.length > 0 ? totalAccess / this.episodicMemory.length : 0,
      learningRate: this.config.metaLearningRate,
      recentError,
      neuralStats: this.neuralNetwork.getStats(),
    };
  }

  /**
   * Clear episodic memory
   */
  clearMemory(): void {
    this.episodicMemory = [];
  }

  /**
   * Export episodic memory
   */
  exportMemory(): any {
    return {
      memories: this.episodicMemory,
      learningHistory: this.learningHistory,
      neuralState: this.neuralNetwork.saveState(),
    };
  }

  /**
   * Import episodic memory
   */
  importMemory(data: any): void {
    this.episodicMemory = data.memories || [];
    this.learningHistory = data.learningHistory || [];
    if (data.neuralState) {
      this.neuralNetwork.loadState(data.neuralState);
    }
  }
}
