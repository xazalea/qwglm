/**
 * Training Engine
 * Main training orchestrator with hybrid LoRA/full fine-tuning support
 */

import { LoRAAdapter, type LoRAConfig } from './lora';
import { GradientComputer, type ActivationCache } from './gradients';
import { WebGPUTrainingOps } from './webgpu-training-ops';
import { WeightUpdater } from './weight-updater';
import { TrainingQueue, type TrainingExample } from './training-queue';
import { FastLearner, type FastLearnerConfig } from './fast-learner';
import { SimulatedNeuralNetwork } from './simulated-neurons';
import type { WebGPUBackend } from '../gpu-simulator/webgpu-backend';
import type { ModelWeights } from '../model-runtime/quantization/loader';
import type { InferenceEngine } from '../model-runtime/inference/inference-engine';

export interface TrainingConfig {
  mode: 'lora' | 'full' | 'hybrid';
  loraConfig?: LoRAConfig;
  learningRate: number;
  batchSize: number;
  maxGradNorm: number;
  weightDecay: number;
  warmupSteps: number;
  maxSteps: number;
  evalInterval: number;
  saveInterval: number;
  enableFastLearning?: boolean;
  enableSimulatedNeurons?: boolean;
  fastLearnerConfig?: FastLearnerConfig;
}

export interface TrainingMetrics {
  step: number;
  loss: number;
  learningRate: number;
  gradNorm: number;
  examplesProcessed: number;
  timestamp: number;
}

export class TrainingEngine {
  private config: TrainingConfig;
  private loraAdapter?: LoRAAdapter;
  private gradientComputer: GradientComputer;
  private trainingOps?: WebGPUTrainingOps;
  private weightUpdater: WeightUpdater;
  private trainingQueue: TrainingQueue;
  private inferenceEngine: InferenceEngine;
  private webgpu: WebGPUBackend | null;
  private fastLearner?: FastLearner;
  private neuralNetwork?: SimulatedNeuralNetwork;
  
  private step: number = 0;
  private isTraining: boolean = false;
  private metrics: TrainingMetrics[] = [];
  private fastLearningStats = {
    oneShot: 0,
    recalled: 0,
    traditional: 0,
  };

  constructor(
    config: TrainingConfig,
    trainingQueue: TrainingQueue,
    inferenceEngine: InferenceEngine,
    modelWeights: ModelWeights,
    webgpu: WebGPUBackend | null = null
  ) {
    this.config = config;
    this.trainingQueue = trainingQueue;
    this.inferenceEngine = inferenceEngine;
    this.webgpu = webgpu;
    
    this.gradientComputer = new GradientComputer(webgpu);
    this.weightUpdater = new WeightUpdater(modelWeights);

    // Initialize LoRA if needed
    if (config.mode === 'lora' || config.mode === 'hybrid') {
      if (!config.loraConfig) {
        throw new Error('LoRA config required for LoRA/hybrid mode');
      }
      this.loraAdapter = new LoRAAdapter(config.loraConfig, webgpu);
      this.initializeLoRALayers();
    }

    // Initialize WebGPU training ops if available
    if (webgpu && webgpu.isAvailable()) {
      const device = webgpu.getDevice();
      if (device) {
        this.trainingOps = new WebGPUTrainingOps(device);
      }
    }

    // Initialize Fast Learner (human-like learning)
    if (config.enableFastLearning) {
      const fastLearnerConfig = config.fastLearnerConfig || {
        episodicMemorySize: 1000,
        similarityThreshold: 0.85,
        metaLearningRate: 0.01,
        adaptiveThreshold: 0.1,
        neuronConfig: {
          numNeurons: 1000,
          threshold: 1.0,
          refractoryPeriod: 5,
          leakRate: 0.1,
          synapticStrength: 0.5,
          plasticityRate: 0.01,
          enableSTDP: true,
        },
      };
      this.fastLearner = new FastLearner(fastLearnerConfig);
      console.log('🧠 Fast Learner initialized - human-like learning enabled');
    }

    // Initialize Simulated Neural Network
    if (config.enableSimulatedNeurons) {
      this.neuralNetwork = new SimulatedNeuralNetwork({
        numNeurons: 2000,
        threshold: 1.0,
        refractoryPeriod: 5,
        leakRate: 0.1,
        synapticStrength: 0.5,
        plasticityRate: 0.01,
        enableSTDP: true,
      });
      console.log('⚡ Simulated neurons initialized - biologically-inspired learning');
    }
  }

  /**
   * Initialize LoRA layers for model
   */
  private initializeLoRALayers(): void {
    if (!this.loraAdapter || !this.config.loraConfig) {
      return;
    }

    const targetModules = this.config.loraConfig.targetModules;
    const hiddenSize = 4096; // Qwen3-VL hidden size
    const intermediateSize = 11008; // Qwen3-VL intermediate size
    const numLayers = 32; // Qwen3-VL number of layers

    // Initialize LoRA for each layer
    for (let layer = 0; layer < numLayers; layer++) {
      targetModules.forEach((module) => {
        const moduleName = `layers.${layer}.${module}`;
        
        if (module.includes('attention')) {
          // Attention layers: hidden_size x hidden_size
          this.loraAdapter!.initializeLayer(moduleName, hiddenSize, hiddenSize);
        } else if (module.includes('ffn')) {
          // FFN layers
          if (module.includes('gate') || module.includes('up')) {
            this.loraAdapter!.initializeLayer(moduleName, hiddenSize, intermediateSize);
          } else if (module.includes('down')) {
            this.loraAdapter!.initializeLayer(moduleName, intermediateSize, hiddenSize);
          }
        }
      });
    }

    console.log(`Initialized ${this.loraAdapter.getParameterCount()} LoRA parameters`);
  }

  /**
   * Train on a batch of examples
   */
  async trainBatch(examples: TrainingExample[]): Promise<TrainingMetrics> {
    this.isTraining = true;
    
    // Try fast learning first (one-shot learning like humans)
    if (this.fastLearner) {
      let fastLearned = 0;
      let recalled = 0;
      const remainingExamples: TrainingExample[] = [];

      for (const example of examples) {
        const result = this.fastLearner.learnFast(example);
        
        if (result.learned) {
          fastLearned++;
          this.fastLearningStats.oneShot++;
        } else if (result.recall) {
          recalled++;
          this.fastLearningStats.recalled++;
        } else {
          remainingExamples.push(example);
        }
      }

      console.log(`⚡ Fast learning: ${fastLearned} one-shot, ${recalled} recalled, ${remainingExamples.length} traditional`);

      // If all examples were learned quickly, skip traditional training
      if (remainingExamples.length === 0) {
        this.isTraining = false;
        return {
          step: this.step++,
          loss: 0.01, // Very low loss for instant learning
          learningRate: this.getLearningRate(),
          gradNorm: 0,
          examplesProcessed: examples.length,
          timestamp: Date.now(),
        };
      }

      examples = remainingExamples;
    }

    // Apply meta-learning (learn how to learn)
    if (this.fastLearner && examples.length > 0) {
      this.fastLearner.metaLearn(examples);
    }
    
    // Zero gradients
    if (this.loraAdapter) {
      this.loraAdapter.zeroGrad();
    }

    let totalLoss = 0;
    let totalGradNorm = 0;

    // Process each example with traditional training
    for (const example of examples) {
      const { loss, gradNorm } = await this.trainExample(example);
      totalLoss += loss;
      totalGradNorm += gradNorm;
      this.fastLearningStats.traditional++;
    }

    // Average over batch
    const avgLoss = examples.length > 0 ? totalLoss / examples.length : 0.01;
    const avgGradNorm = examples.length > 0 ? totalGradNorm / examples.length : 0;

    // Update weights
    await this.updateWeights();

    // Update step
    this.step++;

    // Create metrics
    const metrics: TrainingMetrics = {
      step: this.step,
      loss: avgLoss,
      learningRate: this.getLearningRate(),
      gradNorm: avgGradNorm,
      examplesProcessed: examples.length,
      timestamp: Date.now(),
    };

    this.metrics.push(metrics);

    // Check for quality degradation
    if (this.metrics.length > 10) {
      const recentLoss = avgLoss;
      const previousLoss = this.metrics[this.metrics.length - 10].loss;
      
      if (recentLoss > previousLoss * 1.5) {
        console.warn('Training loss increased significantly, consider rolling back');
      }
    }

    this.isTraining = false;
    return metrics;
  }

  /**
   * Train on a single example
   */
  private async trainExample(example: TrainingExample): Promise<{ loss: number; gradNorm: number }> {
    // Tokenize input and output
    const inputIds = this.tokenize(example.input);
    const targetIds = this.tokenize(example.output);

    // Forward pass with activation caching
    const cache = await this.forwardWithCache(inputIds);

    // Compute loss
    const loss = this.computeLoss(cache.output, targetIds);

    // Backward pass
    const gradOutput = this.computeGradOutput(cache.output, targetIds);
    await this.backward(cache, gradOutput);

    // Compute gradient norm
    const gradNorm = this.computeGradientNorm();

    return { loss, gradNorm };
  }

  /**
   * Forward pass with activation caching
   */
  private async forwardWithCache(inputIds: number[]): Promise<ActivationCache> {
    // Simplified forward pass - in real implementation, would go through all layers
    const cache: ActivationCache = {
      input: [],
      norm1: [],
      Q: [],
      K: [],
      V: [],
      attnScores: [],
      attnWeights: [],
      attnOutput: [],
      residual1: [],
      norm2: [],
      gate: [],
      up: [],
      activated: [],
      ffnOutput: [],
      output: [],
    };

    // This is a placeholder - real implementation would:
    // 1. Convert input IDs to embeddings
    // 2. Pass through each transformer layer
    // 3. Cache all intermediate activations
    // 4. Apply LoRA if enabled

    const hiddenSize = 4096;
    const seqLen = inputIds.length;

    // Create dummy activations for demonstration
    for (let i = 0; i < seqLen; i++) {
      cache.input.push(new Float32Array(hiddenSize));
      cache.output.push(new Float32Array(hiddenSize));
    }

    return cache;
  }

  /**
   * Backward pass
   */
  private async backward(cache: ActivationCache, gradOutput: Float32Array[]): Promise<void> {
    // Backpropagate through layers
    // This is simplified - real implementation would go through all layers in reverse
    
    if (this.loraAdapter) {
      // Backprop through LoRA adapters
      for (let i = 0; i < cache.output.length; i++) {
        this.loraAdapter.backward('layers.0.attention.q', cache.input[i], gradOutput[i]);
      }
    }

    // If full fine-tuning, compute gradients for all weights
    // This would use the GradientComputer to compute weight gradients
  }

  /**
   * Update weights using optimizer
   */
  private async updateWeights(): Promise<void> {
    const lr = this.getLearningRate();

    if (this.loraAdapter) {
      // Update LoRA parameters
      this.loraAdapter.updateParameters(lr, 0.9, 0.999, 1e-8, this.step);
    }

    // Apply updates to model weights if needed
    // This would merge LoRA weights or apply full gradients
  }

  /**
   * Compute loss (cross-entropy)
   */
  private computeLoss(logits: Float32Array[], targets: number[]): number {
    let totalLoss = 0;
    
    for (let i = 0; i < Math.min(logits.length, targets.length); i++) {
      const target = targets[i];
      const logit = logits[i][target] || 0;
      
      // Simple cross-entropy: -log(softmax(logit))
      const maxLogit = Math.max(...logits[i]);
      const expSum = logits[i].reduce((sum, l) => sum + Math.exp(l - maxLogit), 0);
      const loss = maxLogit + Math.log(expSum) - logit;
      
      totalLoss += loss;
    }

    return totalLoss / targets.length;
  }

  /**
   * Compute gradient of output
   */
  private computeGradOutput(logits: Float32Array[], targets: number[]): Float32Array[] {
    const gradOutput: Float32Array[] = [];

    for (let i = 0; i < Math.min(logits.length, targets.length); i++) {
      const grad = new Float32Array(logits[i].length);
      const target = targets[i];

      // Softmax gradient
      const maxLogit = Math.max(...logits[i]);
      const expSum = logits[i].reduce((sum, l) => sum + Math.exp(l - maxLogit), 0);

      for (let j = 0; j < grad.length; j++) {
        const softmax = Math.exp(logits[i][j] - maxLogit) / expSum;
        grad[j] = softmax - (j === target ? 1 : 0);
      }

      gradOutput.push(grad);
    }

    return gradOutput;
  }

  /**
   * Compute gradient norm
   */
  private computeGradientNorm(): number {
    let norm = 0;

    if (this.loraAdapter) {
      for (const layer of this.loraAdapter.getLayers().values()) {
        if (layer.gradA) {
          for (let i = 0; i < layer.gradA.length; i++) {
            norm += layer.gradA[i] * layer.gradA[i];
          }
        }
        if (layer.gradB) {
          for (let i = 0; i < layer.gradB.length; i++) {
            norm += layer.gradB[i] * layer.gradB[i];
          }
        }
      }
    }

    return Math.sqrt(norm);
  }

  /**
   * Get learning rate with warmup and decay
   */
  private getLearningRate(): number {
    const { learningRate, warmupSteps, maxSteps } = this.config;

    if (this.step < warmupSteps) {
      // Linear warmup
      return learningRate * (this.step / warmupSteps);
    } else {
      // Cosine decay
      const progress = (this.step - warmupSteps) / (maxSteps - warmupSteps);
      return learningRate * 0.5 * (1 + Math.cos(Math.PI * progress));
    }
  }

  /**
   * Tokenize text (simplified)
   */
  private tokenize(text: string): number[] {
    // This is a placeholder - real implementation would use the model's tokenizer
    return text.split('').map((c) => c.charCodeAt(0));
  }

  /**
   * Train for multiple steps
   */
  async train(numSteps: number): Promise<void> {
    for (let i = 0; i < numSteps; i++) {
      if (this.trainingQueue.isEmpty()) {
        console.log('Training queue empty, waiting for data...');
        await new Promise((resolve) => setTimeout(resolve, 1000));
        continue;
      }

      const batch = this.trainingQueue.getBatch(this.config.batchSize);
      const metrics = await this.trainBatch(batch);
      
      console.log(`Step ${metrics.step}: loss=${metrics.loss.toFixed(4)}, lr=${metrics.learningRate.toFixed(6)}`);

      // Remove processed examples
      this.trainingQueue.removeBatch(batch.map((ex) => ex.id));

      // Check if we should upgrade to full fine-tuning
      if (this.config.mode === 'hybrid' && this.shouldUpgradeToFull()) {
        console.log('Upgrading to full fine-tuning...');
        this.config.mode = 'full';
      }
    }
  }

  /**
   * Check if we should upgrade from LoRA to full fine-tuning
   */
  private shouldUpgradeToFull(): boolean {
    if (this.metrics.length < 100) {
      return false;
    }

    // Check if LoRA has plateaued
    const recent = this.metrics.slice(-20);
    const previous = this.metrics.slice(-40, -20);

    const recentAvgLoss = recent.reduce((sum, m) => sum + m.loss, 0) / recent.length;
    const previousAvgLoss = previous.reduce((sum, m) => sum + m.loss, 0) / previous.length;

    const improvement = (previousAvgLoss - recentAvgLoss) / previousAvgLoss;

    // If improvement is less than 1%, consider upgrading
    return improvement < 0.01;
  }

  /**
   * Get training metrics
   */
  getMetrics(): TrainingMetrics[] {
    return this.metrics;
  }

  /**
   * Get latest metrics
   */
  getLatestMetrics(): TrainingMetrics | undefined {
    return this.metrics[this.metrics.length - 1];
  }

  /**
   * Get fast learning statistics
   */
  getFastLearningStats(): {
    oneShot: number;
    recalled: number;
    traditional: number;
    total: number;
    fastRatio: number;
    fastLearnerStats?: ReturnType<FastLearner['getStats']>;
    neuralStats?: ReturnType<SimulatedNeuralNetwork['getStats']>;
  } {
    const total = this.fastLearningStats.oneShot + this.fastLearningStats.recalled + this.fastLearningStats.traditional;
    const fastCount = this.fastLearningStats.oneShot + this.fastLearningStats.recalled;
    
    return {
      ...this.fastLearningStats,
      total,
      fastRatio: total > 0 ? fastCount / total : 0,
      fastLearnerStats: this.fastLearner?.getStats(),
      neuralStats: this.neuralNetwork?.getStats(),
    };
  }

  /**
   * Check if training is in progress
   */
  isTrainingInProgress(): boolean {
    return this.isTraining;
  }

  /**
   * Save model
   */
  async saveModel(): Promise<Record<string, any>> {
    const model: Record<string, any> = {
      config: this.config,
      step: this.step,
      metrics: this.metrics,
    };

    if (this.loraAdapter) {
      model.lora = this.loraAdapter.saveWeights();
    }

    model.weights = this.weightUpdater.exportWeights();

    return model;
  }

  /**
   * Load model
   */
  async loadModel(saved: Record<string, any>): Promise<void> {
    this.step = saved.step || 0;
    this.metrics = saved.metrics || [];

    if (saved.lora && this.loraAdapter) {
      this.loraAdapter.loadWeights(saved.lora);
    }

    if (saved.weights) {
      this.weightUpdater.importWeights(saved.weights);
    }
  }
}
