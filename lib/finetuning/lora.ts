/**
 * LoRA (Low-Rank Adaptation) Implementation
 * Efficient fine-tuning using low-rank matrices
 */

import type { WebGPUBackend } from '../gpu-simulator/webgpu-backend';

export interface LoRAConfig {
  rank: number; // Rank of adaptation matrices (typically 4-16)
  alpha: number; // Scaling factor (typically 16-32)
  dropout: number; // Dropout probability (0.0-0.1)
  targetModules: string[]; // Which modules to apply LoRA to
  enableBias: boolean; // Whether to train bias terms
}

export interface LoRALayer {
  // Low-rank matrices
  A: Float32Array; // [rank, in_features]
  B: Float32Array; // [out_features, rank]
  
  // Gradients
  gradA?: Float32Array;
  gradB?: Float32Array;
  
  // Optimizer state (for Adam)
  mA?: Float32Array; // First moment for A
  vA?: Float32Array; // Second moment for A
  mB?: Float32Array; // First moment for B
  vB?: Float32Array; // Second moment for B
  
  // Configuration
  inFeatures: number;
  outFeatures: number;
  rank: number;
  alpha: number;
  scaling: number; // alpha / rank
}

export class LoRAAdapter {
  private config: LoRAConfig;
  private layers: Map<string, LoRALayer> = new Map();
  private webgpu: WebGPUBackend | null;

  constructor(config: LoRAConfig, webgpu: WebGPUBackend | null = null) {
    this.config = config;
    this.webgpu = webgpu;
  }

  /**
   * Initialize LoRA layer for a module
   */
  initializeLayer(
    moduleName: string,
    inFeatures: number,
    outFeatures: number
  ): void {
    const { rank, alpha } = this.config;

    // Initialize with random small values (Gaussian)
    const A = new Float32Array(rank * inFeatures);
    const B = new Float32Array(outFeatures * rank);

    // Initialize A with small random values
    for (let i = 0; i < A.length; i++) {
      A[i] = (Math.random() - 0.5) * 0.01;
    }

    // Initialize B with zeros (as in original LoRA paper)
    B.fill(0);

    const layer: LoRALayer = {
      A,
      B,
      inFeatures,
      outFeatures,
      rank,
      alpha,
      scaling: alpha / rank,
    };

    this.layers.set(moduleName, layer);
  }

  /**
   * Apply LoRA to input
   * Output = W @ x + (B @ A) @ x * scaling
   */
  forward(
    moduleName: string,
    x: Float32Array,
    baseOutput: Float32Array
  ): Float32Array {
    const layer = this.layers.get(moduleName);
    if (!layer) {
      return baseOutput;
    }

    const { A, B, rank, scaling, inFeatures, outFeatures } = layer;
    
    // Compute A @ x
    const Ax = new Float32Array(rank);
    for (let i = 0; i < rank; i++) {
      let sum = 0;
      for (let j = 0; j < inFeatures; j++) {
        sum += A[i * inFeatures + j] * x[j];
      }
      Ax[i] = sum;
    }

    // Compute B @ Ax
    const BAx = new Float32Array(outFeatures);
    for (let i = 0; i < outFeatures; i++) {
      let sum = 0;
      for (let j = 0; j < rank; j++) {
        sum += B[i * rank + j] * Ax[j];
      }
      BAx[i] = sum * scaling;
    }

    // Add to base output
    const output = new Float32Array(outFeatures);
    for (let i = 0; i < outFeatures; i++) {
      output[i] = baseOutput[i] + BAx[i];
    }

    return output;
  }

  /**
   * Compute gradients for LoRA parameters
   * gradA = gradOutput^T @ B^T @ x
   * gradB = gradOutput @ x^T @ A^T
   */
  backward(
    moduleName: string,
    x: Float32Array,
    gradOutput: Float32Array
  ): void {
    const layer = this.layers.get(moduleName);
    if (!layer) {
      return;
    }

    const { A, B, rank, scaling, inFeatures, outFeatures } = layer;

    // Initialize gradients if not exists
    if (!layer.gradA) {
      layer.gradA = new Float32Array(rank * inFeatures);
      layer.gradB = new Float32Array(outFeatures * rank);
    }

    // Compute gradB = gradOutput @ A @ x^T * scaling
    // First compute A @ x
    const Ax = new Float32Array(rank);
    for (let i = 0; i < rank; i++) {
      let sum = 0;
      for (let j = 0; j < inFeatures; j++) {
        sum += A[i * inFeatures + j] * x[j];
      }
      Ax[i] = sum;
    }

    // gradB = gradOutput @ Ax^T * scaling
    for (let i = 0; i < outFeatures; i++) {
      for (let j = 0; j < rank; j++) {
        layer.gradB[i * rank + j] += gradOutput[i] * Ax[j] * scaling;
      }
    }

    // Compute gradA = B^T @ gradOutput @ x^T * scaling
    // First compute B^T @ gradOutput
    const BTgradOut = new Float32Array(rank);
    for (let i = 0; i < rank; i++) {
      let sum = 0;
      for (let j = 0; j < outFeatures; j++) {
        sum += B[j * rank + i] * gradOutput[j];
      }
      BTgradOut[i] = sum * scaling;
    }

    // gradA = BTgradOut @ x^T
    for (let i = 0; i < rank; i++) {
      for (let j = 0; j < inFeatures; j++) {
        layer.gradA[i * inFeatures + j] += BTgradOut[i] * x[j];
      }
    }
  }

  /**
   * Update LoRA parameters using Adam optimizer
   */
  updateParameters(
    learningRate: number,
    beta1: number = 0.9,
    beta2: number = 0.999,
    epsilon: number = 1e-8,
    step: number = 1
  ): void {
    for (const [name, layer] of this.layers) {
      if (!layer.gradA || !layer.gradB) {
        continue;
      }

      // Initialize optimizer state if needed
      if (!layer.mA) {
        layer.mA = new Float32Array(layer.A.length);
        layer.vA = new Float32Array(layer.A.length);
        layer.mB = new Float32Array(layer.B.length);
        layer.vB = new Float32Array(layer.B.length);
      }

      // Bias correction
      const biasCorrectionFactor1 = 1 - Math.pow(beta1, step);
      const biasCorrectionFactor2 = 1 - Math.pow(beta2, step);

      // Update A
      this.adamUpdate(
        layer.A,
        layer.gradA,
        layer.mA,
        layer.vA,
        learningRate,
        beta1,
        beta2,
        epsilon,
        biasCorrectionFactor1,
        biasCorrectionFactor2
      );

      // Update B
      this.adamUpdate(
        layer.B,
        layer.gradB,
        layer.mB,
        layer.vB,
        learningRate,
        beta1,
        beta2,
        epsilon,
        biasCorrectionFactor1,
        biasCorrectionFactor2
      );

      // Zero gradients
      layer.gradA.fill(0);
      layer.gradB.fill(0);
    }
  }

  /**
   * Adam optimizer update step
   */
  private adamUpdate(
    param: Float32Array,
    grad: Float32Array,
    m: Float32Array,
    v: Float32Array,
    lr: number,
    beta1: number,
    beta2: number,
    epsilon: number,
    biasCorrectionFactor1: number,
    biasCorrectionFactor2: number
  ): void {
    for (let i = 0; i < param.length; i++) {
      // Update biased first moment
      m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
      
      // Update biased second moment
      v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
      
      // Bias-corrected moments
      const mHat = m[i] / biasCorrectionFactor1;
      const vHat = v[i] / biasCorrectionFactor2;
      
      // Update parameter
      param[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
    }
  }

  /**
   * Merge LoRA weights into base weights
   * W_merged = W + B @ A * scaling
   */
  mergeIntoBase(
    moduleName: string,
    baseWeight: Float32Array,
    rows: number,
    cols: number
  ): Float32Array {
    const layer = this.layers.get(moduleName);
    if (!layer) {
      return baseWeight;
    }

    const { A, B, rank, scaling } = layer;
    const merged = new Float32Array(baseWeight);

    // Compute B @ A
    const BA = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        let sum = 0;
        for (let k = 0; k < rank; k++) {
          sum += B[i * rank + k] * A[k * cols + j];
        }
        BA[i * cols + j] = sum * scaling;
      }
    }

    // Add to base weight
    for (let i = 0; i < merged.length; i++) {
      merged[i] += BA[i];
    }

    return merged;
  }

  /**
   * Get all LoRA layers
   */
  getLayers(): Map<string, LoRALayer> {
    return this.layers;
  }

  /**
   * Get specific layer
   */
  getLayer(moduleName: string): LoRALayer | undefined {
    return this.layers.get(moduleName);
  }

  /**
   * Zero all gradients
   */
  zeroGrad(): void {
    for (const layer of this.layers.values()) {
      if (layer.gradA) {
        layer.gradA.fill(0);
      }
      if (layer.gradB) {
        layer.gradB.fill(0);
      }
    }
  }

  /**
   * Get total parameter count for LoRA
   */
  getParameterCount(): number {
    let count = 0;
    for (const layer of this.layers.values()) {
      count += layer.A.length + layer.B.length;
    }
    return count;
  }

  /**
   * Save LoRA weights
   */
  saveWeights(): Record<string, { A: Float32Array; B: Float32Array }> {
    const weights: Record<string, { A: Float32Array; B: Float32Array }> = {};
    
    for (const [name, layer] of this.layers) {
      weights[name] = {
        A: new Float32Array(layer.A),
        B: new Float32Array(layer.B),
      };
    }

    return weights;
  }

  /**
   * Load LoRA weights
   */
  loadWeights(weights: Record<string, { A: Float32Array; B: Float32Array }>): void {
    for (const [name, weight] of Object.entries(weights)) {
      const layer = this.layers.get(name);
      if (layer) {
        layer.A.set(weight.A);
        layer.B.set(weight.B);
      }
    }
  }
}
