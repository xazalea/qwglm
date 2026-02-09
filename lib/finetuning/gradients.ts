/**
 * Gradient Computation
 * Backpropagation through transformer layers
 */

import type { WebGPUBackend } from '../gpu-simulator/webgpu-backend';

export interface LayerGradients {
  // Attention gradients
  dWq: Float32Array;
  dWk: Float32Array;
  dWv: Float32Array;
  dWo: Float32Array;
  
  // FFN gradients
  dWgate: Float32Array;
  dWup: Float32Array;
  dWdown: Float32Array;
  
  // Layer norm gradients
  dLn1Gamma: Float32Array;
  dLn1Beta: Float32Array;
  dLn2Gamma: Float32Array;
  dLn2Beta: Float32Array;
}

export interface ActivationCache {
  // Input to layer
  input: Float32Array[];
  
  // After layer norm 1
  norm1: Float32Array[];
  
  // Attention components
  Q: Float32Array[];
  K: Float32Array[];
  V: Float32Array[];
  attnScores: Float32Array[];
  attnWeights: Float32Array[];
  attnOutput: Float32Array[];
  
  // After residual 1
  residual1: Float32Array[];
  
  // After layer norm 2
  norm2: Float32Array[];
  
  // FFN components
  gate: Float32Array[];
  up: Float32Array[];
  activated: Float32Array[];
  ffnOutput: Float32Array[];
  
  // Final output
  output: Float32Array[];
}

export class GradientComputer {
  private webgpu: WebGPUBackend | null;

  constructor(webgpu: WebGPUBackend | null = null) {
    this.webgpu = webgpu;
  }

  /**
   * Compute gradients for attention layer
   */
  computeAttentionGradients(
    gradOutput: Float32Array[],
    cache: ActivationCache,
    weights: {
      Wq: Float32Array;
      Wk: Float32Array;
      Wv: Float32Array;
      Wo: Float32Array;
    },
    hiddenSize: number,
    numHeads: number,
    headDim: number
  ): {
    dWq: Float32Array;
    dWk: Float32Array;
    dWv: Float32Array;
    dWo: Float32Array;
    dInput: Float32Array[];
  } {
    const seqLen = gradOutput.length;

    // Initialize gradients
    const dWq = new Float32Array(hiddenSize * hiddenSize);
    const dWk = new Float32Array(hiddenSize * hiddenSize);
    const dWv = new Float32Array(hiddenSize * hiddenSize);
    const dWo = new Float32Array(hiddenSize * hiddenSize);
    const dInput: Float32Array[] = [];

    // Backprop through output projection
    const dAttnOutput: Float32Array[] = [];
    for (let i = 0; i < seqLen; i++) {
      const dOut = new Float32Array(hiddenSize);
      
      // dWo += gradOutput[i] @ attnOutput[i]^T
      for (let j = 0; j < hiddenSize; j++) {
        for (let k = 0; k < hiddenSize; k++) {
          dWo[j * hiddenSize + k] += gradOutput[i][j] * cache.attnOutput[i][k];
        }
        
        // dAttnOutput = Wo^T @ gradOutput[i]
        for (let k = 0; k < hiddenSize; k++) {
          dOut[k] += weights.Wo[j * hiddenSize + k] * gradOutput[i][j];
        }
      }
      dAttnOutput.push(dOut);
    }

    // Backprop through attention mechanism
    const dV: Float32Array[] = [];
    const dAttnWeights: Float32Array[] = [];

    for (let i = 0; i < seqLen; i++) {
      // dV = attnWeights^T @ dAttnOutput
      const dv = new Float32Array(hiddenSize);
      for (let j = 0; j < seqLen; j++) {
        for (let k = 0; k < hiddenSize; k++) {
          dv[k] += cache.attnWeights[i][j] * dAttnOutput[i][k];
        }
      }
      dV.push(dv);

      // dAttnWeights = dAttnOutput @ V^T
      const dWeights = new Float32Array(seqLen);
      for (let j = 0; j < seqLen; j++) {
        for (let k = 0; k < hiddenSize; k++) {
          dWeights[j] += dAttnOutput[i][k] * cache.V[j][k];
        }
      }
      dAttnWeights.push(dWeights);
    }

    // Backprop through softmax
    const dAttnScores: Float32Array[] = this.softmaxBackward(
      dAttnWeights,
      cache.attnWeights
    );

    // Backprop through Q @ K^T
    const dQ: Float32Array[] = [];
    const dK: Float32Array[] = [];
    const scale = 1.0 / Math.sqrt(headDim);

    for (let i = 0; i < seqLen; i++) {
      const dq = new Float32Array(hiddenSize);
      const dk = new Float32Array(hiddenSize);

      for (let j = 0; j < seqLen; j++) {
        const dScore = dAttnScores[i][j] * scale;
        
        for (let k = 0; k < hiddenSize; k++) {
          dq[k] += dScore * cache.K[j][k];
          dk[k] += dScore * cache.Q[i][k];
        }
      }

      dQ.push(dq);
      dK.push(dk);
    }

    // Backprop through Q, K, V projections
    for (let i = 0; i < seqLen; i++) {
      const dIn = new Float32Array(hiddenSize);

      // dWq += dQ[i] @ input[i]^T
      for (let j = 0; j < hiddenSize; j++) {
        for (let k = 0; k < hiddenSize; k++) {
          dWq[j * hiddenSize + k] += dQ[i][j] * cache.norm1[i][k];
        }
      }

      // dWk += dK[i] @ input[i]^T
      for (let j = 0; j < hiddenSize; j++) {
        for (let k = 0; k < hiddenSize; k++) {
          dWk[j * hiddenSize + k] += dK[i][j] * cache.norm1[i][k];
        }
      }

      // dWv += dV[i] @ input[i]^T
      for (let j = 0; j < hiddenSize; j++) {
        for (let k = 0; k < hiddenSize; k++) {
          dWv[j * hiddenSize + k] += dV[i][j] * cache.norm1[i][k];
        }
      }

      // dInput = Wq^T @ dQ + Wk^T @ dK + Wv^T @ dV
      for (let j = 0; j < hiddenSize; j++) {
        for (let k = 0; k < hiddenSize; k++) {
          dIn[j] += weights.Wq[k * hiddenSize + j] * dQ[i][k];
          dIn[j] += weights.Wk[k * hiddenSize + j] * dK[i][k];
          dIn[j] += weights.Wv[k * hiddenSize + j] * dV[i][k];
        }
      }

      dInput.push(dIn);
    }

    return { dWq, dWk, dWv, dWo, dInput };
  }

  /**
   * Compute gradients for FFN layer
   */
  computeFFNGradients(
    gradOutput: Float32Array[],
    cache: ActivationCache,
    weights: {
      Wgate: Float32Array;
      Wup: Float32Array;
      Wdown: Float32Array;
    },
    hiddenSize: number,
    intermediateSize: number
  ): {
    dWgate: Float32Array;
    dWup: Float32Array;
    dWdown: Float32Array;
    dInput: Float32Array[];
  } {
    const seqLen = gradOutput.length;

    const dWgate = new Float32Array(intermediateSize * hiddenSize);
    const dWup = new Float32Array(intermediateSize * hiddenSize);
    const dWdown = new Float32Array(hiddenSize * intermediateSize);
    const dInput: Float32Array[] = [];

    // Backprop through down projection
    const dActivated: Float32Array[] = [];
    for (let i = 0; i < seqLen; i++) {
      const dAct = new Float32Array(intermediateSize);

      // dWdown += gradOutput[i] @ activated[i]^T
      for (let j = 0; j < hiddenSize; j++) {
        for (let k = 0; k < intermediateSize; k++) {
          dWdown[j * intermediateSize + k] += gradOutput[i][j] * cache.activated[i][k];
        }
      }

      // dActivated = Wdown^T @ gradOutput[i]
      for (let j = 0; j < intermediateSize; j++) {
        for (let k = 0; k < hiddenSize; k++) {
          dAct[j] += weights.Wdown[k * intermediateSize + j] * gradOutput[i][k];
        }
      }

      dActivated.push(dAct);
    }

    // Backprop through SwiGLU activation
    const dGate: Float32Array[] = [];
    const dUp: Float32Array[] = [];

    for (let i = 0; i < seqLen; i++) {
      const dg = new Float32Array(intermediateSize);
      const du = new Float32Array(intermediateSize);

      for (let j = 0; j < intermediateSize; j++) {
        const gate = cache.gate[i][j];
        const up = cache.up[i][j];
        const sigmoid = 1.0 / (1.0 + Math.exp(-gate));

        // SwiGLU: out = sigmoid(gate) * gate * up
        // dgate = dout * (sigmoid * (1 - sigmoid) * gate + sigmoid) * up
        // dup = dout * sigmoid * gate
        dg[j] = dActivated[i][j] * sigmoid * (1 - sigmoid + gate) * up;
        du[j] = dActivated[i][j] * sigmoid * gate;
      }

      dGate.push(dg);
      dUp.push(du);
    }

    // Backprop through gate and up projections
    for (let i = 0; i < seqLen; i++) {
      const dIn = new Float32Array(hiddenSize);

      // dWgate += dGate[i] @ input[i]^T
      for (let j = 0; j < intermediateSize; j++) {
        for (let k = 0; k < hiddenSize; k++) {
          dWgate[j * hiddenSize + k] += dGate[i][j] * cache.norm2[i][k];
        }
      }

      // dWup += dUp[i] @ input[i]^T
      for (let j = 0; j < intermediateSize; j++) {
        for (let k = 0; k < hiddenSize; k++) {
          dWup[j * hiddenSize + k] += dUp[i][j] * cache.norm2[i][k];
        }
      }

      // dInput = Wgate^T @ dGate + Wup^T @ dUp
      for (let j = 0; j < hiddenSize; j++) {
        for (let k = 0; k < intermediateSize; k++) {
          dIn[j] += weights.Wgate[k * hiddenSize + j] * dGate[i][k];
          dIn[j] += weights.Wup[k * hiddenSize + j] * dUp[i][k];
        }
      }

      dInput.push(dIn);
    }

    return { dWgate, dWup, dWdown, dInput };
  }

  /**
   * Compute gradients for layer normalization
   */
  computeLayerNormGradients(
    gradOutput: Float32Array[],
    input: Float32Array[],
    gamma: Float32Array,
    beta: Float32Array,
    eps: number = 1e-5
  ): {
    dGamma: Float32Array;
    dBeta: Float32Array;
    dInput: Float32Array[];
  } {
    const seqLen = gradOutput.length;
    const hiddenSize = gamma.length;

    const dGamma = new Float32Array(hiddenSize);
    const dBeta = new Float32Array(hiddenSize);
    const dInput: Float32Array[] = [];

    for (let i = 0; i < seqLen; i++) {
      // Compute mean and variance
      let mean = 0;
      for (let j = 0; j < hiddenSize; j++) {
        mean += input[i][j];
      }
      mean /= hiddenSize;

      let variance = 0;
      for (let j = 0; j < hiddenSize; j++) {
        const diff = input[i][j] - mean;
        variance += diff * diff;
      }
      variance /= hiddenSize;

      const std = Math.sqrt(variance + eps);

      // Normalized values
      const normalized = new Float32Array(hiddenSize);
      for (let j = 0; j < hiddenSize; j++) {
        normalized[j] = (input[i][j] - mean) / std;
      }

      // Accumulate gamma and beta gradients
      for (let j = 0; j < hiddenSize; j++) {
        dGamma[j] += gradOutput[i][j] * normalized[j];
        dBeta[j] += gradOutput[i][j];
      }

      // Compute input gradient
      const dIn = new Float32Array(hiddenSize);
      let dNormSum = 0;
      let dNormDotNorm = 0;

      for (let j = 0; j < hiddenSize; j++) {
        const dNorm = gradOutput[i][j] * gamma[j];
        dNormSum += dNorm;
        dNormDotNorm += dNorm * normalized[j];
      }

      for (let j = 0; j < hiddenSize; j++) {
        const dNorm = gradOutput[i][j] * gamma[j];
        dIn[j] = (dNorm - dNormSum / hiddenSize - normalized[j] * dNormDotNorm / hiddenSize) / std;
      }

      dInput.push(dIn);
    }

    return { dGamma, dBeta, dInput };
  }

  /**
   * Softmax backward pass
   */
  private softmaxBackward(
    gradOutput: Float32Array[],
    softmaxOutput: Float32Array[]
  ): Float32Array[] {
    const result: Float32Array[] = [];

    for (let i = 0; i < gradOutput.length; i++) {
      const dScores = new Float32Array(gradOutput[i].length);
      const softmax = softmaxOutput[i];
      const grad = gradOutput[i];

      // Compute sum of grad * softmax
      let dotProduct = 0;
      for (let j = 0; j < grad.length; j++) {
        dotProduct += grad[j] * softmax[j];
      }

      // dScores = softmax * (grad - dotProduct)
      for (let j = 0; j < dScores.length; j++) {
        dScores[j] = softmax[j] * (grad[j] - dotProduct);
      }

      result.push(dScores);
    }

    return result;
  }

  /**
   * Clip gradients to prevent explosion
   */
  clipGradients(
    gradients: Float32Array,
    maxNorm: number
  ): Float32Array {
    // Compute gradient norm
    let norm = 0;
    for (let i = 0; i < gradients.length; i++) {
      norm += gradients[i] * gradients[i];
    }
    norm = Math.sqrt(norm);

    // Clip if needed
    if (norm > maxNorm) {
      const scale = maxNorm / norm;
      const clipped = new Float32Array(gradients.length);
      for (let i = 0; i < gradients.length; i++) {
        clipped[i] = gradients[i] * scale;
      }
      return clipped;
    }

    return gradients;
  }

  /**
   * Accumulate gradients
   */
  accumulateGradients(
    target: Float32Array,
    source: Float32Array
  ): void {
    for (let i = 0; i < target.length; i++) {
      target[i] += source[i];
    }
  }

  /**
   * Zero gradients
   */
  zeroGradients(gradients: Float32Array): void {
    gradients.fill(0);
  }
}
