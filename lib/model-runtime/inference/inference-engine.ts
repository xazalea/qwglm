/**
 * Inference Engine
 * Main inference engine for Qwen3-VL model
 */

import { transformerLayer, type TransformerLayerWeights, type TransformerLayerConfig } from '../transformer/transformer-layer';
import { visionEncoder } from '../vision/vision-encoder';
import { KVCacheManager } from './kv-cache';
import { Tokenizer, QwenTokenizer } from './tokenizer';
import { WebGPUBackend } from '../../gpu-simulator/webgpu-backend';
import type { ModelWeights } from '../quantization/loader';

export interface InferenceConfig {
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  headDim: number;
  intermediateSize: number;
  vocabSize: number;
  maxSeqLen: number;
  useWebGPU?: boolean;
}

export interface GenerationConfig {
  maxNewTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  doSample: boolean;
}

/**
 * Sample next token from logits
 */
function sampleToken(
  logits: Float32Array,
  config: GenerationConfig
): number {
  if (!config.doSample) {
    // Greedy decoding
    let maxIdx = 0;
    let maxVal = logits[0];
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  // Apply temperature
  const temperature = Math.max(config.temperature, 0.01);
  const scaledLogits = logits.map((val) => val / temperature);

  // Top-K sampling
  if (config.topK > 0 && config.topK < logits.length) {
    const sorted = Array.from(scaledLogits)
      .map((val, idx) => ({ val, idx }))
      .sort((a, b) => b.val - a.val)
      .slice(0, config.topK);

    // Top-P (nucleus) sampling
    if (config.topP > 0) {
      const probs = sorted.map((item) => Math.exp(item.val));
      const sum = probs.reduce((s, p) => s + p, 0);
      const normalized = probs.map((p) => p / sum);

      let cumsum = 0;
      let cutoff = sorted.length;
      for (let i = 0; i < normalized.length; i++) {
        cumsum += normalized[i];
        if (cumsum >= config.topP) {
          cutoff = i + 1;
          break;
        }
      }
      sorted.splice(cutoff);
    }

    // Sample from filtered tokens
    const filteredProbs = sorted.map((item) => Math.exp(item.val));
    const filteredSum = filteredProbs.reduce((s, p) => s + p, 0);
    const normalized = filteredProbs.map((p) => p / filteredSum);

    let rand = Math.random();
    for (let i = 0; i < normalized.length; i++) {
      rand -= normalized[i];
      if (rand <= 0) {
        return sorted[i].idx;
      }
    }
    return sorted[0].idx;
  }

  // Fallback to greedy
  let maxIdx = 0;
  let maxVal = scaledLogits[0];
  for (let i = 1; i < scaledLogits.length; i++) {
    if (scaledLogits[i] > maxVal) {
      maxVal = scaledLogits[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

/**
 * Inference Engine
 */
export class InferenceEngine {
  private config: InferenceConfig;
  private weights: ModelWeights;
  private tokenizer: Tokenizer;
  private kvCache: KVCacheManager;
  private webgpu: WebGPUBackend | null = null;

  constructor(config: InferenceConfig, weights: ModelWeights, tokenizer?: Tokenizer) {
    this.config = config;
    this.weights = weights;
    this.tokenizer = tokenizer || new QwenTokenizer();
    this.kvCache = new KVCacheManager();

    // Initialize KV caches for all layers
    for (let i = 0; i < config.numLayers; i++) {
      this.kvCache.initializeLayer(i, config.numHeads, config.headDim);
    }

    // Initialize WebGPU if requested
    if (config.useWebGPU) {
      this.webgpu = new WebGPUBackend();
      this.webgpu.initialize().catch((err) => {
        console.warn('WebGPU initialization failed:', err);
      });
    }
  }

  /**
   * Generate tokens from input
   */
  async generate(
    inputIds: number[],
    imageData?: ImageData,
    generationConfig: GenerationConfig = {
      maxNewTokens: 100,
      temperature: 0.7,
      topK: 50,
      topP: 0.9,
      doSample: true,
    }
  ): Promise<number[]> {
    const generated: number[] = [];
    let currentIds = [...inputIds];

    // Process vision input if provided
    let visionEmbeddings: Float32Array[] = [];
    if (imageData) {
      visionEmbeddings = visionEncoder(imageData, {
        imageSize: 224,
        patchSize: 14,
        hiddenSize: this.config.hiddenSize,
        numLayers: 0, // Not used in simplified version
        numHeads: this.config.numHeads,
      }, {} as any);
    }

    // Encode input tokens to embeddings
    let hiddenStates = currentIds.map((id) => {
      const embedding = new Float32Array(this.config.hiddenSize);
      // In real implementation, would use embedding weights
      // For now, use simple encoding
      embedding.fill(id / this.config.vocabSize);
      return embedding;
    });

    // Prepend vision embeddings
    if (visionEmbeddings.length > 0) {
      hiddenStates = [...visionEmbeddings, ...hiddenStates] as any;
    }

    // Generation loop
    for (let step = 0; step < generationConfig.maxNewTokens; step++) {
      // Forward pass through transformer layers
      for (let layer = 0; layer < this.config.numLayers; layer++) {
        const layerWeights = this.getLayerWeights(layer);
        const layerConfig: TransformerLayerConfig = {
          hiddenSize: this.config.hiddenSize,
          numHeads: this.config.numHeads,
          headDim: this.config.headDim,
          intermediateSize: this.config.intermediateSize,
          attention: {
            hiddenSize: this.config.hiddenSize,
            numHeads: this.config.numHeads,
            headDim: this.config.headDim,
          },
        };

        const kvCache = this.kvCache.getCache(layer);
        const { output, newKvCache } = transformerLayer(
          hiddenStates,
          layerWeights,
          layerConfig,
          kvCache as any
        );

        if (newKvCache) {
          this.kvCache.updateCache(layer, newKvCache.key, newKvCache.value);
        }

        hiddenStates = output as any;
      }

      // Get logits from last hidden state
      const lastHidden = hiddenStates[hiddenStates.length - 1];
      const logits = this.computeLogits(lastHidden);

      // Sample next token
      const nextToken = sampleToken(logits, generationConfig);
      generated.push(nextToken);
      currentIds.push(nextToken);

      // Update hidden states with new token embedding
      const newEmbedding = new Float32Array(this.config.hiddenSize);
      newEmbedding.fill(nextToken / this.config.vocabSize);
      hiddenStates = [newEmbedding]; // Only keep last token for next iteration

      // Check for end token
      if (nextToken === 151643) { // EOS token (example)
        break;
      }
    }

    return generated;
  }

  /**
   * Get layer weights
   */
  private getLayerWeights(layer: number): TransformerLayerWeights {
    // In real implementation, would load from weights map
    // This is a simplified version
    const hiddenSize = this.config.hiddenSize;
    const numHeads = this.config.numHeads;
    const headDim = this.config.headDim;
    const intermediateSize = this.config.intermediateSize;

    return {
      attention: {
        qWeight: this.weights.get(`layers.${layer}.attention.q.weight`) || new Float32Array(hiddenSize * hiddenSize),
        kWeight: this.weights.get(`layers.${layer}.attention.k.weight`) || new Float32Array(hiddenSize * hiddenSize),
        vWeight: this.weights.get(`layers.${layer}.attention.v.weight`) || new Float32Array(hiddenSize * hiddenSize),
        oWeight: this.weights.get(`layers.${layer}.attention.o.weight`) || new Float32Array(hiddenSize * hiddenSize),
      },
      ffn: {
        gateWeight: this.weights.get(`layers.${layer}.ffn.gate.weight`) || new Float32Array(intermediateSize * hiddenSize),
        upWeight: this.weights.get(`layers.${layer}.ffn.up.weight`) || new Float32Array(intermediateSize * hiddenSize),
        downWeight: this.weights.get(`layers.${layer}.ffn.down.weight`) || new Float32Array(hiddenSize * intermediateSize),
      },
      ln1Gamma: this.weights.get(`layers.${layer}.ln1.weight`) || new Float32Array(hiddenSize),
      ln1Beta: this.weights.get(`layers.${layer}.ln1.bias`) || new Float32Array(hiddenSize),
      ln2Gamma: this.weights.get(`layers.${layer}.ln2.weight`) || new Float32Array(hiddenSize),
      ln2Beta: this.weights.get(`layers.${layer}.ln2.bias`) || new Float32Array(hiddenSize),
    };
  }

  /**
   * Compute logits from hidden state
   */
  private computeLogits(hidden: Float32Array): Float32Array {
    const vocabSize = this.config.vocabSize;
    const logits = new Float32Array(vocabSize);

    // In real implementation, would use output projection weights
    // For now, use simple projection
    const outputWeight = this.weights.get('lm_head.weight') || new Float32Array(vocabSize * this.config.hiddenSize);

    for (let i = 0; i < vocabSize; i++) {
      let sum = 0;
      for (let j = 0; j < this.config.hiddenSize; j++) {
        sum += hidden[j] * outputWeight[i * this.config.hiddenSize + j];
      }
      logits[i] = sum;
    }

    return logits;
  }

  /**
   * Clear KV cache
   */
  clearCache(): void {
    this.kvCache.clearAll();
  }
}
