/**
 * Multi-Head Attention
 * Implements scaled dot-product attention for transformer
 */

export interface AttentionConfig {
  hiddenSize: number;
  numHeads: number;
  headDim: number;
  dropout?: number;
}

export interface AttentionWeights {
  qWeight: Float32Array;
  kWeight: Float32Array;
  vWeight: Float32Array;
  oWeight: Float32Array;
  qBias?: Float32Array;
  kBias?: Float32Array;
  vBias?: Float32Array;
  oBias?: Float32Array;
}

/**
 * Compute attention scores: Q @ K^T / sqrt(d_k)
 */
function computeAttentionScores(
  Q: Float32Array[],
  K: Float32Array[],
  headDim: number
): Float32Array[] {
  const seqLen = Q.length;
  const scores: Float32Array[] = [];

  for (let i = 0; i < seqLen; i++) {
    const row = new Float32Array(seqLen);
    for (let j = 0; j < seqLen; j++) {
      let sum = 0;
      for (let k = 0; k < headDim; k++) {
        sum += Q[i][k] * K[j][k];
      }
      row[j] = sum / Math.sqrt(headDim);
    }
    scores.push(row);
  }

  return scores;
}

/**
 * Apply softmax to attention scores
 */
function applySoftmax(scores: Float32Array[]): Float32Array[] {
  return scores.map((row) => {
    const max = Math.max(...Array.from(row));
    const exp = row.map((val) => Math.exp(val - max));
    const sum = exp.reduce((s, val) => s + val, 0);
    return new Float32Array(exp.map((val) => val / sum));
  });
}

/**
 * Apply attention weights to values: attention @ V
 */
function applyAttention(
  attention: Float32Array[],
  V: Float32Array[]
): Float32Array[] {
  const seqLen = attention.length;
  const headDim = V[0].length;
  const output: Float32Array[] = [];

  for (let i = 0; i < seqLen; i++) {
    const row = new Float32Array(headDim);
    for (let j = 0; j < seqLen; j++) {
      const attn = attention[i][j];
      for (let k = 0; k < headDim; k++) {
        row[k] += attn * V[j][k];
      }
    }
    output.push(row);
  }

  return output;
}

/**
 * Split into multiple heads
 */
function splitHeads(
  x: Float32Array[],
  numHeads: number,
  headDim: number
): Float32Array[][][] {
  const seqLen = x.length;
  const heads: Float32Array[][][] = [];

  for (let h = 0; h < numHeads; h++) {
    const head: Float32Array[] = [];
    for (let i = 0; i < seqLen; i++) {
      const headVec = new Float32Array(headDim);
      for (let j = 0; j < headDim; j++) {
        headVec[j] = x[i][h * headDim + j];
      }
      head.push(headVec);
    }
      heads.push(head as any);
  }

  return heads;
}

/**
 * Concatenate heads
 */
function concatHeads(heads: Float32Array[][][]): Float32Array[] {
  const numHeads = heads.length;
  const seqLen = heads[0].length;
  const headDim = heads[0][0].length;
  const hiddenSize = numHeads * headDim;

  const output: Float32Array[] = [];

  for (let i = 0; i < seqLen; i++) {
    const vec = new Float32Array(hiddenSize);
    for (let h = 0; h < numHeads; h++) {
      for (let j = 0; j < headDim; j++) {
        vec[h * headDim + j] = (heads[h][i] as any)[j] as number;
      }
    }
    output.push(vec);
  }

  return output;
}

/**
 * Linear transformation: x @ W + b
 */
function linear(
  x: Float32Array[],
  weight: Float32Array,
  inDim: number,
  outDim: number,
  bias?: Float32Array
): Float32Array[] {
  const seqLen = x.length;
  const output: Float32Array[] = [];

  for (let i = 0; i < seqLen; i++) {
    const out = new Float32Array(outDim);
    for (let j = 0; j < outDim; j++) {
      let sum = bias ? bias[j] : 0;
      for (let k = 0; k < inDim; k++) {
        sum += x[i][k] * weight[j * inDim + k];
      }
      out[j] = sum;
    }
    output.push(out);
  }

  return output;
}

/**
 * Multi-head attention forward pass
 */
export function multiHeadAttention(
  hiddenStates: Float32Array[],
  weights: AttentionWeights,
  config: AttentionConfig,
  kvCache?: { key: Float32Array[]; value: Float32Array[] }
): {
  output: Float32Array[];
  newKvCache?: { key: Float32Array[]; value: Float32Array[] };
} {
  const { hiddenSize, numHeads, headDim } = config;
  const seqLen = hiddenStates.length;

  // Project to Q, K, V
  const Q = linear(
    hiddenStates,
    weights.qWeight,
    hiddenSize,
    hiddenSize,
    weights.qBias
  );
  const K = linear(
    hiddenStates,
    weights.kWeight,
    hiddenSize,
    hiddenSize,
    weights.kBias
  );
  const V = linear(
    hiddenStates,
    weights.vWeight,
    hiddenSize,
    hiddenSize,
    weights.vBias
  );

  // Update KV cache if provided
  let finalK = K;
  let finalV = V;
  if (kvCache) {
    finalK = [...kvCache.key, ...K];
    finalV = [...kvCache.value, ...V];
  }

  // Split into heads
  const QHeads = splitHeads(Q, numHeads, headDim);
  const KHeads = splitHeads(finalK, numHeads, headDim);
  const VHeads = splitHeads(finalV, numHeads, headDim);

  // Compute attention for each head
  const headOutputs: Float32Array[][][] = [];

  for (let h = 0; h < numHeads; h++) {
    const scores = computeAttentionScores(QHeads[h] as any, KHeads[h] as any, headDim);
    const attention = applySoftmax(scores);
    const headOut = applyAttention(attention, VHeads[h] as any);
    headOutputs.push(headOut as any);
  }

  // Concatenate heads
  const concat = concatHeads(headOutputs);

  // Output projection
  const output = linear(
    concat,
    weights.oWeight,
    hiddenSize,
    hiddenSize,
    weights.oBias
  );

  return {
    output,
    newKvCache: {
      key: finalK,
      value: finalV,
    },
  };
}
