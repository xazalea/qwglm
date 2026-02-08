/**
 * WebGPU-optimized transformer operations
 * Uses real GPU acceleration instead of simulation
 */

import { WebGPUBackend } from '../../gpu-simulator/webgpu-backend';

// WebGPU types are available globally in browsers
// Using 'any' type to avoid TypeScript errors during build
type GPUDeviceType = any;

/**
 * WebGPU Matrix Multiplication with optimizations
 */
export async function matmulWebGPU(
  device: GPUDeviceType,
  A: Float32Array,
  B: Float32Array,
  M: number,
  N: number,
  K: number
): Promise<Float32Array> {
  // Create optimized compute shader for matrix multiplication
  const shaderModule = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
      @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
      @group(0) @binding(2) var<storage, read_write> matrixC: array<f32>;
      @group(0) @binding(3) var<uniform> params: Params;

      struct Params {
        M: u32,
        N: u32,
        K: u32,
      };

      const TILE_SIZE: u32 = 16u;

      var<workgroup> tileA: array<f32, 256>; // 16x16
      var<workgroup> tileB: array<f32, 256>; // 16x16

      @compute @workgroup_size(16, 16)
      fn main(
        @builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>
      ) {
        let row = global_id.y;
        let col = global_id.x;
        
        if (row >= params.M || col >= params.N) {
          return;
        }

        var sum: f32 = 0.0;

        // Tiled matrix multiplication
        let numTiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;
        
        for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
          // Load tile from A
          let tileRow = local_id.y;
          let tileCol = local_id.x;
          let aCol = t * TILE_SIZE + tileCol;
          
          if (row < params.M && aCol < params.K) {
            tileA[tileRow * TILE_SIZE + tileCol] = matrixA[row * params.K + aCol];
          } else {
            tileA[tileRow * TILE_SIZE + tileCol] = 0.0;
          }

          // Load tile from B
          let bRow = t * TILE_SIZE + tileRow;
          if (bRow < params.K && col < params.N) {
            tileB[tileRow * TILE_SIZE + tileCol] = matrixB[bRow * params.N + col];
          } else {
            tileB[tileRow * TILE_SIZE + tileCol] = 0.0;
          }

          workgroupBarrier();

          // Compute partial sum
          for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tileA[tileRow * TILE_SIZE + k] * tileB[k * TILE_SIZE + tileCol];
          }

          workgroupBarrier();
        }

        matrixC[row * params.N + col] = sum;
      }
    `,
  });

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  });

  // Create buffers
  const bufferA = device.createBuffer({
    size: A.byteLength,
      usage: 0x0008 | 0x0002, // STORAGE | COPY_DST
  });

  const bufferB = device.createBuffer({
    size: B.byteLength,
      usage: 0x0008 | 0x0002, // STORAGE | COPY_DST
  });

  const bufferC = device.createBuffer({
    size: M * N * 4,
      usage: 0x0008 | 0x0004, // STORAGE | COPY_SRC
  });

  const paramsBuffer = device.createBuffer({
    size: 12,
      usage: 0x0040 | 0x0002, // UNIFORM | COPY_DST
  });

  // Upload data
  device.queue.writeBuffer(bufferA, 0, A);
  device.queue.writeBuffer(bufferB, 0, B);
  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([M, N, K]));

  // Create bind group
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufferA } },
      { binding: 1, resource: { buffer: bufferB } },
      { binding: 2, resource: { buffer: bufferC } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ],
  });

  // Execute
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(
    Math.ceil(N / 16),
    Math.ceil(M / 16)
  );
  pass.end();

  const readBuffer = device.createBuffer({
    size: bufferC.size,
      usage: 0x0002 | 0x0001, // COPY_DST | MAP_READ
  });

  encoder.copyBufferToBuffer(bufferC, 0, readBuffer, 0, bufferC.size);
  device.queue.submit([encoder.finish()]);

  // Read result
    await readBuffer.mapAsync(1); // GPUMapMode.READ = 1
  const result = new Float32Array(readBuffer.getMappedRange());
  const output = new Float32Array(result);
  readBuffer.unmap();

  // Cleanup
  bufferA.destroy();
  bufferB.destroy();
  bufferC.destroy();
  paramsBuffer.destroy();
  readBuffer.destroy();

  return output;
}

/**
 * Optimized attention computation on WebGPU
 */
export async function attentionWebGPU(
  device: GPUDeviceType,
  Q: Float32Array,
  K: Float32Array,
  V: Float32Array,
  seqLen: number,
  headDim: number
): Promise<Float32Array> {
  // Compute Q @ K^T
  const scores = await matmulWebGPU(
    device,
    Q,
    transposeMatrix(K, seqLen, headDim),
    seqLen,
    seqLen,
    headDim
  );

  // Scale by sqrt(head_dim)
  const scale = 1.0 / Math.sqrt(headDim);
  for (let i = 0; i < scores.length; i++) {
    scores[i] *= scale;
  }

  // Apply softmax
  const attention = applySoftmaxRows(scores, seqLen, seqLen);

  // Compute attention @ V
  const output = await matmulWebGPU(
    device,
    attention,
    V,
    seqLen,
    headDim,
    seqLen
  );

  return output;
}

/**
 * Transpose matrix
 */
function transposeMatrix(matrix: Float32Array, rows: number, cols: number): Float32Array {
  const result = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j * rows + i] = matrix[i * cols + j];
    }
  }
  return result;
}

/**
 * Apply softmax row-wise
 */
function applySoftmaxRows(matrix: Float32Array, rows: number, cols: number): Float32Array {
  const result = new Float32Array(rows * cols);
  
  for (let i = 0; i < rows; i++) {
    const offset = i * cols;
    const row = matrix.slice(offset, offset + cols);
    
    // Find max for numerical stability
    const max = Math.max(...Array.from(row));
    
    // Compute exp and sum
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      const exp = Math.exp(row[j] - max);
      result[offset + j] = exp;
      sum += exp;
    }
    
    // Normalize
    for (let j = 0; j < cols; j++) {
      result[offset + j] /= sum;
    }
  }
  
  return result;
}

/**
 * Layer normalization on WebGPU
 */
export async function layerNormWebGPU(
  device: GPUDeviceType,
  input: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  seqLen: number,
  hiddenSize: number,
  eps: number = 1e-5
): Promise<Float32Array> {
  const shaderModule = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read> gamma: array<f32>;
      @group(0) @binding(2) var<storage, read> beta: array<f32>;
      @group(0) @binding(3) var<storage, read_write> output: array<f32>;
      @group(0) @binding(4) var<uniform> params: Params;

      struct Params {
        seq_len: u32,
        hidden_size: u32,
        eps: f32,
      };

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx >= params.seq_len) {
          return;
        }

        let offset = idx * params.hidden_size;
        
        // Compute mean
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
          sum = sum + input[offset + i];
        }
        let mean = sum / f32(params.hidden_size);

        // Compute variance
        var var_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
          let diff = input[offset + i] - mean;
          var_sum = var_sum + diff * diff;
        }
        let variance = var_sum / f32(params.hidden_size);
        let std = sqrt(variance + params.eps);

        // Normalize and apply affine transform
        for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
          let normalized = (input[offset + i] - mean) / std;
          output[offset + i] = normalized * gamma[i] + beta[i];
        }
      }
    `,
  });

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  });

  const inputBuffer = device.createBuffer({
    size: input.byteLength,
      usage: 0x0008 | 0x0002, // STORAGE | COPY_DST
  });

  const gammaBuffer = device.createBuffer({
    size: gamma.byteLength,
      usage: 0x0008 | 0x0002, // STORAGE | COPY_DST
  });

  const betaBuffer = device.createBuffer({
    size: beta.byteLength,
      usage: 0x0008 | 0x0002, // STORAGE | COPY_DST
  });

  const outputBuffer = device.createBuffer({
    size: input.byteLength,
      usage: 0x0008 | 0x0004, // STORAGE | COPY_SRC
  });

  const paramsBuffer = device.createBuffer({
    size: 12,
      usage: 0x0040 | 0x0002, // UNIFORM | COPY_DST
  });

  device.queue.writeBuffer(inputBuffer, 0, input);
  device.queue.writeBuffer(gammaBuffer, 0, gamma);
  device.queue.writeBuffer(betaBuffer, 0, beta);
  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([seqLen, hiddenSize]));
  device.queue.writeBuffer(paramsBuffer, 8, new Float32Array([eps]));

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: gammaBuffer } },
      { binding: 2, resource: { buffer: betaBuffer } },
      { binding: 3, resource: { buffer: outputBuffer } },
      { binding: 4, resource: { buffer: paramsBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(seqLen / 256));
  pass.end();

  const readBuffer = device.createBuffer({
    size: outputBuffer.size,
      usage: 0x0002 | 0x0001, // COPY_DST | MAP_READ
  });

  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputBuffer.size);
  device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(1); // GPUMapMode.READ = 1
  const result = new Float32Array(readBuffer.getMappedRange());
  const output = new Float32Array(result);
  readBuffer.unmap();

  // Cleanup
  inputBuffer.destroy();
  gammaBuffer.destroy();
  betaBuffer.destroy();
  outputBuffer.destroy();
  paramsBuffer.destroy();
  readBuffer.destroy();

  return output;
}
