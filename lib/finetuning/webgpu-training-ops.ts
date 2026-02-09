/**
 * WebGPU Training Operations
 * GPU-accelerated gradient computation and weight updates
 */

type GPUDeviceType = any;

export class WebGPUTrainingOps {
  private device: GPUDeviceType;

  constructor(device: GPUDeviceType) {
    this.device = device;
  }

  /**
   * Matrix multiplication for gradients: C = A @ B^T
   */
  async matmulGrad(
    A: Float32Array,
    B: Float32Array,
    M: number,
    N: number,
    K: number
  ): Promise<Float32Array> {
    const shaderModule = this.device.createShaderModule({
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

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let row = global_id.y;
          let col = global_id.x;
          
          if (row >= params.M || col >= params.N) {
            return;
          }

          var sum: f32 = 0.0;
          for (var k: u32 = 0u; k < params.K; k = k + 1u) {
            sum = sum + matrixA[row * params.K + k] * matrixB[col * params.K + k];
          }

          matrixC[row * params.N + col] = sum;
        }
      `,
    });

    const pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });

    // Create buffers
    const bufferA = this.device.createBuffer({
      size: A.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const bufferB = this.device.createBuffer({
      size: B.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const bufferC = this.device.createBuffer({
      size: M * N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const paramsData = new Uint32Array([M, N, K]);
    const paramsBuffer = this.device.createBuffer({
      size: paramsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Write data
    this.device.queue.writeBuffer(bufferA, 0, A);
    this.device.queue.writeBuffer(bufferB, 0, B);
    this.device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: bufferC } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    // Dispatch compute
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(
      Math.ceil(N / 16),
      Math.ceil(M / 16)
    );
    passEncoder.end();

    // Read result
    const readBuffer = this.device.createBuffer({
      size: M * N * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    commandEncoder.copyBufferToBuffer(bufferC, 0, readBuffer, 0, M * N * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
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
   * Element-wise operations for gradient accumulation
   */
  async elementWiseOp(
    A: Float32Array,
    B: Float32Array,
    size: number,
    operation: 'add' | 'subtract' | 'multiply' | 'divide'
  ): Promise<Float32Array> {
    const opCode = {
      add: 'result = a + b;',
      subtract: 'result = a - b;',
      multiply: 'result = a * b;',
      divide: 'result = a / b;',
    }[operation];

    const shaderModule = this.device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> inputA: array<f32>;
        @group(0) @binding(1) var<storage, read> inputB: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          if (idx >= ${size}u) {
            return;
          }

          let a = inputA[idx];
          let b = inputB[idx];
          var result: f32;
          ${opCode}
          output[idx] = result;
        }
      `,
    });

    const pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });

    // Create buffers
    const bufferA = this.device.createBuffer({
      size: A.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const bufferB = this.device.createBuffer({
      size: B.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const bufferOut = this.device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.device.queue.writeBuffer(bufferA, 0, A);
    this.device.queue.writeBuffer(bufferB, 0, B);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: bufferOut } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(size / 256));
    passEncoder.end();

    const readBuffer = this.device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    commandEncoder.copyBufferToBuffer(bufferOut, 0, readBuffer, 0, size * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange());
    const output = new Float32Array(result);
    readBuffer.unmap();

    bufferA.destroy();
    bufferB.destroy();
    bufferOut.destroy();
    readBuffer.destroy();

    return output;
  }

  /**
   * Adam optimizer step on GPU
   */
  async adamUpdate(
    param: Float32Array,
    grad: Float32Array,
    m: Float32Array,
    v: Float32Array,
    size: number,
    lr: number,
    beta1: number,
    beta2: number,
    epsilon: number,
    step: number
  ): Promise<{
    param: Float32Array;
    m: Float32Array;
    v: Float32Array;
  }> {
    const shaderModule = this.device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read_write> param: array<f32>;
        @group(0) @binding(1) var<storage, read> grad: array<f32>;
        @group(0) @binding(2) var<storage, read_write> m: array<f32>;
        @group(0) @binding(3) var<storage, read_write> v: array<f32>;
        @group(0) @binding(4) var<uniform> config: Config;

        struct Config {
          lr: f32,
          beta1: f32,
          beta2: f32,
          epsilon: f32,
          bias_correction1: f32,
          bias_correction2: f32,
        };

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          if (idx >= ${size}u) {
            return;
          }

          let g = grad[idx];
          
          // Update biased first moment
          m[idx] = config.beta1 * m[idx] + (1.0 - config.beta1) * g;
          
          // Update biased second moment
          v[idx] = config.beta2 * v[idx] + (1.0 - config.beta2) * g * g;
          
          // Bias-corrected moments
          let m_hat = m[idx] / config.bias_correction1;
          let v_hat = v[idx] / config.bias_correction2;
          
          // Update parameter
          param[idx] = param[idx] - config.lr * m_hat / (sqrt(v_hat) + config.epsilon);
        }
      `,
    });

    const pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });

    // Create buffers
    const bufferParam = this.device.createBuffer({
      size: param.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const bufferGrad = this.device.createBuffer({
      size: grad.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const bufferM = this.device.createBuffer({
      size: m.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const bufferV = this.device.createBuffer({
      size: v.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const biasCorrectionFactor1 = 1 - Math.pow(beta1, step);
    const biasCorrectionFactor2 = 1 - Math.pow(beta2, step);

    const configData = new Float32Array([
      lr,
      beta1,
      beta2,
      epsilon,
      biasCorrectionFactor1,
      biasCorrectionFactor2,
    ]);

    const bufferConfig = this.device.createBuffer({
      size: configData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(bufferParam, 0, param);
    this.device.queue.writeBuffer(bufferGrad, 0, grad);
    this.device.queue.writeBuffer(bufferM, 0, m);
    this.device.queue.writeBuffer(bufferV, 0, v);
    this.device.queue.writeBuffer(bufferConfig, 0, configData);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferParam } },
        { binding: 1, resource: { buffer: bufferGrad } },
        { binding: 2, resource: { buffer: bufferM } },
        { binding: 3, resource: { buffer: bufferV } },
        { binding: 4, resource: { buffer: bufferConfig } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(size / 256));
    passEncoder.end();

    // Read results
    const readBufferParam = this.device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const readBufferM = this.device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const readBufferV = this.device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    commandEncoder.copyBufferToBuffer(bufferParam, 0, readBufferParam, 0, size * 4);
    commandEncoder.copyBufferToBuffer(bufferM, 0, readBufferM, 0, size * 4);
    commandEncoder.copyBufferToBuffer(bufferV, 0, readBufferV, 0, size * 4);
    
    this.device.queue.submit([commandEncoder.finish()]);

    await Promise.all([
      readBufferParam.mapAsync(GPUMapMode.READ),
      readBufferM.mapAsync(GPUMapMode.READ),
      readBufferV.mapAsync(GPUMapMode.READ),
    ]);

    const outParam = new Float32Array(readBufferParam.getMappedRange());
    const outM = new Float32Array(readBufferM.getMappedRange());
    const outV = new Float32Array(readBufferV.getMappedRange());

    const result = {
      param: new Float32Array(outParam),
      m: new Float32Array(outM),
      v: new Float32Array(outV),
    };

    readBufferParam.unmap();
    readBufferM.unmap();
    readBufferV.unmap();

    // Cleanup
    bufferParam.destroy();
    bufferGrad.destroy();
    bufferM.destroy();
    bufferV.destroy();
    bufferConfig.destroy();
    readBufferParam.destroy();
    readBufferM.destroy();
    readBufferV.destroy();

    return result;
  }

  /**
   * Gradient clipping on GPU
   */
  async clipGradients(
    gradients: Float32Array,
    maxNorm: number
  ): Promise<Float32Array> {
    const size = gradients.length;

    // First compute norm
    const shaderNorm = this.device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          if (idx >= ${size}u) {
            return;
          }

          let val = input[idx];
          output[0] = output[0] + val * val;
        }
      `,
    });

    // Then scale if needed
    const shaderClip = this.device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @group(0) @binding(2) var<uniform> scale: f32;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          if (idx >= ${size}u) {
            return;
          }

          output[idx] = input[idx] * scale;
        }
      `,
    });

    // Compute norm (simplified - using CPU for now)
    let norm = 0;
    for (let i = 0; i < size; i++) {
      norm += gradients[i] * gradients[i];
    }
    norm = Math.sqrt(norm);

    if (norm <= maxNorm) {
      return gradients;
    }

    const scale = maxNorm / norm;
    return await this.elementWiseOp(
      gradients,
      new Float32Array(size).fill(scale),
      size,
      'multiply'
    );
  }
}
