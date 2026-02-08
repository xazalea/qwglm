/**
 * WebGPU Backend
 * Provides GPU acceleration for the simulator using WebGPU compute shaders
 */

// WebGPU types are available globally in browsers
// Using 'any' type to avoid TypeScript errors during build
type GPUDeviceType = any;
type GPUQueueType = any;
type GPUComputePipelineType = any;
type GPUBindGroupType = any;
type GPUAdapterType = any;

// Extend Navigator interface for WebGPU
declare global {
  interface Navigator {
    gpu?: {
      requestAdapter(): Promise<GPUAdapterType | null>;
    };
  }
}

export interface WebGPUDevice {
  device: GPUDeviceType;
  queue: GPUQueueType;
  computePipeline: GPUComputePipelineType;
  bindGroup: GPUBindGroupType;
}

export class WebGPUBackend {
  private device: GPUDeviceType | null = null;
  private adapter: GPUAdapterType | null = null;
  private initialized = false;

  /**
   * Initialize WebGPU device
   */
  async initialize(): Promise<boolean> {
    if (!navigator.gpu) {
      console.warn('WebGPU not supported in this browser');
      return false;
    }

    try {
      this.adapter = await navigator.gpu.requestAdapter();
      if (!this.adapter) {
        console.warn('No WebGPU adapter available');
        return false;
      }

      this.device = await this.adapter.requestDevice();
      this.initialized = true;
      return true;
    } catch (error) {
      console.error('Failed to initialize WebGPU:', error);
      return false;
    }
  }

  /**
   * Check if WebGPU is available
   */
  isAvailable(): boolean {
    return this.initialized && this.device !== null;
  }

  /**
   * Get WebGPU device
   */
  getDevice(): GPUDeviceType | null {
    return this.device;
  }

  /**
   * Create compute shader for matrix operations
   */
  createMatrixMultiplyPipeline(
    device: GPUDeviceType,
    workgroupSize: number = 8
  ): GPUComputePipelineType {
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

        @compute @workgroup_size(${workgroupSize}, ${workgroupSize})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let row = global_id.y;
          let col = global_id.x;

          if (row >= params.M || col >= params.N) {
            return;
          }

          var sum: f32 = 0.0;
          for (var k: u32 = 0u; k < params.K; k = k + 1u) {
            sum = sum + matrixA[row * params.K + k] * matrixB[k * params.N + col];
          }

          matrixC[row * params.N + col] = sum;
        }
      `,
    });

    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
  }

  /**
   * Create compute shader for element-wise operations
   */
  createElementWisePipeline(
    device: GPUDeviceType,
    operation: 'add' | 'multiply' | 'relu'
  ): GPUComputePipelineType {
    const opCode = {
      add: 'result = a + b;',
      multiply: 'result = a * b;',
      relu: 'result = max(a, 0.0);',
    }[operation];

    const shaderModule = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> inputA: array<f32>;
        @group(0) @binding(1) var<storage, read> inputB: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
        @group(0) @binding(3) var<uniform> size: u32;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          if (index >= size) {
            return;
          }

          let a = inputA[index];
          let b = inputB[index];
          var result: f32;
          ${opCode}
          output[index] = result;
        }
      `,
    });

    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
  }

  /**
   * Execute matrix multiplication on GPU
   */
  async matrixMultiply(
    device: GPUDeviceType,
    A: Float32Array,
    B: Float32Array,
    M: number,
    N: number,
    K: number
  ): Promise<Float32Array> {
    const pipeline = this.createMatrixMultiplyPipeline(device);
    const workgroupSize = 8;

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
      size: 12, // 3 u32s
      usage: 0x0040 | 0x0002, // UNIFORM | COPY_DST
    });

    // Upload data
    device.queue.writeBuffer(bufferA, 0, A);
    device.queue.writeBuffer(bufferB, 0, B);

    const params = new Uint32Array([M, N, K]);
    device.queue.writeBuffer(paramsBuffer, 0, params);

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
      Math.ceil(N / workgroupSize),
      Math.ceil(M / workgroupSize)
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

    return output;
  }

  /**
   * Execute element-wise operation on GPU
   */
  async elementWise(
    device: GPUDeviceType,
    A: Float32Array,
    B: Float32Array,
    operation: 'add' | 'multiply' | 'relu',
    size: number
  ): Promise<Float32Array> {
    const pipeline = this.createElementWisePipeline(device, operation);

    // Create buffers
    const bufferA = device.createBuffer({
      size: A.byteLength,
      usage: 0x0008 | 0x0002, // STORAGE | COPY_DST
    });

    const bufferB = device.createBuffer({
      size: B.byteLength,
      usage: 0x0008 | 0x0002, // STORAGE | COPY_DST
    });

    const bufferOut = device.createBuffer({
      size: size * 4,
      usage: 0x0008 | 0x0004, // STORAGE | COPY_SRC
    });

    const sizeBuffer = device.createBuffer({
      size: 4,
      usage: 0x0040 | 0x0002, // UNIFORM | COPY_DST
    });

    // Upload data
    device.queue.writeBuffer(bufferA, 0, A);
    device.queue.writeBuffer(bufferB, 0, B);
    device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([size]));

    // Create bind group
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: bufferOut } },
        { binding: 3, resource: { buffer: sizeBuffer } },
      ],
    });

    // Execute
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(size / 64));
    pass.end();

    const readBuffer = device.createBuffer({
      size: bufferOut.size,
      usage: 0x0002 | 0x0001, // COPY_DST | MAP_READ
    });

    encoder.copyBufferToBuffer(bufferOut, 0, readBuffer, 0, bufferOut.size);
    device.queue.submit([encoder.finish()]);

    // Read result
    await readBuffer.mapAsync(1); // GPUMapMode.READ = 1
    const result = new Float32Array(readBuffer.getMappedRange());
    const output = new Float32Array(result);
    readBuffer.unmap();

    return output;
  }
}
