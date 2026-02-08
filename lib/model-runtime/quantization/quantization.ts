/**
 * Quantization Support
 * Handles loading and dequantizing 4-bit and 8-bit quantized weights
 */

export type QuantizationType = 'int4' | 'int8' | 'fp16' | 'fp32';

export interface QuantizedWeight {
  data: Uint8Array | Uint16Array;
  scale: Float32Array;
  zeroPoint?: Uint8Array;
  shape: number[];
  dtype: QuantizationType;
}

/**
 * Dequantize 4-bit weights
 */
export function dequantizeInt4(
  quantized: Uint8Array,
  scale: Float32Array,
  zeroPoint: Uint8Array,
  shape: number[]
): Float32Array {
  const totalElements = shape.reduce((a, b) => a * b, 1);
  const result = new Float32Array(totalElements);

  for (let i = 0; i < totalElements; i++) {
    const byteIdx = Math.floor(i / 2);
    const isHigh = i % 2 === 1;

    let quantizedValue: number;
    if (isHigh) {
      quantizedValue = (quantized[byteIdx] >>> 4) & 0xf;
    } else {
      quantizedValue = quantized[byteIdx] & 0xf;
    }

    const scaleIdx = Math.floor(i / (totalElements / scale.length));
    const zpIdx = Math.floor(i / (totalElements / zeroPoint.length));

    result[i] = (quantizedValue - zeroPoint[zpIdx]) * scale[scaleIdx];
  }

  return result;
}

/**
 * Dequantize 8-bit weights
 */
export function dequantizeInt8(
  quantized: Uint8Array,
  scale: Float32Array,
  zeroPoint: Uint8Array,
  shape: number[]
): Float32Array {
  const totalElements = shape.reduce((a, b) => a * b, 1);
  const result = new Float32Array(totalElements);

  for (let i = 0; i < totalElements; i++) {
    const scaleIdx = Math.floor(i / (totalElements / scale.length));
    const zpIdx = Math.floor(i / (totalElements / zeroPoint.length));

    result[i] = (quantized[i] - zeroPoint[zpIdx]) * scale[scaleIdx];
  }

  return result;
}

/**
 * Dequantize FP16 weights
 */
export function dequantizeFP16(quantized: Uint16Array, shape: number[]): Float32Array {
  const totalElements = shape.reduce((a, b) => a * b, 1);
  const result = new Float32Array(totalElements);

  for (let i = 0; i < totalElements; i++) {
    result[i] = fp16ToFp32(quantized[i]);
  }

  return result;
}

/**
 * Convert FP16 to FP32
 */
function fp16ToFp32(half: number): number {
  const sign = (half >>> 15) & 0x1;
  const exp = (half >>> 10) & 0x1f;
  const mantissa = half & 0x3ff;

  if (exp === 0) {
    // Denormalized
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024);
  } else if (exp === 31) {
    // Infinity or NaN
    return mantissa === 0
      ? (sign ? -Infinity : Infinity)
      : NaN;
  } else {
    // Normalized
    return (
      (sign ? -1 : 1) *
      Math.pow(2, exp - 15) *
      (1 + mantissa / 1024)
    );
  }
}

/**
 * Dequantize weights based on type
 */
export function dequantize(
  quantized: QuantizedWeight
): Float32Array {
  switch (quantized.dtype) {
    case 'int4':
      if (!quantized.zeroPoint) {
        throw new Error('Zero point required for int4 quantization');
      }
      return dequantizeInt4(
        quantized.data as Uint8Array,
        quantized.scale,
        quantized.zeroPoint,
        quantized.shape
      );
    case 'int8':
      if (!quantized.zeroPoint) {
        throw new Error('Zero point required for int8 quantization');
      }
      return dequantizeInt8(
        quantized.data as Uint8Array,
        quantized.scale,
        quantized.zeroPoint,
        quantized.shape
      );
    case 'fp16':
      return dequantizeFP16(quantized.data as Uint16Array, quantized.shape);
    case 'fp32':
      return new Float32Array(quantized.data as unknown as ArrayBuffer);
    default:
      throw new Error(`Unsupported quantization type: ${quantized.dtype}`);
  }
}

/**
 * Load quantized weight from buffer
 */
export function loadQuantizedWeight(
  buffer: ArrayBuffer,
  offset: number,
  shape: number[],
  dtype: QuantizationType,
  scale?: Float32Array,
  zeroPoint?: Uint8Array
): QuantizedWeight {
  const totalElements = shape.reduce((a, b) => a * b, 1);
  let data: Uint8Array | Uint16Array;

  switch (dtype) {
    case 'int4':
      data = new Uint8Array(buffer, offset, Math.ceil(totalElements / 2));
      break;
    case 'int8':
      data = new Uint8Array(buffer, offset, totalElements);
      break;
    case 'fp16':
      data = new Uint16Array(buffer, offset, totalElements);
      break;
    case 'fp32':
      data = new Uint8Array(buffer, offset, totalElements * 4);
      break;
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }

  return {
    data,
    scale: scale || new Float32Array([1.0]),
    zeroPoint,
    shape,
    dtype,
  };
}
