/**
 * KV Cache
 * Manages key-value cache for efficient generation
 */

export interface KVCache {
  key: Float32Array[][];
  value: Float32Array[][];
}

export class KVCacheManager {
  private cache: Map<number, KVCache> = new Map(); // layer -> cache

  /**
   * Initialize KV cache for a layer
   */
  initializeLayer(layerId: number, numHeads: number, headDim: number): void {
    this.cache.set(layerId, {
      key: [],
      value: [],
    });
  }

  /**
   * Get KV cache for a layer
   */
  getCache(layerId: number): KVCache | undefined {
    return this.cache.get(layerId);
  }

  /**
   * Update KV cache for a layer
   */
  updateCache(
    layerId: number,
    key: Float32Array[],
    value: Float32Array[]
  ): void {
    const cache = this.cache.get(layerId);
    if (cache) {
      cache.key.push(...(key as any));
      cache.value.push(...(value as any));
    }
  }

  /**
   * Clear cache for a layer
   */
  clearLayer(layerId: number): void {
    const cache = this.cache.get(layerId);
    if (cache) {
      cache.key = [];
      cache.value = [];
    }
  }

  /**
   * Clear all caches
   */
  clearAll(): void {
    this.cache.clear();
  }

  /**
   * Get cache size for a layer
   */
  getCacheSize(layerId: number): number {
    const cache = this.cache.get(layerId);
    return cache ? cache.key.length : 0;
  }
}
