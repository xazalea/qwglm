/**
 * Weight Updater
 * Incremental weight update system with rollback mechanism
 */

import type { ModelWeights } from '../model-runtime/quantization/loader';

export interface WeightSnapshot {
  timestamp: number;
  weights: Map<string, Float32Array>;
  metrics?: {
    loss: number;
    perplexity?: number;
    accuracy?: number;
  };
}

export interface UpdateConfig {
  updateInterval: number; // milliseconds between updates
  snapshotInterval: number; // number of updates between snapshots
  maxSnapshots: number; // maximum snapshots to keep
  rollbackThreshold: number; // quality degradation threshold for rollback
}

export class WeightUpdater {
  private currentWeights: ModelWeights;
  private snapshots: WeightSnapshot[] = [];
  private config: UpdateConfig;
  private updateCount: number = 0;
  private lastMetrics?: WeightSnapshot['metrics'];

  constructor(
    initialWeights: ModelWeights,
    config: UpdateConfig = {
      updateInterval: 1000,
      snapshotInterval: 10,
      maxSnapshots: 5,
      rollbackThreshold: 0.2, // 20% degradation triggers rollback
    }
  ) {
    this.currentWeights = initialWeights;
    this.config = config;
    
    // Create initial snapshot
    this.createSnapshot();
  }

  /**
   * Apply incremental weight update
   */
  applyUpdate(
    weightName: string,
    delta: Float32Array,
    alpha: number = 0.1 // Learning rate / update strength
  ): void {
    const currentWeight = this.currentWeights.get(weightName);
    if (!currentWeight) {
      console.warn(`Weight ${weightName} not found`);
      return;
    }

    // Incremental update: W = W + alpha * delta
    for (let i = 0; i < currentWeight.length; i++) {
      currentWeight[i] += alpha * delta[i];
    }

    this.updateCount++;

    // Create snapshot periodically
    if (this.updateCount % this.config.snapshotInterval === 0) {
      this.createSnapshot();
    }
  }

  /**
   * Apply batch of weight updates
   */
  applyBatchUpdate(
    updates: Map<string, Float32Array>,
    alpha: number = 0.1
  ): void {
    for (const [name, delta] of updates) {
      this.applyUpdate(name, delta, alpha);
    }
  }

  /**
   * Replace entire weight
   */
  replaceWeight(weightName: string, newWeight: Float32Array): void {
    const currentWeight = this.currentWeights.get(weightName);
    if (!currentWeight) {
      console.warn(`Weight ${weightName} not found`);
      return;
    }

    currentWeight.set(newWeight);
    this.updateCount++;
  }

  /**
   * Create weight snapshot
   */
  createSnapshot(metrics?: WeightSnapshot['metrics']): void {
    const snapshot: WeightSnapshot = {
      timestamp: Date.now(),
      weights: new Map(),
      metrics,
    };

    // Deep copy weights
    for (const [name, weight] of this.currentWeights) {
      snapshot.weights.set(name, new Float32Array(weight));
    }

    this.snapshots.push(snapshot);
    this.lastMetrics = metrics;

    // Remove old snapshots
    if (this.snapshots.length > this.config.maxSnapshots) {
      this.snapshots.shift();
    }

    console.log(`Created weight snapshot #${this.snapshots.length}`);
  }

  /**
   * Check if quality has degraded and should rollback
   */
  shouldRollback(currentMetrics: WeightSnapshot['metrics']): boolean {
    if (!this.lastMetrics || !currentMetrics) {
      return false;
    }

    // Check loss increase
    if (currentMetrics.loss > this.lastMetrics.loss * (1 + this.config.rollbackThreshold)) {
      console.warn('Loss increased significantly, rollback recommended');
      return true;
    }

    // Check accuracy decrease
    if (currentMetrics.accuracy && this.lastMetrics.accuracy) {
      const accuracyDrop = (this.lastMetrics.accuracy - currentMetrics.accuracy) / this.lastMetrics.accuracy;
      if (accuracyDrop > this.config.rollbackThreshold) {
        console.warn('Accuracy decreased significantly, rollback recommended');
        return true;
      }
    }

    return false;
  }

  /**
   * Rollback to previous snapshot
   */
  rollback(snapshotIndex?: number): boolean {
    if (this.snapshots.length === 0) {
      console.warn('No snapshots available for rollback');
      return false;
    }

    const targetIndex = snapshotIndex !== undefined 
      ? snapshotIndex 
      : this.snapshots.length - 1;

    if (targetIndex < 0 || targetIndex >= this.snapshots.length) {
      console.warn('Invalid snapshot index');
      return false;
    }

    const snapshot = this.snapshots[targetIndex];

    // Restore weights
    for (const [name, weight] of snapshot.weights) {
      const currentWeight = this.currentWeights.get(name);
      if (currentWeight) {
        currentWeight.set(weight);
      }
    }

    this.lastMetrics = snapshot.metrics;

    console.log(`Rolled back to snapshot from ${new Date(snapshot.timestamp).toISOString()}`);
    return true;
  }

  /**
   * Get current weights
   */
  getCurrentWeights(): ModelWeights {
    return this.currentWeights;
  }

  /**
   * Get weight by name
   */
  getWeight(name: string): Float32Array | undefined {
    return this.currentWeights.get(name);
  }

  /**
   * Get snapshots
   */
  getSnapshots(): WeightSnapshot[] {
    return this.snapshots;
  }

  /**
   * Get last snapshot
   */
  getLastSnapshot(): WeightSnapshot | undefined {
    return this.snapshots[this.snapshots.length - 1];
  }

  /**
   * Clear all snapshots except the last one
   */
  clearOldSnapshots(): void {
    if (this.snapshots.length > 1) {
      const lastSnapshot = this.snapshots[this.snapshots.length - 1];
      this.snapshots = [lastSnapshot];
    }
  }

  /**
   * Export weights for saving
   */
  exportWeights(): Record<string, Float32Array> {
    const exported: Record<string, Float32Array> = {};
    
    for (const [name, weight] of this.currentWeights) {
      exported[name] = new Float32Array(weight);
    }

    return exported;
  }

  /**
   * Import weights
   */
  importWeights(weights: Record<string, Float32Array>): void {
    for (const [name, weight] of Object.entries(weights)) {
      const currentWeight = this.currentWeights.get(name);
      if (currentWeight) {
        currentWeight.set(weight);
      } else {
        this.currentWeights.set(name, new Float32Array(weight));
      }
    }

    this.createSnapshot();
  }

  /**
   * Get update statistics
   */
  getStats(): {
    totalUpdates: number;
    snapshotCount: number;
    lastSnapshotAge: number;
    lastMetrics?: WeightSnapshot['metrics'];
  } {
    const lastSnapshot = this.getLastSnapshot();
    
    return {
      totalUpdates: this.updateCount,
      snapshotCount: this.snapshots.length,
      lastSnapshotAge: lastSnapshot ? Date.now() - lastSnapshot.timestamp : 0,
      lastMetrics: this.lastMetrics,
    };
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<UpdateConfig>): void {
    this.config = { ...this.config, ...config };
  }
}
