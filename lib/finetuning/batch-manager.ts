/**
 * Batch Manager
 * Manages training batches with adaptive sizing
 */

import { TrainingQueue, TrainingExample, TrainingPriority } from './training-queue';

export interface BatchConfig {
  minBatchSize: number;
  maxBatchSize: number;
  activeBatchSize: number; // During active chat
  idleBatchSize: number; // During idle time
  priorityWeights: Record<TrainingPriority, number>;
}

export class BatchManager {
  private queue: TrainingQueue;
  private config: BatchConfig;
  private isUserActive: boolean = false;
  private lastActivityTime: number = Date.now();
  private activityTimeoutMs: number = 5000; // 5 seconds

  constructor(
    queue: TrainingQueue,
    config: BatchConfig = {
      minBatchSize: 1,
      maxBatchSize: 32,
      activeBatchSize: 2,
      idleBatchSize: 16,
      priorityWeights: {
        [TrainingPriority.HIGH]: 3.0,
        [TrainingPriority.MEDIUM]: 2.0,
        [TrainingPriority.LOW]: 1.0,
      },
    }
  ) {
    this.queue = queue;
    this.config = config;

    // Monitor user activity
    this.setupActivityMonitoring();
  }

  /**
   * Setup user activity monitoring
   */
  private setupActivityMonitoring(): void {
    if (typeof window !== 'undefined') {
      const events = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'];
      
      events.forEach((event) => {
        window.addEventListener(event, () => {
          this.isUserActive = true;
          this.lastActivityTime = Date.now();
        }, { passive: true });
      });

      // Check activity periodically
      setInterval(() => {
        const timeSinceActivity = Date.now() - this.lastActivityTime;
        if (timeSinceActivity > this.activityTimeoutMs) {
          this.isUserActive = false;
        }
      }, 1000);
    }
  }

  /**
   * Get next batch with adaptive sizing
   */
  getNextBatch(): TrainingExample[] {
    if (this.queue.isEmpty()) {
      return [];
    }

    const batchSize = this.getAdaptiveBatchSize();
    const batch = this.queue.getBatch(batchSize);

    // Sort by priority and quality
    return this.sortBatch(batch);
  }

  /**
   * Get adaptive batch size based on user activity
   */
  private getAdaptiveBatchSize(): number {
    if (this.isUserActive) {
      return this.config.activeBatchSize;
    } else {
      return this.config.idleBatchSize;
    }
  }

  /**
   * Sort batch by priority and other factors
   */
  private sortBatch(batch: TrainingExample[]): TrainingExample[] {
    return batch.sort((a, b) => {
      // First by priority
      const priorityDiff = this.config.priorityWeights[b.priority] - this.config.priorityWeights[a.priority];
      if (Math.abs(priorityDiff) > 0.1) {
        return priorityDiff;
      }

      // Then by recency
      return b.timestamp - a.timestamp;
    });
  }

  /**
   * Get batch with specific size
   */
  getBatch(size: number): TrainingExample[] {
    const actualSize = Math.min(size, this.config.maxBatchSize);
    const actualSize2 = Math.max(actualSize, this.config.minBatchSize);
    
    if (this.queue.isEmpty()) {
      return [];
    }

    const batch = this.queue.getBatch(actualSize2);
    return this.sortBatch(batch);
  }

  /**
   * Get batch for high priority items only
   */
  getHighPriorityBatch(): TrainingExample[] {
    const allBatch = this.queue.getBatch(this.config.maxBatchSize);
    const highPriority = allBatch.filter((ex) => ex.priority === TrainingPriority.HIGH);
    return highPriority.slice(0, this.config.activeBatchSize);
  }

  /**
   * Check if there are enough examples to train
   */
  hasEnoughExamples(): boolean {
    return this.queue.size() >= this.config.minBatchSize;
  }

  /**
   * Get recommended batch size
   */
  getRecommendedBatchSize(): number {
    const queueSize = this.queue.size();
    const adaptive = this.getAdaptiveBatchSize();

    return Math.min(adaptive, queueSize, this.config.maxBatchSize);
  }

  /**
   * Check if user is active
   */
  isActive(): boolean {
    return this.isUserActive;
  }

  /**
   * Mark examples as processed
   */
  markProcessed(examples: TrainingExample[]): void {
    const ids = examples.map((ex) => ex.id);
    this.queue.removeBatch(ids);
  }

  /**
   * Get batch statistics
   */
  getStats(): {
    queueSize: number;
    isUserActive: boolean;
    recommendedBatchSize: number;
    timeSinceActivity: number;
  } {
    return {
      queueSize: this.queue.size(),
      isUserActive: this.isUserActive,
      recommendedBatchSize: this.getRecommendedBatchSize(),
      timeSinceActivity: Date.now() - this.lastActivityTime,
    };
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<BatchConfig>): void {
    this.config = { ...this.config, ...config };
  }
}
