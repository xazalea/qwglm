/**
 * Training Queue
 * Manages training data queue with priority levels
 */

export enum TrainingPriority {
  HIGH = 3,    // Recent chat interactions
  MEDIUM = 2,  // Imported content
  LOW = 1,     // Historical data
}

export interface TrainingExample {
  id: string;
  input: string;
  output: string;
  priority: TrainingPriority;
  timestamp: number;
  metadata?: {
    source: 'chat' | 'import' | 'manual';
    contentType?: 'text' | 'image' | 'web';
    url?: string;
  };
}

export class TrainingQueue {
  private queue: TrainingExample[] = [];
  private processedIds: Set<string> = new Set();
  private maxQueueSize: number = 10000;

  /**
   * Add training example to queue
   */
  add(example: TrainingExample): void {
    // Deduplicate based on ID
    if (this.processedIds.has(example.id)) {
      return;
    }

    // Remove oldest items if queue is full
    if (this.queue.length >= this.maxQueueSize) {
      const removed = this.queue.shift();
      if (removed) {
        this.processedIds.delete(removed.id);
      }
    }

    this.queue.push(example);
    this.sortByPriority();
  }

  /**
   * Add multiple examples at once
   */
  addBatch(examples: TrainingExample[]): void {
    examples.forEach((ex) => this.add(ex));
  }

  /**
   * Get next batch of examples
   */
  getBatch(size: number): TrainingExample[] {
    const batch = this.queue.slice(0, size);
    return batch;
  }

  /**
   * Remove examples from queue after processing
   */
  removeBatch(ids: string[]): void {
    const idSet = new Set(ids);
    this.queue = this.queue.filter((ex) => !idSet.has(ex.id));
    ids.forEach((id) => this.processedIds.add(id));
  }

  /**
   * Get queue size
   */
  size(): number {
    return this.queue.length;
  }

  /**
   * Check if queue is empty
   */
  isEmpty(): boolean {
    return this.queue.length === 0;
  }

  /**
   * Clear the queue
   */
  clear(): void {
    this.queue = [];
    this.processedIds.clear();
  }

  /**
   * Get statistics
   */
  getStats(): {
    total: number;
    byPriority: Record<TrainingPriority, number>;
    bySource: Record<string, number>;
  } {
    const byPriority: Record<TrainingPriority, number> = {
      [TrainingPriority.HIGH]: 0,
      [TrainingPriority.MEDIUM]: 0,
      [TrainingPriority.LOW]: 0,
    };

    const bySource: Record<string, number> = {};

    this.queue.forEach((ex) => {
      byPriority[ex.priority]++;
      const source = ex.metadata?.source || 'unknown';
      bySource[source] = (bySource[source] || 0) + 1;
    });

    return {
      total: this.queue.length,
      byPriority,
      bySource,
    };
  }

  /**
   * Sort queue by priority (high to low) and timestamp (recent first)
   */
  private sortByPriority(): void {
    this.queue.sort((a, b) => {
      if (a.priority !== b.priority) {
        return b.priority - a.priority;
      }
      return b.timestamp - a.timestamp;
    });
  }

  /**
   * Filter examples by quality
   */
  filterByQuality(minLength: number = 10): void {
    this.queue = this.queue.filter((ex) => {
      const inputLength = ex.input.trim().length;
      const outputLength = ex.output.trim().length;
      return inputLength >= minLength && outputLength >= minLength;
    });
  }
}
