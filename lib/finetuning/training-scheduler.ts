/**
 * Training Scheduler
 * Schedules training in background using requestIdleCallback
 */

import { TrainingEngine, type TrainingMetrics } from './training-engine';
import { BatchManager } from './batch-manager';
import { WeightUpdater } from './weight-updater';

export interface SchedulerConfig {
  enableBackgroundTraining: boolean;
  maxIdleTimeMs: number; // Maximum time to use per idle callback
  minIdleTimeMs: number; // Minimum idle time before starting training
  checkIntervalMs: number; // How often to check for training opportunities
  maxCPUUsage: number; // Maximum CPU usage (0-1)
  pauseOnInference: boolean; // Pause training during inference
}

export class TrainingScheduler {
  private trainingEngine: TrainingEngine;
  private batchManager: BatchManager;
  private weightUpdater: WeightUpdater;
  private config: SchedulerConfig;
  
  private isScheduled: boolean = false;
  private isTrainingActive: boolean = false;
  private checkIntervalId?: number;
  private lastTrainingTime: number = 0;
  private trainingCount: number = 0;
  private inferenceInProgress: boolean = false;

  constructor(
    trainingEngine: TrainingEngine,
    batchManager: BatchManager,
    weightUpdater: WeightUpdater,
    config: SchedulerConfig = {
      enableBackgroundTraining: true,
      maxIdleTimeMs: 50, // 50ms per idle callback
      minIdleTimeMs: 10, // Start if at least 10ms idle time
      checkIntervalMs: 1000, // Check every second
      maxCPUUsage: 0.2, // 20% CPU usage
      pauseOnInference: true,
    }
  ) {
    this.trainingEngine = trainingEngine;
    this.batchManager = batchManager;
    this.weightUpdater = weightUpdater;
    this.config = config;

    if (config.enableBackgroundTraining) {
      this.start();
    }
  }

  /**
   * Start background training scheduler
   */
  start(): void {
    if (this.isScheduled) {
      return;
    }

    this.isScheduled = true;
    this.scheduleNextTraining();

    console.log('Background training scheduler started');
  }

  /**
   * Stop background training scheduler
   */
  stop(): void {
    this.isScheduled = false;
    
    if (this.checkIntervalId) {
      clearInterval(this.checkIntervalId);
      this.checkIntervalId = undefined;
    }

    console.log('Background training scheduler stopped');
  }

  /**
   * Schedule next training session
   */
  private scheduleNextTraining(): void {
    if (!this.isScheduled) {
      return;
    }

    // Use requestIdleCallback if available, otherwise use setTimeout
    if (typeof requestIdleCallback !== 'undefined') {
      requestIdleCallback(
        (deadline) => this.handleIdleCallback(deadline),
        { timeout: this.config.checkIntervalMs }
      );
    } else {
      setTimeout(() => this.checkAndTrain(), this.config.checkIntervalMs);
    }
  }

  /**
   * Handle idle callback
   */
  private async handleIdleCallback(deadline: IdleDeadline): Promise<void> {
    const timeRemaining = deadline.timeRemaining();

    // Check if we have enough idle time
    if (timeRemaining < this.config.minIdleTimeMs) {
      this.scheduleNextTraining();
      return;
    }

    // Check if we should train
    if (this.shouldTrain()) {
      await this.performTraining(Math.min(timeRemaining, this.config.maxIdleTimeMs));
    }

    // Schedule next training
    this.scheduleNextTraining();
  }

  /**
   * Check conditions and train if appropriate
   */
  private async checkAndTrain(): Promise<void> {
    if (this.shouldTrain()) {
      await this.performTraining(this.config.maxIdleTimeMs);
    }

    if (this.isScheduled) {
      this.scheduleNextTraining();
    }
  }

  /**
   * Check if we should train now
   */
  private shouldTrain(): boolean {
    // Don't train if already training
    if (this.isTrainingActive) {
      return false;
    }

    // Don't train if inference is in progress
    if (this.config.pauseOnInference && this.inferenceInProgress) {
      return false;
    }

    // Check if we have examples to train on
    if (!this.batchManager.hasEnoughExamples()) {
      return false;
    }

    // Check CPU usage (simplified - would need actual CPU monitoring)
    const timeSinceLastTraining = Date.now() - this.lastTrainingTime;
    const minInterval = 1000 / this.config.maxCPUUsage;
    
    if (timeSinceLastTraining < minInterval) {
      return false;
    }

    return true;
  }

  /**
   * Perform training session
   */
  private async performTraining(maxTimeMs: number): Promise<void> {
    this.isTrainingActive = true;
    const startTime = Date.now();

    try {
      // Get batch
      const batch = this.batchManager.getNextBatch();
      
      if (batch.length === 0) {
        return;
      }

      console.log(`Background training: ${batch.length} examples`);

      // Train on batch
      const metrics = await this.trainingEngine.trainBatch(batch);

      // Mark as processed
      this.batchManager.markProcessed(batch);

      // Check training time
      const trainingTime = Date.now() - startTime;
      if (trainingTime > maxTimeMs * 1.5) {
        console.warn(`Training took ${trainingTime}ms, exceeding idle time budget`);
      }

      // Update counters
      this.trainingCount++;
      this.lastTrainingTime = Date.now();

      // Create snapshot periodically
      if (this.trainingCount % 10 === 0) {
        this.weightUpdater.createSnapshot({
          loss: metrics.loss,
        });
      }

      // Check for quality degradation
      const shouldRollback = this.weightUpdater.shouldRollback({
        loss: metrics.loss,
      });

      if (shouldRollback) {
        console.warn('Quality degraded, rolling back weights');
        this.weightUpdater.rollback();
      }

    } catch (error) {
      console.error('Error during background training:', error);
    } finally {
      this.isTrainingActive = false;
    }
  }

  /**
   * Notify scheduler about inference state
   */
  setInferenceInProgress(inProgress: boolean): void {
    this.inferenceInProgress = inProgress;
  }

  /**
   * Force training now (bypass idle check)
   */
  async forceTraining(): Promise<void> {
    if (this.isTrainingActive) {
      console.warn('Training already in progress');
      return;
    }

    await this.performTraining(1000); // Allow up to 1 second
  }

  /**
   * Get scheduler statistics
   */
  getStats(): {
    isScheduled: boolean;
    isTrainingActive: boolean;
    trainingCount: number;
    lastTrainingTime: number;
    timeSinceLastTraining: number;
    batchStats: ReturnType<BatchManager['getStats']>;
  } {
    return {
      isScheduled: this.isScheduled,
      isTrainingActive: this.isTrainingActive,
      trainingCount: this.trainingCount,
      lastTrainingTime: this.lastTrainingTime,
      timeSinceLastTraining: Date.now() - this.lastTrainingTime,
      batchStats: this.batchManager.getStats(),
    };
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<SchedulerConfig>): void {
    this.config = { ...this.config, ...config };

    if (config.enableBackgroundTraining !== undefined) {
      if (config.enableBackgroundTraining) {
        this.start();
      } else {
        this.stop();
      }
    }
  }
}
