/**
 * Fine-tuning Module
 * Main exports for the fine-tuning system
 */

// Data Collection
export { DataCollector, type DataCollectorConfig } from './data-collector';
export { ChatLogger } from './chat-logger';
export { WebScraper, type ScrapedContent } from './web-scraper';
export { ContentProcessor, type ImageContent, type TextContent } from './content-processor';
export { TrainingQueue, TrainingPriority, type TrainingExample } from './training-queue';

// Fine-tuning Infrastructure
export { LoRAAdapter, type LoRAConfig, type LoRALayer } from './lora';
export { GradientComputer, type LayerGradients, type ActivationCache } from './gradients';
export { WebGPUTrainingOps } from './webgpu-training-ops';
export { WeightUpdater, type WeightSnapshot, type UpdateConfig } from './weight-updater';
export { TrainingEngine, type TrainingConfig, type TrainingMetrics } from './training-engine';

// Background Training
export { TrainingScheduler, type SchedulerConfig } from './training-scheduler';
export { BatchManager, type BatchConfig } from './batch-manager';
export { TrainingWorker, workerImplementation } from './training-worker';

// Import Handlers
export { WebsiteImporter, type WebsiteImportResult } from './importers/website-importer';
export { ImageImporter, type ImageImportResult } from './importers/image-importer';
export { TextImporter, type TextImportResult } from './importers/text-importer';

// Fast Learning & Simulated Neurons
export { FastLearner, type FastLearnerConfig, type EpisodicMemory } from './fast-learner';
export { SimulatedNeuralNetwork, type NeuronConfig, type Neuron } from './simulated-neurons';
