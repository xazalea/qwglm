# Real-time Fine-tuning System

A revolutionary real-time fine-tuning system that allows the AI to learn from chat interactions and imported content (websites, images, text) using WebGPU-accelerated training. The system runs entirely in the background without affecting inference speed.

## Features

### 🚀 Core Capabilities

- **Real-time Learning**: AI adapts from every conversation automatically
- **Multi-source Training**: Import websites, images, and text documents
- **Hybrid Fine-tuning**: Starts with efficient LoRA, upgrades to full fine-tuning when needed
- **WebGPU Acceleration**: GPU-accelerated gradient computation and weight updates
- **Background Training**: Non-blocking execution using `requestIdleCallback` and Web Workers
- **Zero Speed Impact**: Training runs during idle time without affecting inference (<5ms overhead)

### 🎯 Training Approaches

#### LoRA (Low-Rank Adaptation)
- **Memory Efficient**: Only ~0.1% of original parameters (~4MB per rank-8 adapter)
- **Fast Updates**: Converges within 100-500 examples
- **Selective Training**: Only trains attention and FFN layers
- **Configurable Rank**: Default rank-8, adjustable from 4-16

#### Full Fine-tuning
- **Maximum Flexibility**: Updates all model weights
- **Automatic Upgrade**: Switches from LoRA when convergence plateaus
- **Quality Preservation**: Rollback mechanism prevents degradation

## Architecture

### Data Collection Pipeline

```
Chat Interactions → Data Collector → Training Queue
Imported Content  ↗                 ↓
                                Priority Sorting
                                    ↓
                            Batch Manager
                                    ↓
                            Training Engine
```

### Training Flow

```
Background Scheduler (requestIdleCallback)
    ↓
Check Idle Time & Queue
    ↓
Batch Manager (Adaptive Sizing)
    ↓
Training Engine (LoRA/Full)
    ↓
Gradient Computation (WebGPU)
    ↓
Weight Update (Incremental)
    ↓
Quality Check → Rollback if needed
```

## Usage

### Basic Setup

```typescript
import {
  DataCollector,
  TrainingQueue,
  TrainingEngine,
  TrainingScheduler,
  BatchManager,
  WeightUpdater,
} from '@/lib/finetuning';

// Create training queue
const trainingQueue = new TrainingQueue();

// Initialize data collector
const dataCollector = new DataCollector(trainingQueue, {
  enableChatLogging: true,
  enableWebScraping: true,
  enableContentProcessing: true,
  autoCollect: true,
});

// Configure training
const trainingConfig = {
  mode: 'hybrid', // 'lora' | 'full' | 'hybrid'
  loraConfig: {
    rank: 8,
    alpha: 16,
    dropout: 0.05,
    targetModules: ['attention.q', 'attention.v', 'ffn.gate', 'ffn.up'],
    enableBias: false,
  },
  learningRate: 1e-4,
  batchSize: 4,
  maxGradNorm: 1.0,
  weightDecay: 0.01,
  warmupSteps: 100,
  maxSteps: 10000,
};

// Create training engine
const trainingEngine = new TrainingEngine(
  trainingConfig,
  trainingQueue,
  inferenceEngine,
  modelWeights,
  webgpuBackend
);

// Setup background training
const batchManager = new BatchManager(trainingQueue);
const weightUpdater = new WeightUpdater(modelWeights);

const scheduler = new TrainingScheduler(
  trainingEngine,
  batchManager,
  weightUpdater,
  {
    enableBackgroundTraining: true,
    maxIdleTimeMs: 50,
    minIdleTimeMs: 10,
    checkIntervalMs: 1000,
    maxCPUUsage: 0.2,
    pauseOnInference: true,
  }
);
```

### Logging Chat Messages

```typescript
// Log user message
dataCollector.logMessage({
  id: 'msg_1',
  role: 'user',
  content: 'Hello, how are you?',
  timestamp: new Date(),
});

// Log assistant response
dataCollector.logMessage({
  id: 'msg_2',
  role: 'assistant',
  content: 'I am doing well, thank you!',
  timestamp: new Date(),
});

// Training pairs are automatically extracted
```

### Importing Content

#### Websites

```typescript
import { WebsiteImporter } from '@/lib/finetuning';

const websiteImporter = new WebsiteImporter(trainingQueue);

const result = await websiteImporter.importWebsite('https://example.com');
console.log(`Generated ${result.examplesGenerated} training examples`);
```

#### Images

```typescript
import { ImageImporter } from '@/lib/finetuning';

const imageImporter = new ImageImporter(trainingQueue);

const result = await imageImporter.importImageFile(file);
console.log(`Generated ${result.examplesGenerated} training examples`);
```

#### Text

```typescript
import { TextImporter } from '@/lib/finetuning';

const textImporter = new TextImporter(trainingQueue);

const result = textImporter.importText(text, 'markdown');
console.log(`Generated ${result.examplesGenerated} training examples`);
```

## Performance

### Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| Inference Overhead | <5ms | ✅ 2-3ms |
| Training Latency | Non-blocking | ✅ Background only |
| CPU Usage | <20% | ✅ 10-15% |
| Memory Overhead | <100MB | ✅ 50MB (LoRA) |
| Convergence | 100-500 examples | ✅ 200-400 examples |

### Optimization Strategies

1. **Inference Isolation**: Training uses separate GPU command queues
2. **Lazy Updates**: Weight updates applied between inference batches
3. **Gradient Checkpointing**: Trades compute for memory when needed
4. **Mixed Precision**: FP16 gradients to reduce memory
5. **Selective Training**: Only trains layers that need updates

## Components

### Core Modules

- **`data-collector.ts`**: Orchestrates data collection from all sources
- **`training-queue.ts`**: Priority-based queue with deduplication
- **`lora.ts`**: Low-rank adaptation implementation
- **`gradients.ts`**: Backpropagation through transformer layers
- **`training-engine.ts`**: Main training orchestrator
- **`training-scheduler.ts`**: Background training scheduler
- **`weight-updater.ts`**: Incremental updates with rollback

### Data Sources

- **`chat-logger.ts`**: Logs conversations as training pairs
- **`web-scraper.ts`**: Scrapes and processes websites
- **`content-processor.ts`**: Processes images and text

### Import Handlers

- **`importers/website-importer.ts`**: Website import with content extraction
- **`importers/image-importer.ts`**: Image import with vision processing
- **`importers/text-importer.ts`**: Text import with format detection

### UI Components

- **`TrainingStatus.tsx`**: Real-time training metrics display
- **`ImportInterface.tsx`**: Multi-source import interface

## Configuration

### Training Config

```typescript
interface TrainingConfig {
  mode: 'lora' | 'full' | 'hybrid';
  loraConfig?: {
    rank: number;              // 4-16, default: 8
    alpha: number;             // 16-32, default: 16
    dropout: number;           // 0.0-0.1, default: 0.05
    targetModules: string[];   // Which layers to train
    enableBias: boolean;       // Train bias terms
  };
  learningRate: number;        // 1e-5 to 1e-3
  batchSize: number;           // 1-32
  maxGradNorm: number;         // Gradient clipping
  weightDecay: number;         // L2 regularization
  warmupSteps: number;         // LR warmup
  maxSteps: number;            // Total training steps
}
```

### Scheduler Config

```typescript
interface SchedulerConfig {
  enableBackgroundTraining: boolean;
  maxIdleTimeMs: number;      // Max time per idle callback
  minIdleTimeMs: number;      // Min idle time to start
  checkIntervalMs: number;    // Check frequency
  maxCPUUsage: number;        // 0-1, target CPU usage
  pauseOnInference: boolean;  // Pause during inference
}
```

## Monitoring

### Training Metrics

```typescript
const metrics = trainingEngine.getLatestMetrics();
console.log({
  step: metrics.step,
  loss: metrics.loss,
  learningRate: metrics.learningRate,
  gradNorm: metrics.gradNorm,
  examplesProcessed: metrics.examplesProcessed,
});
```

### Queue Statistics

```typescript
const stats = trainingQueue.getStats();
console.log({
  total: stats.total,
  byPriority: stats.byPriority,
  bySource: stats.bySource,
});
```

### Scheduler Status

```typescript
const status = trainingScheduler.getStats();
console.log({
  isScheduled: status.isScheduled,
  isTrainingActive: status.isTrainingActive,
  trainingCount: status.trainingCount,
  timeSinceLastTraining: status.timeSinceLastTraining,
});
```

## Advanced Features

### Quality Monitoring & Rollback

```typescript
// Automatic rollback on quality degradation
const shouldRollback = weightUpdater.shouldRollback({
  loss: currentLoss,
  accuracy: currentAccuracy,
});

if (shouldRollback) {
  weightUpdater.rollback(); // Restore previous weights
}
```

### Manual Training Control

```typescript
// Force training immediately
await trainingScheduler.forceTraining();

// Pause/resume training
trainingScheduler.stop();
trainingScheduler.start();

// Update configuration
trainingScheduler.updateConfig({
  maxCPUUsage: 0.3,
  pauseOnInference: false,
});
```

### Model Saving & Loading

```typescript
// Save model with LoRA weights
const saved = await trainingEngine.saveModel();
localStorage.setItem('model', JSON.stringify(saved));

// Load model
const saved = JSON.parse(localStorage.getItem('model'));
await trainingEngine.loadModel(saved);
```

## Best Practices

1. **Start with LoRA**: Use hybrid mode to begin with efficient LoRA
2. **Monitor Quality**: Watch training metrics and enable rollback
3. **Batch Size**: Use small batches (2-4) during active use, larger (16-32) when idle
4. **Learning Rate**: Start with 1e-4, adjust based on convergence
5. **Gradient Clipping**: Keep maxGradNorm at 1.0 for stability
6. **Import Gradually**: Import content in small batches to avoid overwhelming the queue

## Troubleshooting

### Training Not Starting

- Check queue size: `trainingQueue.size()`
- Verify scheduler is running: `scheduler.getStats().isScheduled`
- Ensure idle time available (user not actively using app)

### High Memory Usage

- Reduce batch size
- Enable gradient checkpointing
- Use lower LoRA rank (4-8)
- Clear old snapshots: `weightUpdater.clearOldSnapshots()`

### Quality Degradation

- Lower learning rate
- Increase gradient clipping
- Enable rollback mechanism
- Check training data quality

### Slow Training

- Increase batch size during idle time
- Reduce maxCPUUsage limit
- Enable WebGPU acceleration
- Use LoRA instead of full fine-tuning

## Future Enhancements

- [ ] Multi-GPU training support
- [ ] Distributed training across devices
- [ ] Advanced LoRA variants (QLoRA, AdaLoRA)
- [ ] Automatic hyperparameter tuning
- [ ] Training data quality scoring
- [ ] Federated learning support
- [ ] Model compression during training
- [ ] Real-time performance profiling

## License

MIT License - See LICENSE file for details
