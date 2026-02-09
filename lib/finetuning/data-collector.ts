/**
 * Data Collector
 * Main data collection orchestrator
 */

import { ChatLogger } from './chat-logger';
import { WebScraper } from './web-scraper';
import { ContentProcessor, ImageContent, TextContent } from './content-processor';
import { TrainingQueue, TrainingExample } from './training-queue';

export interface DataCollectorConfig {
  enableChatLogging: boolean;
  enableWebScraping: boolean;
  enableContentProcessing: boolean;
  autoCollect: boolean;
  collectInterval?: number; // milliseconds
}

export class DataCollector {
  private chatLogger: ChatLogger;
  private webScraper: WebScraper;
  private contentProcessor: ContentProcessor;
  private trainingQueue: TrainingQueue;
  private config: DataCollectorConfig;
  private collectIntervalId?: number;

  constructor(
    trainingQueue: TrainingQueue,
    config: DataCollectorConfig = {
      enableChatLogging: true,
      enableWebScraping: true,
      enableContentProcessing: true,
      autoCollect: true,
      collectInterval: 5000, // 5 seconds
    }
  ) {
    this.trainingQueue = trainingQueue;
    this.config = config;
    this.chatLogger = new ChatLogger();
    this.webScraper = new WebScraper();
    this.contentProcessor = new ContentProcessor();

    if (config.autoCollect) {
      this.startAutoCollect();
    }
  }

  /**
   * Log a chat message
   */
  logMessage(message: {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
  }): void {
    if (!this.config.enableChatLogging) {
      return;
    }

    this.chatLogger.logMessage(message);
  }

  /**
   * Collect training data from chat
   */
  collectFromChat(): void {
    if (!this.config.enableChatLogging) {
      return;
    }

    // Extract training pairs from conversation
    const pairs = this.chatLogger.extractTrainingPairs();
    this.trainingQueue.addBatch(pairs);

    console.log(`Collected ${pairs.length} training examples from chat`);
  }

  /**
   * Collect training data from website
   */
  async collectFromWebsite(url: string): Promise<void> {
    if (!this.config.enableWebScraping) {
      return;
    }

    try {
      const scraped = await this.webScraper.scrapeWebsite(url);
      const examples = this.webScraper.generateTrainingExamples(scraped);
      this.trainingQueue.addBatch(examples);

      console.log(`Collected ${examples.length} training examples from ${url}`);
    } catch (error) {
      console.error('Error collecting from website:', error);
    }
  }

  /**
   * Collect training data from image
   */
  async collectFromImage(image: ImageContent): Promise<void> {
    if (!this.config.enableContentProcessing) {
      return;
    }

    try {
      const examples = await this.contentProcessor.processImage(image);
      this.trainingQueue.addBatch(examples);

      console.log(`Collected ${examples.length} training examples from image`);
    } catch (error) {
      console.error('Error collecting from image:', error);
    }
  }

  /**
   * Collect training data from text
   */
  collectFromText(text: TextContent): void {
    if (!this.config.enableContentProcessing) {
      return;
    }

    try {
      const examples = this.contentProcessor.processText(text);
      this.trainingQueue.addBatch(examples);

      console.log(`Collected ${examples.length} training examples from text`);
    } catch (error) {
      console.error('Error collecting from text:', error);
    }
  }

  /**
   * Collect training data from multiple sources
   */
  async collectBatch(sources: {
    websites?: string[];
    images?: ImageContent[];
    texts?: TextContent[];
  }): Promise<void> {
    const promises: Promise<void>[] = [];

    if (sources.websites) {
      promises.push(...sources.websites.map((url) => this.collectFromWebsite(url)));
    }

    if (sources.images) {
      for (const image of sources.images) {
        promises.push(this.collectFromImage(image));
      }
    }

    if (sources.texts) {
      sources.texts.forEach((text) => this.collectFromText(text));
    }

    await Promise.all(promises);
  }

  /**
   * Start automatic collection
   */
  startAutoCollect(): void {
    if (this.collectIntervalId) {
      return;
    }

    this.collectIntervalId = window.setInterval(() => {
      this.collectFromChat();
    }, this.config.collectInterval || 5000);
  }

  /**
   * Stop automatic collection
   */
  stopAutoCollect(): void {
    if (this.collectIntervalId) {
      window.clearInterval(this.collectIntervalId);
      this.collectIntervalId = undefined;
    }
  }

  /**
   * Get collection statistics
   */
  getStats(): {
    chat: ReturnType<ChatLogger['getStats']>;
    queue: ReturnType<TrainingQueue['getStats']>;
  } {
    return {
      chat: this.chatLogger.getStats(),
      queue: this.trainingQueue.getStats(),
    };
  }

  /**
   * Clear all collected data
   */
  clear(): void {
    this.chatLogger.clear();
    this.trainingQueue.clear();
  }

  /**
   * Get training queue
   */
  getQueue(): TrainingQueue {
    return this.trainingQueue;
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<DataCollectorConfig>): void {
    this.config = { ...this.config, ...config };

    if (config.autoCollect !== undefined) {
      if (config.autoCollect) {
        this.startAutoCollect();
      } else {
        this.stopAutoCollect();
      }
    }
  }
}
