/**
 * Website Importer
 * Import and process websites for training
 */

import { WebScraper, type ScrapedContent } from '../web-scraper';
import { TrainingQueue, type TrainingExample } from '../training-queue';

export interface WebsiteImportResult {
  success: boolean;
  url: string;
  examplesGenerated: number;
  error?: string;
}

export class WebsiteImporter {
  private scraper: WebScraper;
  private trainingQueue: TrainingQueue;

  constructor(trainingQueue: TrainingQueue) {
    this.scraper = new WebScraper();
    this.trainingQueue = trainingQueue;
  }

  /**
   * Import single website
   */
  async importWebsite(url: string): Promise<WebsiteImportResult> {
    try {
      // Validate URL
      if (!this.isValidUrl(url)) {
        return {
          success: false,
          url,
          examplesGenerated: 0,
          error: 'Invalid URL',
        };
      }

      console.log(`Importing website: ${url}`);

      // Scrape website
      const scraped = await this.scraper.scrapeWebsite(url);

      // Generate training examples
      const examples = this.scraper.generateTrainingExamples(scraped);

      // Add to training queue
      this.trainingQueue.addBatch(examples);

      console.log(`Generated ${examples.length} training examples from ${url}`);

      return {
        success: true,
        url,
        examplesGenerated: examples.length,
      };
    } catch (error) {
      console.error(`Error importing website ${url}:`, error);
      return {
        success: false,
        url,
        examplesGenerated: 0,
        error: (error as Error).message,
      };
    }
  }

  /**
   * Import multiple websites
   */
  async importWebsites(urls: string[]): Promise<WebsiteImportResult[]> {
    const results: WebsiteImportResult[] = [];

    for (const url of urls) {
      const result = await this.importWebsite(url);
      results.push(result);

      // Add delay to avoid overwhelming the server
      await this.delay(500);
    }

    return results;
  }

  /**
   * Import website with content filtering
   */
  async importWebsiteFiltered(
    url: string,
    options: {
      minContentLength?: number;
      maxExamples?: number;
      contentTypes?: string[];
    }
  ): Promise<WebsiteImportResult> {
    try {
      const scraped = await this.scraper.scrapeWebsite(url);
      let examples = this.scraper.generateTrainingExamples(scraped);

      // Filter by content length
      if (options.minContentLength) {
        examples = examples.filter(
          (ex) => ex.output.length >= options.minContentLength!
        );
      }

      // Limit number of examples
      if (options.maxExamples) {
        examples = examples.slice(0, options.maxExamples);
      }

      this.trainingQueue.addBatch(examples);

      return {
        success: true,
        url,
        examplesGenerated: examples.length,
      };
    } catch (error) {
      return {
        success: false,
        url,
        examplesGenerated: 0,
        error: (error as Error).message,
      };
    }
  }

  /**
   * Validate URL
   */
  private isValidUrl(url: string): boolean {
    try {
      const parsed = new URL(url);
      return parsed.protocol === 'http:' || parsed.protocol === 'https:';
    } catch {
      return false;
    }
  }

  /**
   * Delay helper
   */
  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Get import statistics
   */
  getStats(): {
    queueSize: number;
  } {
    return {
      queueSize: this.trainingQueue.size(),
    };
  }
}
