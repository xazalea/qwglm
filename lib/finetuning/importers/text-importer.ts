/**
 * Text Importer
 * Import and process text files for training
 */

import { ContentProcessor, type TextContent } from '../content-processor';
import { TrainingQueue } from '../training-queue';

export interface TextImportResult {
  success: boolean;
  textId: string;
  examplesGenerated: number;
  error?: string;
}

export class TextImporter {
  private processor: ContentProcessor;
  private trainingQueue: TrainingQueue;

  constructor(trainingQueue: TrainingQueue) {
    this.processor = new ContentProcessor();
    this.trainingQueue = trainingQueue;
  }

  /**
   * Import text from file
   */
  async importTextFile(file: File): Promise<TextImportResult> {
    try {
      const text = await this.readFileAsText(file);
      const format = this.detectFormat(file.name, text);
      const textId = this.generateTextId(file.name);

      const textContent: TextContent = {
        id: textId,
        text,
        format,
      };

      const examples = this.processor.processText(textContent);
      this.trainingQueue.addBatch(examples);

      return {
        success: true,
        textId,
        examplesGenerated: examples.length,
      };
    } catch (error) {
      console.error('Error importing text file:', error);
      return {
        success: false,
        textId: file.name,
        examplesGenerated: 0,
        error: (error as Error).message,
      };
    }
  }

  /**
   * Import text directly
   */
  importText(
    text: string,
    format: 'plain' | 'markdown' | 'json' = 'plain'
  ): TextImportResult {
    try {
      const textId = this.generateTextId('direct');

      const textContent: TextContent = {
        id: textId,
        text,
        format,
      };

      const examples = this.processor.processText(textContent);
      this.trainingQueue.addBatch(examples);

      return {
        success: true,
        textId,
        examplesGenerated: examples.length,
      };
    } catch (error) {
      console.error('Error importing text:', error);
      return {
        success: false,
        textId: 'direct',
        examplesGenerated: 0,
        error: (error as Error).message,
      };
    }
  }

  /**
   * Import multiple text files
   */
  async importTextFiles(files: File[]): Promise<TextImportResult[]> {
    const results: TextImportResult[] = [];

    for (const file of files) {
      const result = await this.importTextFile(file);
      results.push(result);
    }

    return results;
  }

  /**
   * Import text with specific format
   */
  async importTextWithFormat(
    file: File,
    format: 'plain' | 'markdown' | 'json'
  ): Promise<TextImportResult> {
    try {
      const text = await this.readFileAsText(file);
      const textId = this.generateTextId(file.name);

      const textContent: TextContent = {
        id: textId,
        text,
        format,
      };

      const examples = this.processor.processText(textContent);
      this.trainingQueue.addBatch(examples);

      return {
        success: true,
        textId,
        examplesGenerated: examples.length,
      };
    } catch (error) {
      return {
        success: false,
        textId: file.name,
        examplesGenerated: 0,
        error: (error as Error).message,
      };
    }
  }

  /**
   * Read file as text
   */
  private readFileAsText(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (e) => {
        resolve(e.target?.result as string);
      };

      reader.onerror = () => {
        reject(new Error('Failed to read file'));
      };

      reader.readAsText(file);
    });
  }

  /**
   * Detect text format
   */
  private detectFormat(filename: string, text: string): 'plain' | 'markdown' | 'json' {
    // Check file extension
    if (filename.endsWith('.md') || filename.endsWith('.markdown')) {
      return 'markdown';
    }

    if (filename.endsWith('.json')) {
      return 'json';
    }

    // Check content
    if (text.trim().startsWith('{') || text.trim().startsWith('[')) {
      try {
        JSON.parse(text);
        return 'json';
      } catch {
        // Not valid JSON
      }
    }

    // Check for markdown patterns
    if (text.includes('# ') || text.includes('## ') || text.includes('```')) {
      return 'markdown';
    }

    return 'plain';
  }

  /**
   * Generate text ID
   */
  private generateTextId(name: string): string {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(7);
    return `text_${timestamp}_${random}`;
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
