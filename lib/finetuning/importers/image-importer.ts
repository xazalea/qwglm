/**
 * Image Importer
 * Import and process images for training
 */

import { ContentProcessor, type ImageContent } from '../content-processor';
import { TrainingQueue } from '../training-queue';

export interface ImageImportResult {
  success: boolean;
  imageId: string;
  examplesGenerated: number;
  error?: string;
}

export class ImageImporter {
  private processor: ContentProcessor;
  private trainingQueue: TrainingQueue;

  constructor(trainingQueue: TrainingQueue) {
    this.processor = new ContentProcessor();
    this.trainingQueue = trainingQueue;
  }

  /**
   * Import image from file
   */
  async importImageFile(file: File): Promise<ImageImportResult> {
    try {
      const imageData = await this.loadImageFromFile(file);
      const imageId = this.generateImageId(file.name);

      const imageContent: ImageContent = {
        id: imageId,
        imageData,
        caption: file.name,
      };

      const examples = await this.processor.processImage(imageContent);
      this.trainingQueue.addBatch(examples);

      return {
        success: true,
        imageId,
        examplesGenerated: examples.length,
      };
    } catch (error) {
      console.error('Error importing image file:', error);
      return {
        success: false,
        imageId: file.name,
        examplesGenerated: 0,
        error: (error as Error).message,
      };
    }
  }

  /**
   * Import image from URL
   */
  async importImageUrl(url: string, caption?: string): Promise<ImageImportResult> {
    try {
      const imageData = await this.loadImageFromUrl(url);
      const imageId = this.generateImageId(url);

      const imageContent: ImageContent = {
        id: imageId,
        imageData,
        caption: caption || url,
      };

      const examples = await this.processor.processImage(imageContent);
      this.trainingQueue.addBatch(examples);

      return {
        success: true,
        imageId,
        examplesGenerated: examples.length,
      };
    } catch (error) {
      console.error('Error importing image URL:', error);
      return {
        success: false,
        imageId: url,
        examplesGenerated: 0,
        error: (error as Error).message,
      };
    }
  }

  /**
   * Import multiple images
   */
  async importImages(files: File[]): Promise<ImageImportResult[]> {
    const results: ImageImportResult[] = [];

    for (const file of files) {
      const result = await this.importImageFile(file);
      results.push(result);
    }

    return results;
  }

  /**
   * Import image with caption
   */
  async importImageWithCaption(
    file: File,
    caption: string
  ): Promise<ImageImportResult> {
    try {
      const imageData = await this.loadImageFromFile(file);
      const imageId = this.generateImageId(file.name);

      const imageContent: ImageContent = {
        id: imageId,
        imageData,
        caption,
      };

      const examples = await this.processor.processImage(imageContent);
      this.trainingQueue.addBatch(examples);

      return {
        success: true,
        imageId,
        examplesGenerated: examples.length,
      };
    } catch (error) {
      return {
        success: false,
        imageId: file.name,
        examplesGenerated: 0,
        error: (error as Error).message,
      };
    }
  }

  /**
   * Load image from file
   */
  private loadImageFromFile(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (e) => {
        const img = new Image();
        
        img.onload = () => {
          const canvas = document.createElement('canvas');
          canvas.width = img.width;
          canvas.height = img.height;
          
          const ctx = canvas.getContext('2d');
          if (!ctx) {
            reject(new Error('Failed to get canvas context'));
            return;
          }

          ctx.drawImage(img, 0, 0);
          const imageData = ctx.getImageData(0, 0, img.width, img.height);
          resolve(imageData);
        };

        img.onerror = () => {
          reject(new Error('Failed to load image'));
        };

        img.src = e.target?.result as string;
      };

      reader.onerror = () => {
        reject(new Error('Failed to read file'));
      };

      reader.readAsDataURL(file);
    });
  }

  /**
   * Load image from URL
   */
  private loadImageFromUrl(url: string): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';

      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Failed to get canvas context'));
          return;
        }

        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };

      img.onerror = () => {
        reject(new Error('Failed to load image from URL'));
      };

      img.src = url;
    });
  }

  /**
   * Generate image ID
   */
  private generateImageId(name: string): string {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(7);
    return `img_${timestamp}_${random}`;
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
