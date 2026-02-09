/**
 * Content Processor
 * Processes images and text imports for training data
 */

import { TrainingExample, TrainingPriority } from './training-queue';
import { visionEncoder } from '../model-runtime/vision/vision-encoder';

export interface ImageContent {
  id: string;
  imageData: ImageData;
  caption?: string;
  metadata?: Record<string, any>;
}

export interface TextContent {
  id: string;
  text: string;
  format: 'plain' | 'markdown' | 'json';
  metadata?: Record<string, any>;
}

export class ContentProcessor {
  /**
   * Process image and generate training examples
   */
  async processImage(image: ImageContent): Promise<TrainingExample[]> {
    const examples: TrainingExample[] = [];
    const timestamp = Date.now();

    // Generate image description using vision encoder
    const description = await this.generateImageDescription(image.imageData);

    // Create image description example
    examples.push({
      id: `image_desc_${image.id}`,
      input: 'Describe this image',
      output: description,
      priority: TrainingPriority.MEDIUM,
      timestamp,
      metadata: {
        source: 'import',
        contentType: 'image',
      },
    });

    // If caption provided, create caption example
    if (image.caption) {
      examples.push({
        id: `image_caption_${image.id}`,
        input: 'What is in this image?',
        output: image.caption,
        priority: TrainingPriority.MEDIUM,
        timestamp,
        metadata: {
          source: 'import',
          contentType: 'image',
        },
      });
    }

    return examples;
  }

  /**
   * Process text content and generate training examples
   */
  processText(content: TextContent): TrainingExample[] {
    const examples: TrainingExample[] = [];
    const timestamp = Date.now();

    switch (content.format) {
      case 'plain':
        examples.push(...this.processPlainText(content.text, content.id, timestamp));
        break;
      case 'markdown':
        examples.push(...this.processMarkdown(content.text, content.id, timestamp));
        break;
      case 'json':
        examples.push(...this.processJSON(content.text, content.id, timestamp));
        break;
    }

    return examples;
  }

  /**
   * Generate image description using vision encoder
   */
  private async generateImageDescription(imageData: ImageData): Promise<string> {
    // Use vision encoder to extract features
    const config = {
      imageSize: 224,
      patchSize: 14,
      hiddenSize: 768,
      numLayers: 12,
      numHeads: 12,
    };

    try {
      // Extract vision features
      const features = visionEncoder(imageData, config, {} as any);
      
      // Generate simple description based on features
      // In a real implementation, this would use a caption model
      return 'An image with various visual elements';
    } catch (error) {
      console.error('Error generating image description:', error);
      return 'Unable to describe image';
    }
  }

  /**
   * Process plain text
   */
  private processPlainText(text: string, id: string, timestamp: number): TrainingExample[] {
    const examples: TrainingExample[] = [];
    
    // Split into paragraphs
    const paragraphs = text.split(/\n\n+/).filter((p) => p.trim().length > 50);

    // Create summarization examples
    paragraphs.forEach((para, idx) => {
      examples.push({
        id: `text_para_${id}_${idx}`,
        input: 'Summarize this text',
        output: para.trim(),
        priority: TrainingPriority.MEDIUM,
        timestamp,
        metadata: {
          source: 'import',
          contentType: 'text',
        },
      });
    });

    // Create full text example
    if (text.length < 2000) {
      examples.push({
        id: `text_full_${id}`,
        input: 'What is this text about?',
        output: text.trim(),
        priority: TrainingPriority.MEDIUM,
        timestamp,
        metadata: {
          source: 'import',
          contentType: 'text',
        },
      });
    }

    return examples;
  }

  /**
   * Process markdown
   */
  private processMarkdown(markdown: string, id: string, timestamp: number): TrainingExample[] {
    const examples: TrainingExample[] = [];

    // Extract headings and content
    const headingRegex = /^#+\s+(.+)$/gm;
    const headings: { level: number; text: string; content: string }[] = [];
    
    let match;
    let lastIndex = 0;

    while ((match = headingRegex.exec(markdown)) !== null) {
      if (headings.length > 0) {
        headings[headings.length - 1].content = markdown.substring(
          lastIndex,
          match.index
        ).trim();
      }

      const level = match[0].indexOf(' ');
      headings.push({
        level,
        text: match[1],
        content: '',
      });
      lastIndex = match.index + match[0].length;
    }

    // Set content for last heading
    if (headings.length > 0) {
      headings[headings.length - 1].content = markdown.substring(lastIndex).trim();
    }

    // Create Q&A pairs from headings
    headings.forEach((heading, idx) => {
      if (heading.content.length > 50) {
        examples.push({
          id: `md_section_${id}_${idx}`,
          input: `Explain: ${heading.text}`,
          output: heading.content,
          priority: TrainingPriority.MEDIUM,
          timestamp,
          metadata: {
            source: 'import',
            contentType: 'text',
          },
        });
      }
    });

    return examples;
  }

  /**
   * Process JSON
   */
  private processJSON(jsonText: string, id: string, timestamp: number): TrainingExample[] {
    const examples: TrainingExample[] = [];

    try {
      const data = JSON.parse(jsonText);

      // If it's an array of objects, create examples from each
      if (Array.isArray(data)) {
        data.forEach((item, idx) => {
          if (typeof item === 'object' && item !== null) {
            examples.push({
              id: `json_item_${id}_${idx}`,
              input: 'Describe this data',
              output: JSON.stringify(item, null, 2),
              priority: TrainingPriority.MEDIUM,
              timestamp,
              metadata: {
                source: 'import',
                contentType: 'text',
              },
            });
          }
        });
      } else {
        // Single object
        examples.push({
          id: `json_obj_${id}`,
          input: 'What is in this data?',
          output: JSON.stringify(data, null, 2),
          priority: TrainingPriority.MEDIUM,
          timestamp,
          metadata: {
            source: 'import',
            contentType: 'text',
          },
        });
      }
    } catch (error) {
      console.error('Error parsing JSON:', error);
    }

    return examples;
  }

  /**
   * Process batch of images
   */
  async processBatchImages(images: ImageContent[]): Promise<TrainingExample[]> {
    const allExamples: TrainingExample[] = [];
    
    for (const image of images) {
      const examples = await this.processImage(image);
      allExamples.push(...examples);
    }

    return allExamples;
  }

  /**
   * Process batch of text content
   */
  processBatchText(texts: TextContent[]): TrainingExample[] {
    const allExamples: TrainingExample[] = [];
    
    for (const text of texts) {
      const examples = this.processText(text);
      allExamples.push(...examples);
    }

    return allExamples;
  }
}
