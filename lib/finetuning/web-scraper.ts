/**
 * Web Scraper
 * Scrapes and processes websites for training data
 */

import { TrainingExample, TrainingPriority } from './training-queue';

export interface ScrapedContent {
  url: string;
  title: string;
  content: string;
  headings: string[];
  metadata: {
    author?: string;
    publishDate?: string;
    description?: string;
  };
}

export class WebScraper {
  /**
   * Scrape website and extract main content
   */
  async scrapeWebsite(url: string): Promise<ScrapedContent> {
    try {
      const response = await fetch(url);
      const html = await response.text();
      
      return this.parseHTML(html, url);
    } catch (error) {
      console.error('Error scraping website:', error);
      throw error;
    }
  }

  /**
   * Parse HTML and extract content
   */
  private parseHTML(html: string, url: string): ScrapedContent {
    // Create a DOM parser
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');

    // Extract title
    const title = doc.querySelector('title')?.textContent || 'Untitled';

    // Extract meta description
    const description = doc.querySelector('meta[name="description"]')?.getAttribute('content') || '';

    // Extract author
    const author = doc.querySelector('meta[name="author"]')?.getAttribute('content') || '';

    // Extract publish date
    const publishDate = doc.querySelector('meta[property="article:published_time"]')?.getAttribute('content') || '';

    // Extract main content
    const content = this.extractMainContent(doc);

    // Extract headings
    const headings = this.extractHeadings(doc);

    return {
      url,
      title,
      content,
      headings,
      metadata: {
        author,
        publishDate,
        description,
      },
    };
  }

  /**
   * Extract main content (article text)
   */
  private extractMainContent(doc: Document): string {
    // Try common article selectors
    const selectors = [
      'article',
      'main',
      '[role="main"]',
      '.article-content',
      '.post-content',
      '.entry-content',
      '#content',
    ];

    for (const selector of selectors) {
      const element = doc.querySelector(selector);
      if (element) {
        return this.cleanText(element.textContent || '');
      }
    }

    // Fallback: extract all paragraphs
    const paragraphs = Array.from(doc.querySelectorAll('p'))
      .map((p) => p.textContent || '')
      .filter((text) => text.trim().length > 50);

    return paragraphs.join('\n\n');
  }

  /**
   * Extract headings
   */
  private extractHeadings(doc: Document): string[] {
    const headings: string[] = [];
    
    for (let i = 1; i <= 6; i++) {
      const elements = doc.querySelectorAll(`h${i}`);
      elements.forEach((el) => {
        const text = el.textContent?.trim();
        if (text) {
          headings.push(text);
        }
      });
    }

    return headings;
  }

  /**
   * Clean text (remove extra whitespace, etc.)
   */
  private cleanText(text: string): string {
    return text
      .replace(/\s+/g, ' ')
      .replace(/\n\s*\n/g, '\n\n')
      .trim();
  }

  /**
   * Generate training examples from scraped content
   */
  generateTrainingExamples(scraped: ScrapedContent): TrainingExample[] {
    const examples: TrainingExample[] = [];
    const timestamp = Date.now();

    // Create summary example
    examples.push({
      id: `web_summary_${this.hashString(scraped.url)}`,
      input: `Summarize this article: ${scraped.title}`,
      output: scraped.metadata.description || this.generateSummary(scraped.content),
      priority: TrainingPriority.MEDIUM,
      timestamp,
      metadata: {
        source: 'import',
        contentType: 'web',
        url: scraped.url,
      },
    });

    // Create Q&A pairs from headings
    scraped.headings.forEach((heading, idx) => {
      if (heading.endsWith('?')) {
        // Heading is a question, extract answer from content
        const answer = this.extractAnswerForQuestion(heading, scraped.content);
        if (answer) {
          examples.push({
            id: `web_qa_${this.hashString(scraped.url)}_${idx}`,
            input: heading,
            output: answer,
            priority: TrainingPriority.MEDIUM,
            timestamp,
            metadata: {
              source: 'import',
              contentType: 'web',
              url: scraped.url,
            },
          });
        }
      } else {
        // Generate question from heading
        const question = `What is ${heading.toLowerCase()}?`;
        const answer = this.extractAnswerForQuestion(question, scraped.content);
        if (answer) {
          examples.push({
            id: `web_qa_${this.hashString(scraped.url)}_${idx}`,
            input: question,
            output: answer,
            priority: TrainingPriority.MEDIUM,
            timestamp,
            metadata: {
              source: 'import',
              contentType: 'web',
              url: scraped.url,
            },
          });
        }
      }
    });

    // Create content chunking examples
    const chunks = this.chunkContent(scraped.content, 500);
    chunks.forEach((chunk, idx) => {
      examples.push({
        id: `web_chunk_${this.hashString(scraped.url)}_${idx}`,
        input: `Explain: ${scraped.title}`,
        output: chunk,
        priority: TrainingPriority.MEDIUM,
        timestamp,
        metadata: {
          source: 'import',
          contentType: 'web',
          url: scraped.url,
        },
      });
    });

    return examples;
  }

  /**
   * Generate a simple summary from content
   */
  private generateSummary(content: string, maxLength: number = 200): string {
    const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 20);
    const summary = sentences.slice(0, 3).join('. ');
    return summary.length > maxLength 
      ? summary.substring(0, maxLength) + '...'
      : summary;
  }

  /**
   * Extract answer for a question from content
   */
  private extractAnswerForQuestion(question: string, content: string): string | null {
    const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 20);
    
    // Simple keyword matching
    const keywords = question.toLowerCase().split(/\s+/).filter((w) => w.length > 3);
    
    for (const sentence of sentences) {
      const sentenceLower = sentence.toLowerCase();
      const matchCount = keywords.filter((kw) => sentenceLower.includes(kw)).length;
      
      if (matchCount >= Math.min(keywords.length / 2, 3)) {
        return sentence.trim();
      }
    }

    return null;
  }

  /**
   * Chunk content into smaller pieces
   */
  private chunkContent(content: string, maxChunkSize: number): string[] {
    const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 20);
    const chunks: string[] = [];
    let currentChunk = '';

    for (const sentence of sentences) {
      if (currentChunk.length + sentence.length > maxChunkSize) {
        if (currentChunk) {
          chunks.push(currentChunk.trim());
        }
        currentChunk = sentence;
      } else {
        currentChunk += (currentChunk ? '. ' : '') + sentence;
      }
    }

    if (currentChunk) {
      chunks.push(currentChunk.trim());
    }

    return chunks.slice(0, 5); // Limit to 5 chunks per page
  }

  /**
   * Hash string to generate ID
   */
  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(36);
  }
}
