/**
 * Chat Logger
 * Logs chat interactions as training pairs
 */

import { TrainingExample, TrainingPriority } from './training-queue';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export class ChatLogger {
  private conversationHistory: Message[] = [];
  private maxHistorySize: number = 1000;

  /**
   * Log a message to conversation history
   */
  logMessage(message: Message): void {
    this.conversationHistory.push(message);

    // Keep history within limits
    if (this.conversationHistory.length > this.maxHistorySize) {
      this.conversationHistory.shift();
    }
  }

  /**
   * Extract training pairs from conversation
   */
  extractTrainingPairs(): TrainingExample[] {
    const pairs: TrainingExample[] = [];
    
    for (let i = 0; i < this.conversationHistory.length - 1; i++) {
      const current = this.conversationHistory[i];
      const next = this.conversationHistory[i + 1];

      // Look for user -> assistant pairs
      if (current.role === 'user' && next.role === 'assistant') {
        const example: TrainingExample = {
          id: `chat_${current.id}_${next.id}`,
          input: current.content,
          output: next.content,
          priority: this.calculatePriority(next.timestamp),
          timestamp: next.timestamp.getTime(),
          metadata: {
            source: 'chat',
            contentType: 'text',
          },
        };
        pairs.push(example);
      }
    }

    return pairs;
  }

  /**
   * Extract training pairs with context (multi-turn)
   */
  extractContextualPairs(contextWindow: number = 3): TrainingExample[] {
    const pairs: TrainingExample[] = [];
    
    for (let i = 0; i < this.conversationHistory.length - 1; i++) {
      const current = this.conversationHistory[i];
      const next = this.conversationHistory[i + 1];

      if (current.role === 'user' && next.role === 'assistant') {
        // Include previous context
        const contextStart = Math.max(0, i - contextWindow);
        const context = this.conversationHistory
          .slice(contextStart, i)
          .map((m) => `${m.role}: ${m.content}`)
          .join('\n');

        const input = context 
          ? `${context}\nuser: ${current.content}`
          : current.content;

        const example: TrainingExample = {
          id: `chat_ctx_${current.id}_${next.id}`,
          input,
          output: next.content,
          priority: this.calculatePriority(next.timestamp),
          timestamp: next.timestamp.getTime(),
          metadata: {
            source: 'chat',
            contentType: 'text',
          },
        };
        pairs.push(example);
      }
    }

    return pairs;
  }

  /**
   * Get recent conversations
   */
  getRecentConversations(count: number): Message[] {
    return this.conversationHistory.slice(-count);
  }

  /**
   * Calculate priority based on recency
   */
  private calculatePriority(timestamp: Date): TrainingPriority {
    const age = Date.now() - timestamp.getTime();
    const minutes = age / (1000 * 60);

    if (minutes < 5) {
      return TrainingPriority.HIGH;
    } else if (minutes < 30) {
      return TrainingPriority.MEDIUM;
    } else {
      return TrainingPriority.LOW;
    }
  }

  /**
   * Clear conversation history
   */
  clear(): void {
    this.conversationHistory = [];
  }

  /**
   * Get conversation statistics
   */
  getStats(): {
    totalMessages: number;
    userMessages: number;
    assistantMessages: number;
    potentialPairs: number;
  } {
    const userMessages = this.conversationHistory.filter((m) => m.role === 'user').length;
    const assistantMessages = this.conversationHistory.filter((m) => m.role === 'assistant').length;
    const potentialPairs = Math.min(userMessages, assistantMessages);

    return {
      totalMessages: this.conversationHistory.length,
      userMessages,
      assistantMessages,
      potentialPairs,
    };
  }
}
