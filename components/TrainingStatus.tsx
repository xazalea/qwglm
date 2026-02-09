/**
 * Training Status Component
 * Displays real-time training progress and metrics
 */

'use client';

import { useEffect, useState } from 'react';
import type { TrainingMetrics } from '@/lib/finetuning/training-engine';

interface TrainingStatusProps {
  isTraining: boolean;
  metrics?: TrainingMetrics;
  queueSize: number;
  onToggleTraining?: () => void;
  fastLearningStats?: {
    oneShot: number;
    recalled: number;
    traditional: number;
    fastRatio: number;
  };
}

export default function TrainingStatus({
  isTraining,
  metrics,
  queueSize,
  onToggleTraining,
  fastLearningStats,
}: TrainingStatusProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950/50 p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isTraining ? 'bg-green-500 animate-pulse' : 'bg-neutral-600'}`} />
          <h3 className="text-sm font-medium">Real-time Fine-tuning</h3>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs text-neutral-400 hover:text-neutral-200"
          >
            {expanded ? 'Collapse' : 'Expand'}
          </button>
          
          {onToggleTraining && (
            <button
              onClick={onToggleTraining}
              className={`px-2 py-1 text-xs rounded ${
                isTraining 
                  ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30' 
                  : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
              }`}
            >
              {isTraining ? 'Pause' : 'Resume'}
            </button>
          )}
        </div>
      </div>

      {/* Status Info */}
      <div className="grid grid-cols-3 gap-3 text-xs">
        <div>
          <div className="text-neutral-500">Queue</div>
          <div className="font-medium">{queueSize} examples</div>
        </div>
        
        {metrics && (
          <>
            <div>
              <div className="text-neutral-500">Step</div>
              <div className="font-medium">{metrics.step}</div>
            </div>
            
            <div>
              <div className="text-neutral-500">Loss</div>
              <div className="font-medium">{metrics.loss.toFixed(4)}</div>
            </div>
          </>
        )}
      </div>

      {/* Expanded Details */}
      {expanded && metrics && (
        <div className="mt-4 pt-4 border-t border-neutral-800 space-y-2 text-xs">
          <div className="flex justify-between">
            <span className="text-neutral-500">Learning Rate:</span>
            <span className="font-medium">{metrics.learningRate.toFixed(6)}</span>
          </div>
          
          <div className="flex justify-between">
            <span className="text-neutral-500">Gradient Norm:</span>
            <span className="font-medium">{metrics.gradNorm.toFixed(4)}</span>
          </div>
          
          <div className="flex justify-between">
            <span className="text-neutral-500">Examples Processed:</span>
            <span className="font-medium">{metrics.examplesProcessed}</span>
          </div>
          
          <div className="flex justify-between">
            <span className="text-neutral-500">Last Update:</span>
            <span className="font-medium">
              {new Date(metrics.timestamp).toLocaleTimeString()}
            </span>
          </div>

          {/* Fast Learning Stats */}
          {fastLearningStats && (
            <>
              <div className="mt-3 pt-3 border-t border-neutral-700">
                <div className="text-neutral-400 mb-2 font-medium">🧠 Fast Learning</div>
                <div className="flex justify-between">
                  <span className="text-neutral-500">One-shot:</span>
                  <span className="font-medium text-green-400">{fastLearningStats.oneShot}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-500">Recalled:</span>
                  <span className="font-medium text-blue-400">{fastLearningStats.recalled}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-500">Traditional:</span>
                  <span className="font-medium text-neutral-400">{fastLearningStats.traditional}</span>
                </div>
                <div className="flex justify-between mt-1 pt-1 border-t border-neutral-800">
                  <span className="text-neutral-500">Fast Ratio:</span>
                  <span className="font-medium text-purple-400">
                    {(fastLearningStats.fastRatio * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Progress Indicator */}
      {isTraining && (
        <div className="mt-3">
          <div className="h-1 bg-neutral-800 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-blue-500 to-purple-500 animate-pulse" style={{ width: '100%' }} />
          </div>
        </div>
      )}
    </div>
  );
}
