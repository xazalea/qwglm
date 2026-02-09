/**
 * Import Interface Component
 * UI for importing websites, images, and text
 */

'use client';

import { useState, useRef } from 'react';

export interface ImportResult {
  type: 'website' | 'image' | 'text';
  success: boolean;
  id: string;
  examplesGenerated: number;
  error?: string;
}

interface ImportInterfaceProps {
  onImportWebsite?: (url: string) => Promise<ImportResult>;
  onImportImage?: (file: File) => Promise<ImportResult>;
  onImportText?: (text: string, format: 'plain' | 'markdown' | 'json') => Promise<ImportResult>;
}

export default function ImportInterface({
  onImportWebsite,
  onImportImage,
  onImportText,
}: ImportInterfaceProps) {
  const [activeTab, setActiveTab] = useState<'website' | 'image' | 'text'>('website');
  const [websiteUrl, setWebsiteUrl] = useState('');
  const [textContent, setTextContent] = useState('');
  const [textFormat, setTextFormat] = useState<'plain' | 'markdown' | 'json'>('plain');
  const [isImporting, setIsImporting] = useState(false);
  const [importResults, setImportResults] = useState<ImportResult[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImportWebsite = async () => {
    if (!websiteUrl || !onImportWebsite) return;

    setIsImporting(true);
    try {
      const result = await onImportWebsite(websiteUrl);
      setImportResults([result, ...importResults]);
      
      if (result.success) {
        setWebsiteUrl('');
      }
    } catch (error) {
      console.error('Error importing website:', error);
    } finally {
      setIsImporting(false);
    }
  };

  const handleImportImage = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0 || !onImportImage) return;

    setIsImporting(true);
    try {
      for (const file of Array.from(files)) {
        const result = await onImportImage(file);
        setImportResults([result, ...importResults]);
      }
    } catch (error) {
      console.error('Error importing images:', error);
    } finally {
      setIsImporting(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleImportText = async () => {
    if (!textContent || !onImportText) return;

    setIsImporting(true);
    try {
      const result = await onImportText(textContent, textFormat);
      setImportResults([result, ...importResults]);
      
      if (result.success) {
        setTextContent('');
      }
    } catch (error) {
      console.error('Error importing text:', error);
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950/50 p-4">
      <h2 className="text-sm font-medium mb-4">Import Training Data</h2>

      {/* Tabs */}
      <div className="flex gap-2 mb-4 border-b border-neutral-800">
        {(['website', 'image', 'text'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-3 py-2 text-sm capitalize ${
              activeTab === tab
                ? 'border-b-2 border-blue-500 text-blue-400'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Website Import */}
      {activeTab === 'website' && (
        <div className="space-y-3">
          <input
            type="url"
            value={websiteUrl}
            onChange={(e) => setWebsiteUrl(e.target.value)}
            placeholder="https://example.com"
            className="w-full px-3 py-2 bg-neutral-900 border border-neutral-700 rounded text-sm focus:outline-none focus:border-blue-500"
            disabled={isImporting}
          />
          
          <button
            onClick={handleImportWebsite}
            disabled={isImporting || !websiteUrl}
            className="w-full px-4 py-2 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isImporting ? 'Importing...' : 'Import Website'}
          </button>
        </div>
      )}

      {/* Image Import */}
      {activeTab === 'image' && (
        <div className="space-y-3">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            onChange={handleImportImage}
            disabled={isImporting}
            className="w-full px-3 py-2 bg-neutral-900 border border-neutral-700 rounded text-sm focus:outline-none file:mr-4 file:py-1 file:px-3 file:rounded file:border-0 file:bg-neutral-800 file:text-neutral-300 hover:file:bg-neutral-700"
          />
          
          <p className="text-xs text-neutral-500">
            Select one or more images to import. The AI will learn from them.
          </p>
        </div>
      )}

      {/* Text Import */}
      {activeTab === 'text' && (
        <div className="space-y-3">
          <select
            value={textFormat}
            onChange={(e) => setTextFormat(e.target.value as any)}
            className="w-full px-3 py-2 bg-neutral-900 border border-neutral-700 rounded text-sm focus:outline-none focus:border-blue-500"
            disabled={isImporting}
          >
            <option value="plain">Plain Text</option>
            <option value="markdown">Markdown</option>
            <option value="json">JSON</option>
          </select>

          <textarea
            value={textContent}
            onChange={(e) => setTextContent(e.target.value)}
            placeholder="Paste your text here..."
            rows={6}
            className="w-full px-3 py-2 bg-neutral-900 border border-neutral-700 rounded text-sm focus:outline-none focus:border-blue-500 resize-none"
            disabled={isImporting}
          />
          
          <button
            onClick={handleImportText}
            disabled={isImporting || !textContent}
            className="w-full px-4 py-2 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isImporting ? 'Importing...' : 'Import Text'}
          </button>
        </div>
      )}

      {/* Import Results */}
      {importResults.length > 0 && (
        <div className="mt-4 pt-4 border-t border-neutral-800">
          <h3 className="text-xs font-medium text-neutral-400 mb-2">Recent Imports</h3>
          
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {importResults.slice(0, 5).map((result, idx) => (
              <div
                key={idx}
                className={`p-2 rounded text-xs ${
                  result.success
                    ? 'bg-green-500/10 border border-green-500/20'
                    : 'bg-red-500/10 border border-red-500/20'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium capitalize">{result.type}</span>
                  <span className={result.success ? 'text-green-400' : 'text-red-400'}>
                    {result.success ? '✓' : '✗'}
                  </span>
                </div>
                
                {result.success ? (
                  <div className="text-neutral-400 mt-1">
                    Generated {result.examplesGenerated} training examples
                  </div>
                ) : (
                  <div className="text-red-400 mt-1">{result.error}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
