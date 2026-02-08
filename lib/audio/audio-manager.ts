/**
 * Audio Manager
 * Unified interface for microphone input, TTS playback, and real-time streaming
 */

import { PiperTTS, type PiperConfig } from './piper/piper';
import { ParakeetTTS, type ParakeetConfig } from './parakeet/parakeet';

export type TTSProvider = 'piper' | 'parakeet';

export interface AudioManagerConfig {
  ttsProvider: TTSProvider;
  piperConfig?: PiperConfig;
  parakeetConfig?: ParakeetConfig;
  sampleRate?: number;
}

export class AudioManager {
  private piper: PiperTTS | null = null;
  private parakeet: ParakeetTTS | null = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private isRecording = false;
  private config: AudioManagerConfig;
  private currentProvider: TTSProvider;

  constructor(config: AudioManagerConfig) {
    this.config = config;
    this.currentProvider = config.ttsProvider;

    // Initialize audio context
    if (typeof window !== 'undefined') {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    }

    // Initialize TTS engines
    if (config.piperConfig) {
      this.piper = new PiperTTS(config.piperConfig);
    }
    if (config.parakeetConfig) {
      this.parakeet = new ParakeetTTS(config.parakeetConfig);
    }
  }

  /**
   * Initialize audio manager
   */
  async initialize(): Promise<boolean> {
    try {
      if (this.piper && this.currentProvider === 'piper') {
        await this.piper.initialize();
      }
      if (this.parakeet && this.currentProvider === 'parakeet') {
        await this.parakeet.initialize();
      }
      return true;
    } catch (error) {
      console.error('Failed to initialize audio manager:', error);
      return false;
    }
  }

  /**
   * Start microphone input
   */
  async startMicrophone(): Promise<boolean> {
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: this.config.sampleRate || 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      if (this.audioContext) {
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        // Audio processing can be added here
      }

      return true;
    } catch (error) {
      console.error('Failed to start microphone:', error);
      return false;
    }
  }

  /**
   * Stop microphone input
   */
  stopMicrophone(): void {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
  }

  /**
   * Start recording audio
   */
  async startRecording(): Promise<boolean> {
    if (!this.mediaStream) {
      const started = await this.startMicrophone();
      if (!started) return false;
    }

    if (!this.mediaStream) {
      return false;
    }

    try {
      this.mediaRecorder = new MediaRecorder(this.mediaStream);
      this.isRecording = true;
      return true;
    } catch (error) {
      console.error('Failed to start recording:', error);
      return false;
    }
  }

  /**
   * Stop recording and get audio data
   */
  async stopRecording(): Promise<Blob | null> {
    if (!this.mediaRecorder || !this.isRecording) {
      return null;
    }

    return new Promise((resolve) => {
      const chunks: Blob[] = [];
      this.mediaRecorder!.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      this.mediaRecorder!.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        this.isRecording = false;
        resolve(blob);
      };

      this.mediaRecorder!.stop();
    });
  }

  /**
   * Synthesize text to speech
   */
  async synthesize(text: string): Promise<Float32Array> {
    if (this.currentProvider === 'piper' && this.piper) {
      return await this.piper.synthesize(text);
    } else if (this.currentProvider === 'parakeet' && this.parakeet) {
      return await this.parakeet.synthesize(text);
    }
    throw new Error('TTS provider not initialized');
  }

  /**
   * Play audio buffer
   */
  async playAudio(audioData: Float32Array, sampleRate: number = 22050): Promise<void> {
    if (!this.audioContext) {
      throw new Error('Audio context not initialized');
    }

    const buffer = this.audioContext.createBuffer(1, audioData.length, sampleRate);
    buffer.getChannelData(0).set(audioData);

    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);
    source.start();

    return new Promise((resolve) => {
      source.onended = () => resolve();
    });
  }

  /**
   * Synthesize and play text
   */
  async speak(text: string): Promise<void> {
    const audioData = await this.synthesize(text);
    const sampleRate = this.currentProvider === 'piper'
      ? (this.config.piperConfig?.sampleRate || 22050)
      : (this.config.parakeetConfig?.sampleRate || 22050);
    await this.playAudio(audioData, sampleRate);
  }

  /**
   * Stream audio (for real-time TTS)
   */
  async streamAudio(
    audioData: Float32Array,
    sampleRate: number,
    onChunk?: (chunk: Float32Array) => void
  ): Promise<void> {
    const chunkSize = sampleRate * 0.1; // 100ms chunks
    for (let i = 0; i < audioData.length; i += chunkSize) {
      const chunk = audioData.slice(i, i + chunkSize);
      if (onChunk) {
        onChunk(chunk);
      }
      await this.playAudio(chunk, sampleRate);
    }
  }

  /**
   * Set TTS provider
   */
  setProvider(provider: TTSProvider): void {
    this.currentProvider = provider;
  }

  /**
   * Get current provider
   */
  getProvider(): TTSProvider {
    return this.currentProvider;
  }

  /**
   * Check if recording
   */
  isRecordingActive(): boolean {
    return this.isRecording;
  }

  /**
   * Cleanup
   */
  cleanup(): void {
    this.stopMicrophone();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}
