/**
 * Piper TTS Model Loader
 * Loads Piper models from CDN or Hugging Face
 */

export interface PiperModelConfig {
  modelUrl?: string;
  modelPath?: string;
  configUrl?: string;
  useHuggingFace?: boolean;
  huggingFaceModelId?: string;
}

/**
 * Load Piper model from URL
 */
export async function loadPiperModel(config: PiperModelConfig): Promise<{
  model: ArrayBuffer;
  config: any;
}> {
  let modelUrl: string;
  let configUrl: string;

  if (config.useHuggingFace && config.huggingFaceModelId) {
    // Load from Hugging Face
    const modelId = config.huggingFaceModelId;
    modelUrl = `https://huggingface.co/${modelId}/resolve/main/model.onnx`;
    configUrl = `https://huggingface.co/${modelId}/raw/main/config.json`;
  } else if (config.modelUrl) {
    // Load from CDN
    modelUrl = config.modelUrl;
    configUrl = config.configUrl || modelUrl.replace('.onnx', '.json');
  } else {
    throw new Error('No model URL or Hugging Face model ID provided');
  }

  console.log(`Loading Piper model from ${modelUrl}...`);

  const [modelResponse, configResponse] = await Promise.all([
    fetch(modelUrl),
    fetch(configUrl),
  ]);

  if (!modelResponse.ok) {
    throw new Error(`Failed to load Piper model: ${modelResponse.statusText}`);
  }

  const model = await modelResponse.arrayBuffer();
  const modelConfig = configResponse.ok ? await configResponse.json() : {};

  return { model, config: modelConfig };
}
