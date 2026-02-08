# QWGLM - High-Performance Browser AI

A Next.js website that runs an 8B AI model in the browser using WebGPU acceleration, with screen sharing and voice interaction capabilities.

## Features

- **Browser-based AI inference** using Qwen3-VL-8B model loaded from Hugging Face
- **WebGPU acceleration** for transformer operations (attention, FFN, layer norm)
- **Optimized matrix operations** with tiled matrix multiplication
- **Screen sharing** via Chrome Screen Capture API
- **Voice interaction** with Piper TTS and Parakeet models from Hugging Face
- **Deployed on Cloudflare Pages**

## Performance Architecture

### Hybrid GPU Strategy

This project uses a unique **hybrid approach** combining real GPU acceleration with theoretical simulation:

**Real WebGPU Acceleration** for production workloads:

1. **Tiled Matrix Multiplication** - 16x16 tiles with workgroup shared memory
2. **Optimized Attention** - Fused attention kernels on GPU
3. **Layer Normalization** - GPU-accelerated normalization
4. **Quantization Support** - 4-bit/8-bit weights with on-the-fly dequantization

**Impossible GPU Simulation** for theoretical demonstrations:
- Shows what's theoretically possible with perfect hardware
- 500x speedup over real GPUs (educational)
- Perfect cache hits (100%), zero latency (0.1ns)
- Used for small operations to demonstrate potential

### Performance Optimizations

- **Hybrid Execution** - Mixes real and theoretical GPU for demos
- **KV Cache** - Caches key/value tensors to avoid recomputation
- **Batch Processing** - Processes multiple tokens efficiently
- **Memory Coalescing** - Optimized memory access patterns
- **Pipeline Parallelism** - Overlaps computation and data transfer

See [IMPOSSIBLE_GPU.md](IMPOSSIBLE_GPU.md) for details on the theoretical simulator.

## Model Loading

The application loads models from Hugging Face and CDN:

### Qwen3-VL Model
- **Model ID**: `DavidAU/Qwen3-VL-8B-GLM-4.7-Flash-Heretic-Uncensored-Thinking`
- **Source**: Hugging Face Hub
- **Quantization**: 4-bit (configurable)
- **Format**: Safetensors

### TTS Models
- **Piper TTS**: Loaded from Hugging Face (`rhasspy/piper`)
- **Parakeet TTS**: Loaded from Hugging Face (`nvidia/parakeet-tdt-0.6b-v2`)

## Project Structure

```
qwglm/
├── lib/                    # Core libraries
│   ├── gpu-simulator/     # Ported tiny-gpu
│   ├── model-runtime/     # Qwen3-VL model
│   │   └── loader/       # Hugging Face & CDN loaders
│   ├── audio/             # Voice processing
│   └── screen-capture/    # Screen sharing
├── app/                   # Next.js app directory
├── components/            # React components
└── public/                # Static assets
```

## Development

```bash
npm install
npm run dev
```

## Model Loading Configuration

Models are automatically loaded from Hugging Face on first run. You can configure:

1. **Model ID**: Change in `app/page.tsx` - `loadQwenModelWithProgress`
2. **Quantization**: Set `quantizationBits` (4 or 8)
3. **CDN Fallback**: Configure in `lib/model-runtime/loader/cdn-loader.ts`

## Deployment

**Via Cloudflare Pages Dashboard (Recommended):**
1. Push code to GitHub
2. Connect repository to Cloudflare Pages
3. Configure build settings:
   - Build command: `npm run build`
   - Output directory: `out`
   - Node version: `22.16.0`
4. Deploy automatically on every push

**Via CLI:**
```bash
npm run build
npx wrangler pages deploy out --project-name=qwglm
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

Models are loaded from Hugging Face at runtime (no large files in deployment).

## Browser Requirements

- **WebGPU support** (Chrome 113+, Edge 113+)
- **Screen Capture API** (Chrome/Edge)
- **MediaDevices API** (for microphone)

## License

MIT
