# Cloudflare Pages Deployment Guide

**Repository**: [https://github.com/xazalea/qwglm](https://github.com/xazalea/qwglm)

## Prerequisites
- Cloudflare account
- GitHub repository connected to Cloudflare Pages
- Node.js 22.x

## Deployment via Cloudflare Pages Dashboard

### Initial Setup

1. Go to [Cloudflare Pages Dashboard](https://dash.cloudflare.com/?to=/:account/pages)
2. Click "Create a project" → "Connect to Git"
3. Select the repository: [`xazalea/qwglm`](https://github.com/xazalea/qwglm)
4. Configure the build settings:

```
Framework preset: Next.js (Static HTML Export)
Build command: npm run build
Build output directory: out
Root directory: /
Deploy command: (leave EMPTY - Cloudflare Pages auto-deploys)
```

**⚠️ IMPORTANT:** Do NOT set a deploy command. Cloudflare Pages automatically deploys the build output. If you see an error about `wrangler deploy`, remove the deploy command from Settings → Builds & deployments.

5. Environment variables (if needed):
```
NODE_VERSION=22.16.0
NPM_VERSION=10.x
```

6. Click "Save and Deploy"

### Subsequent Deployments

Cloudflare Pages automatically deploys on every push to your main/master branch.

## Manual Deployment via CLI

### One-time Setup

```bash
npm install -g wrangler
wrangler login
```

### Deploy

```bash
# Build the project
npm run build

# Deploy to Pages
npx wrangler pages deploy out --project-name=qwglm
```

Or use the shortcut:

```bash
npm run pages:deploy
```

## Build Configuration

The project uses:
- **Next.js Static Export** (`output: 'export'` in `next.config.js`)
- **Output Directory**: `out/`
- **Node Version**: 22.16.0 (specified in `.node-version`)

## Important Notes

### WebGPU Support
- WebGPU requires a secure context (HTTPS)
- Cloudflare Pages provides automatic HTTPS
- Custom domains must have SSL enabled

### Large Model Files
- Model files are loaded from Hugging Face CDN at runtime
- No large files are included in the build
- This keeps deployment size small and fast

### Browser Requirements
Users need:
- Chrome 113+ or Edge 113+ (for WebGPU)
- Modern browser with JavaScript enabled
- Minimum 4GB RAM recommended for 8B model

## Troubleshooting

### Build Fails

**Error**: "It looks like you've run a Workers-specific command"
- **Solution**: Use `npx wrangler pages deploy` not `npx wrangler deploy`

**Error**: "Module not found"
- **Solution**: Run `npm install` to ensure all dependencies are installed

**Error**: "TypeScript errors"
- **Solution**: These are usually type-only errors. The build will complete. To fix, ensure `@types/react` is installed.

### Runtime Errors

**WebGPU Not Available**
- Ensure HTTPS is enabled
- Check browser compatibility
- Verify user is on Chrome/Edge 113+

**Model Loading Fails**
- Check Hugging Face model URL is accessible
- Verify CDN fallbacks are working
- Check browser console for CORS errors

## Custom Domain

1. Go to Pages project → Custom domains
2. Click "Set up a custom domain"
3. Enter your domain (e.g., `qwglm.yourdomain.com`)
4. Update DNS records as instructed
5. Wait for SSL certificate provisioning (automatic)

## Environment Variables

Add these in Cloudflare Pages settings if needed:

```
# Optional: Custom model URL
NEXT_PUBLIC_MODEL_URL=https://your-cdn.com/model

# Optional: Analytics
NEXT_PUBLIC_ANALYTICS_ID=your-id
```

## Performance Optimization

Cloudflare Pages automatically provides:
- ✅ Global CDN (distributed across 300+ cities)
- ✅ Automatic HTTPS
- ✅ HTTP/3 support
- ✅ Brotli compression
- ✅ Smart caching

## Monitoring

View deployment logs and analytics:
1. Go to Pages project dashboard
2. Click on any deployment
3. View build logs, function invocations, and analytics

## Rollback

To rollback to a previous deployment:
1. Go to project → Deployments
2. Find the previous working deployment
3. Click "···" → "Rollback to this deployment"
