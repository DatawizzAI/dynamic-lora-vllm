# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automated Docker image building.

## Workflows

### `build-images.yml` (Main Workflow)
- **Triggers**: Push to `main` branch, manual workflow dispatch
- **Purpose**: Coordinates building Docker images for multiple models
- **Features**:
  - Builds predefined models automatically on push to main
  - Allows manual builds with custom model IDs
  - Sets up date tags and model configurations

### `build-single-image.yml` (Reusable Workflow)
- **Type**: Reusable workflow called by main workflow
- **Purpose**: Builds a single Docker image with specified model
- **Features**:
  - Pre-downloads specified HuggingFace model during build
  - Handles authentication for private models
  - Pushes to GitHub Container Registry with proper tags
  - Uses GPU runner for efficient builds

## Usage

### Automatic Builds
1. Push to `main` branch
2. Workflow automatically builds all predefined models
3. Images are tagged with model name and date
4. Each model builds in parallel as separate jobs

### Manual Builds
1. Go to GitHub Actions tab
2. Select "Build Docker Images" workflow
3. Click "Run workflow"
4. Enter custom model ID and token requirements
5. Run workflow

## Configuration

### Repository Secrets
- `HF_TOKEN`: HuggingFace token (optional, for private models)

### Self-Hosted Runner
- Runner labeled `GPU_Runner` required
- Must have NVIDIA GPU and Docker with NVIDIA Container Toolkit

### Predefined Models
Edit the `models` array in `build-images.yml` to modify the list of automatically built models.

## Image Tags

Images are pushed to `ghcr.io/owner/repo-name` with tags:
- `{model-suffix}-latest`: Latest build for this model
- `{model-suffix}-YYYYMMDD`: Date-specific build
- `custom-latest` / `custom-YYYYMMDD`: Manual builds