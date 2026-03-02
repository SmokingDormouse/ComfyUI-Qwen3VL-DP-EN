### 1. Node Development Standards
- All nodes must include a complete class definition, including `INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION`, and `CATEGORY`.
- Node categorization must uniformly use: `CATEGORY = "🍭DaPao-Qwen3VL"`
- Nodes must be registered at the end of the file in both `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.

### 2. Image Processing Standards
- Input image tensor shape: `[B, H, W, C]` (batch, height, width, channels).
- Input mask tensor shape: `[B, H, W]`.
- Ensure image value range is between 0 and 1 (`float32`).
- Use `pil2tensor()` and `tensor2pil()` for format conversion.

### 3. Path and File Management Standards (Core Refactor – Fixing Model Loading Bugs)
- Model paths: Absolute paths are strictly forbidden. All model loading and downloading paths (Checkpoint, VAE, LoRA) must be dynamically obtained via ComfyUI’s `folder_paths` module.
- Resource references: Internal plugin files (configs, icons) must be referenced using relative paths.
- Download logic: Target paths must not be hardcoded. File existence must be verified before downloading.
