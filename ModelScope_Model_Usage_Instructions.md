# ModelScope Model Usage Instructions

## 📋 Overview

This project now supports downloading community models from ModelScope. ModelScope is a Chinese-friendly model hosting platform with typically faster access speeds than HuggingFace.

## 🆕 New Models

### Huihui-Qwen3-VL-4B-Instruct-Abliterated

- **Model Source**: ModelScope
- **Repository URL**: https://modelscope.cn/models/fireicewolf/Huihui-Qwen3-VL-4B-Instruct-abliterated
- **Base Model**: Qwen3-VL-4B-Instruct
- **Feature**: Safety filters removed (abliterated)

#### VRAM Requirements
- **Full Precision (FP16)**: 6GB
- **8-bit Quantization**: 3.5GB
- **4-bit Quantization**: 2GB

#### ⚠️ Important Warning
This model has had its safety filtering mechanism removed and may generate sensitive or inappropriate content. Please only use in the following scenarios:
- Research and academic purposes
- Controlled testing environments
- Scenarios where you understand the risks and can take responsibility

**This model is not recommended for production environments or public-facing applications.**

## 📦 Installing Dependencies

### Automatic Installation (Recommended)

When using a ModelScope model for the first time, the system will automatically prompt you to install dependencies:

```bash
pip install modelscope
```

### Manual Installation

If you need to install in advance, run:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Qwen3VL-DP
pip install -r requirements.txt
```

`requirements.txt` already includes the `modelscope` dependency.

## 🚀 Usage

### 1. Select Model in Node

In the model dropdown list of any Qwen3VL node (main node, batch captioning, comparison captioning, etc.), select:

```
Huihui-Qwen3-VL-4B-Instruct-Abliterated
```

### 2. Automatic Download

On first use, the model will automatically download from ModelScope to:

```
ComfyUI/models/llm/Qwen-VL/Huihui-Qwen3-VL-4B-Instruct-abliterated/
```

### 3. Download Process

```
📥 Downloading model 'Huihui-Qwen3-VL-4B-Instruct-Abliterated' from MODELSCOPE to ...
📁 Target path: ComfyUI/models/llm/Qwen-VL/Huihui-Qwen3-VL-4B-Instruct-abliterated/
⏳ Note: First download may take a long time, please be patient...
```

### 4. Subsequent Use

After the model is downloaded, it will load directly on next use without re-downloading:

```
✅ Model 'Huihui-Qwen3-VL-4B-Instruct-Abliterated' already exists at ...
📁 Model path: ComfyUI/models/llm/Qwen-VL/Huihui-Qwen3-VL-4B-Instruct-abliterated/
```

## 🔧 Technical Details

### Multi-Source Support

The project now supports two model sources:

1. **HuggingFace** (default)
   - Official Qwen models
   - Most community models

2. **ModelScope** (new)
   - Chinese community models
   - Faster access within China
   - Requires `modelscope` library installation

### Configuration File

Model sources are configured in `config.json`:

```json
{
  "Huihui-Qwen3-VL-4B-Instruct-Abliterated": {
    "repo_id": "fireicewolf/Huihui-Qwen3-VL-4B-Instruct-abliterated",
    "source": "modelscope",  // Specify source as ModelScope
    "default": false,
    "quantized": false,
    "abliterated": true,
    "vram_requirement": {
      "full": 6.0,
      "8bit": 3.5,
      "4bit": 2.0
    },
    "warning": "This model has safety filters removed and may generate sensitive content. For research and testing environments only."
  }
}
```

### Download Logic

The code automatically selects the download method based on the `source` field:

```python
# Check model source
source = model_info.get('source', 'huggingface')

# Select download function based on source
if source == 'modelscope':
    # Use ModelScope download
    from modelscope.hub.snapshot_download import snapshot_download
    snapshot_download(model_id=repo_id, ...)
else:
    # Use HuggingFace download
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, ...)
```

## ❓ FAQ

### Q1: What if the ModelScope library is not installed?

**A**: Run the following command to install:

```bash
pip install modelscope
```

Or reinstall project dependencies:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Qwen3VL-DP
pip install -r requirements.txt
```

### Q2: What if the download fails?

**A**: If automatic download fails, you can:

1. Check your network connection
2. Manually download model files from ModelScope
3. Place the files in: `ComfyUI/models/llm/Qwen-VL/Huihui-Qwen3-VL-4B-Instruct-abliterated/`

Manual download URL:
https://modelscope.cn/models/fireicewolf/Huihui-Qwen3-VL-4B-Instruct-abliterated/files

### Q3: How to add more ModelScope models?

**A**: Edit `config.json` and add new model configuration:

```json
{
  "Your Model Name": {
    "repo_id": "repository ID on ModelScope",
    "source": "modelscope",
    "default": false,
    "quantized": false,
    "vram_requirement": {
      "full": 6.0,
      "8bit": 3.5,
      "4bit": 2.0
    }
  }
}
```

### Q4: What's the difference between ModelScope and HuggingFace?

**A**: 
- **ModelScope**: Chinese platform, faster access within China, some community models
- **HuggingFace**: International platform, most comprehensive model selection, but may have slower access within China

The project automatically selects the appropriate download source based on model configuration.

## 📚 Related Links

- **ModelScope Official Website**: https://modelscope.cn
- **Model Repository**: https://modelscope.cn/models/fireicewolf/Huihui-Qwen3-VL-4B-Instruct-abliterated
- **Project Documentation**: README.md
- **Changelog**: CHANGELOG_Modification_Notes.md

## 🤝 Contributing

If you have good ModelScope model recommendations, feel free to submit an Issue or Pull Request!
