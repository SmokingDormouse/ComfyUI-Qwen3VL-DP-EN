# 🗂️ Project File Structure

## 📁 Core Files

### 🔧 Configuration Files
- `__init__.py` - Automatic node loading and registration
- `config.json` - Model configuration and preset prompts
- `requirements.txt` - Python dependency package list

### 🎮 Node Files
- `qwen3vl_node.py` - Core Qwen3VL nodes (Simple and Advanced versions)
- `qwen3vl_batch_caption.py` - Batch captioning node
- `qwen3vl_compare_caption.py` - Comparison captioning node
- `qwen3vl_extra_options.py` - Extra options configuration node

### 📚 Documentation Files
- `README.md` - Chinese project documentation
- `README_EN.md` - English project documentation
- `PROJECT_STRUCTURE.md` - Project file structure documentation

## 🎯 Node Function Mapping

### Main Nodes
1. **🍭Dapao-Qwen3VL (Simple)** → `qwen3vl_node.py:Qwen3VL_Simple`
2. **🍭Dapao-Qwen3VL (Advanced)** → `qwen3vl_node.py:Qwen3VL_Advanced`

### Batch Processing Nodes
3. **🍭Dapao-Qwen3VL Batch Caption** → `qwen3vl_batch_caption.py:Qwen3VL_Batch_Caption`
4. **🍭Dapao-Qwen3VL Compare Caption** → `qwen3vl_compare_caption.py:Qwen3VL_Compare_Caption`

### Configuration Nodes
5. **🍭Dapao-Qwen3VL Extra Options** → `qwen3vl_extra_options.py:Qwen3VL_ExtraOptions`

## 📊 File Size Statistics

| Filename | Size | Function Description |
|----------|------|----------------------|
| `qwen3vl_node.py` | ~33KB | Core model processing logic |
| `qwen3vl_compare_caption.py` | ~19KB | Comparison captioning functionality |
| `qwen3vl_batch_caption.py` | ~18KB | Batch captioning functionality |
| `README.md` | ~12KB | Chinese documentation |
| `qwen3vl_extra_options.py` | ~10KB | Extra options configuration |
| `README_EN.md` | ~9KB | English documentation |
| `config.json` | ~6KB | Configuration file |
| `__init__.py` | ~2KB | Initialization file |
| `requirements.txt` | ~0.4KB | Dependency list |

## 🔗 Dependency Relationships

```
qwen3vl_node.py (Core)
    ↑
    ├── qwen3vl_batch_caption.py (Depends on core)
    ├── qwen3vl_compare_caption.py (Depends on core)
    └── qwen3vl_extra_options.py (Independent, called by batch captioning)
```

## 🎨 Code Architecture Features

### Modular Design
- **Core Separation**: Main model logic in `qwen3vl_node.py`
- **Feature Extension**: Batch and comparison features as independent modules
- **Configuration Decoupling**: Extra options as optional module

### Code Reuse
- Both batch captioning and comparison captioning reuse the `Qwen3VL_Advanced` class from the core node
- Unified error handling and progress display mechanism
- Shared model configuration and prompt system

### Chinese-Friendly
- All Chinese parameter names (with emoji icons)
- Detailed Chinese comments and docstrings
- Error messages and logs oriented toward Chinese users

## 🚀 Deployment Checklist

Confirm the following files before release:
- [x] `qwen3vl_node.py` - Core functionality
- [x] `qwen3vl_batch_caption.py` - Batch captioning
- [x] `qwen3vl_compare_caption.py` - Comparison captioning
- [x] `qwen3vl_extra_options.py` - Extra options
- [x] `__init__.py` - Node registration
- [x] `config.json` - Configuration file
- [x] `requirements.txt` - Dependency list
- [x] `README.md` - Chinese documentation
- [x] `README_EN.md` - English documentation
- [x] `PROJECT_STRUCTURE.md` - Project structure documentation

## 📝 Version Information
- **Current Version**: v2.0.0
- **Release Date**: 2025-11-14
- **Main Features**: Batch captioning, comparison captioning, modular design
```