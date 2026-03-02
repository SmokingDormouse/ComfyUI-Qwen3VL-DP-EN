#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test model download functionality
"""

import sys
from pathlib import Path

# Add ComfyUI path to sys.path
comfyui_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(comfyui_path))

import folder_paths
from qwen3vl_node import ModelDownloader, load_model_configs

def test_model_download():
    """Test model download functionality"""
    print("=" * 60)
    print("🧪 Testing Qwen3VL Model Download Functionality")
    print("=" * 60)
    
    # Load model configuration
    load_model_configs()
    from qwen3vl_node import MODEL_CONFIGS
    
    print(f"\n📋 Available models list:")
    for i, model_name in enumerate(MODEL_CONFIGS.keys(), 1):
        if not model_name.startswith('_'):
            model_info = MODEL_CONFIGS[model_name]
            print(f"  {i}. {model_name}")
            print(f"     Repo: {model_info.get('repo_id', 'N/A')}")
    
    # Create downloader
    downloader = ModelDownloader(MODEL_CONFIGS)
    
    print(f"\n📁 Model storage directory: {downloader.models_dir}")
    print(f"   Directory exists: {'✅ Yes' if downloader.models_dir.exists() else '❌ No'}")
    
    # Check downloaded models
    print(f"\n🔍 Checking downloaded models:")
    if downloader.models_dir.exists():
        downloaded_models = list(downloader.models_dir.iterdir())
        if downloaded_models:
            for model_dir in downloaded_models:
                if model_dir.is_dir():
                    config_file = model_dir / "config.json"
                    model_file = model_dir / "model.safetensors"
                    model_index = model_dir / "model.safetensors.index.json"
                    
                    status = "✅ Complete" if config_file.exists() and (model_file.exists() or model_index.exists()) else "⚠️ Incomplete"
                    print(f"  - {model_dir.name}: {status}")
        else:
            print("  ❌ No downloaded models")
    else:
        print("  ❌ Model directory does not exist")
    
    print("\n" + "=" * 60)
    print("💡 Tips:")
    print("  - On first use, models will be downloaded automatically")
    print("  - Models are saved to ComfyUI/models/llm/Qwen-VL/")
    print("  - If model already exists, it won't be re-downloaded")
    print("=" * 60)
    
    # Ask whether to test download
    print("\n🤔 Do you want to test downloading a small model? (Qwen3-VL-2B-Instruct, ~4GB)")
    print("   Enter 'yes' to start download, any other key to skip")
    
    try:
        choice = input(">>> ").strip().lower()
        if choice == 'yes':
            print("\n📥 Starting download test...")
            try:
                model_path = downloader.ensure_model_available("Qwen3-VL-2B-Instruct")
                print(f"\n✅ Test successful! Model path: {model_path}")
            except Exception as e:
                print(f"\n❌ Download failed: {e}")
        else:
            print("\n⏭️ Skipping download test")
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted")
    
    print("\n✨ Test complete!")

if __name__ == "__main__":
    test_model_download()
