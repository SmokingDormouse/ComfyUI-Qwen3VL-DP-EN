#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test configuration file and new models"""

import json
from pathlib import Path

# Read configuration file
config_path = Path(__file__).parent / "config.json"
with open(config_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Count number of models
models = [k for k in data.keys() if not k.startswith('_')]
print(f"✅ Configuration file loaded successfully!")
print(f"📊 Found {len(models)} models in total")

# Check new model
new_model_name = 'Huihui-Qwen3-VL-4B-Instruct-Abliterated'
if new_model_name in data:
    model_config = data[new_model_name]
    print(f"\n✅ New model '{new_model_name}' configured successfully!")
    print(f"   📦 Repo ID: {model_config.get('repo_id')}")
    print(f"   🌐 Source: {model_config.get('source', 'huggingface')}")
    print(f"   💾 VRAM requirement: {model_config.get('vram_requirement')}")
    print(f"   ⚠️  Warning: {model_config.get('warning', 'None')}")
else:
    print(f"\n❌ New model '{new_model_name}' not found")

# List all models
print(f"\n📋 All available models:")
for i, model_name in enumerate(models, 1):
    model_info = data[model_name]
    source = model_info.get('source', 'huggingface')
    print(f"  {i}. {model_name} (source: {source})")
