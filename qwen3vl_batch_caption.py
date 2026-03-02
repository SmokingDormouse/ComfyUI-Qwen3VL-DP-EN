"""
Qwen3VL batch caption node.
Batch-process images in a folder and write a matching `.txt` caption/description file for each image.

Features:
- Batch-process common image formats (jpg, jpeg, png, bmp, webp, gif)
- Choose from preset prompts or provide a custom prompt
- Optional overwrite of existing caption files (regenerate with a different style)
- Optional renaming and prefix/suffix text injection
- Optional connection to the Qwen3VL Extra Options node for modular, fine-grained control
- Detailed logs and error output for easier debugging

Modular design:
- Optionally connect the "Qwen3VL Extra Options" node to control what gets described
- Keeps the main node simple by moving advanced controls into a dedicated node
"""

import os
import sys
import time
import torch
import random
import numpy as np
from PIL import Image
from pathlib import Path
import comfy.utils

# Dynamically import qwen3vl_node from the same directory
# to avoid relative import issues
try:
    from .qwen3vl_node import (
        Qwen3VL_Advanced,
        MODEL_CONFIGS,
        SYSTEM_PROMPTS,
        Quantization,
        ImageProcessor
    )
except ImportError:
    # Fall back to direct import if relative import fails
    import qwen3vl_node
    Qwen3VL_Advanced = qwen3vl_node.Qwen3VL_Advanced
    MODEL_CONFIGS = qwen3vl_node.MODEL_CONFIGS
    SYSTEM_PROMPTS = qwen3vl_node.SYSTEM_PROMPTS
    Quantization = qwen3vl_node.Quantization
    ImageProcessor = qwen3vl_node.ImageProcessor


class Qwen3VL_Batch_Caption:
    """Qwen3-VL batch caption node - batch-process images in a folder."""

    def __init__(self):
        # Reuse the core functionality of Qwen3VL_Advanced
        self.advanced_node = Qwen3VL_Advanced()
        self.last_seed = -1

    @classmethod
    def INPUT_TYPES(cls):
        """Define input types for the batch caption node."""
        model_names = [name for name in MODEL_CONFIGS.keys() if not name.startswith('_')]
        default_model = model_names[4] if len(model_names) > 4 else model_names[0]
        preset_prompts = MODEL_CONFIGS.get("_preset_prompts", ["Describe this image in detail"])

        return {
            "required": {
                "🤖 Model": (model_names, {"default": default_model}),
                "⚙️ Quantization": (list(Quantization.get_values()), {"default": Quantization.NONE}),
                "🖼️ Max Long Side": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "📁 Input Folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to the folder containing images"
                }),
                "📂 Output Folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Leave empty to save next to the input images"
                }),
                "💭 Preset Prompt": (preset_prompts, {"default": preset_prompts[2]}),
                "✏️ Custom Prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Pick a preset prompt or type a custom one"
                }),
                "🔢 Max Tokens": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "🌡️ Temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "🎯 Top-p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "🚀 Enable TF32": ("BOOLEAN", {"default": False, "tooltip": "Enable TF32 acceleration (Ampere+ GPUs only; can significantly improve speed)."}),
                "🔄 Keep Model Loaded": ("BOOLEAN", {"default": False}),
                "🎲 Seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (-1 means random)."
                }),
                "🎯 Seed Mode": (["Random", "Fixed", "Increment"], {"default": "Random"}),
                "📝 Prefix Text": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Text to prepend to the caption"
                }),
                "📌 Suffix Text": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Text to append to the caption"
                }),
                "🔄 Rename Files": ("BOOLEAN", {"default": False}),
                "🏷️ Filename Prefix": ("STRING", {
                    "default": "image_",
                    "multiline": False,
                    "placeholder": "Prefix used when renaming files"
                }),
                "🔢 Start Number": ("INT", {"default": 1, "min": 0, "max": 9999999, "step": 1}),
                "🔄 Force Overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overwrite existing `.txt` files to regenerate captions with a different style."
                }),
            },
            "optional": {
                "🎯 Qwen3VL Extra Options": ("QWEN3VL_EXTRA_OPTIONS", {
                    "tooltip": "Optional Qwen3VL extra options (connect the Qwen3VL Extra Options node)."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Result",)
    FUNCTION = "batch_process"
    CATEGORY = "Qwen3VL-DP"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        seed_mode = kwargs.get("🎯 Seed Mode", "Random")
        seed = kwargs.get("🎲 Seed", -1)

        # Random and Increment modes always force update (return NaN)
        if seed_mode in ["Random", "Increment"]:
            return float("nan")

        # Fixed mode: only update when seed value changes
        return seed

    def process_single_image(self, image_path: str, prompt_text: str, **kwargs) -> str:
        """
        Process a single image.

        Args:
            image_path: Image file path
            prompt_text: Prompt text
            **kwargs: Additional parameters

        Returns:
            Generated caption text
        """
        # Let errors propagate directly without try-except
        print(f"   🔍 Processing image: {os.path.basename(image_path)}")
        # Load image
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            max_long_side = kwargs.get("max_long_side", 768)
            if max_long_side and max_long_side > 0:
                w, h = img.size
                if w > max_long_side or h > max_long_side:
                    resample = getattr(Image, "Resampling", Image).LANCZOS
                    img.thumbnail((max_long_side, max_long_side), resample=resample)

            # Convert to tensor format (ComfyUI format: H,W,C, range 0-1)
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dimension

            # Call the advanced node's process function
            # Note: keys must match qwen3vl_node.py's INPUT_TYPES
            result = self.advanced_node.process(
                **{
                    "🤖 Model": kwargs.get("model_name"),
                    "⚙️ Quantization": kwargs.get("quantization"),
                    "🖼️ Max Long Side": kwargs.get("max_long_side", 768),
                    "💭 Preset Prompt": kwargs.get("preset_prompt"),
                    "✏️ Custom Prompt": prompt_text,
                    "🔢 Max Tokens": kwargs.get("max_tokens"),
                    "🌡️ Temperature": kwargs.get("temperature"),
                    "🎯 Top-p": kwargs.get("top_p"),
                    "🔍 Num Beams": 1,
                    "🚫 Repetition Penalty": 1.2,
                    "🎬 Video Frames": 16,
                    "💻 Device": "auto",
                    "🚀 Enable TF32": kwargs.get("enable_tf32", False),
                    "🔄 Keep Model Loaded": kwargs.get("keep_model_loaded", False),
                    "🎲 Seed": kwargs.get("seed"),
                    "🎯 Seed Mode": kwargs.get("seed_mode"),
                    "🖼️ Image 1": img_tensor,
                }
            )

            return result[0] if result else ""

    @torch.no_grad()
    def batch_process(self, **kwargs):
        """Batch-process images in a folder."""
        # Extract parameters
        model_name = kwargs.get("🤖 Model")
        quantization = kwargs.get("⚙️ Quantization")
        max_long_side = kwargs.get("🖼️ Max Long Side", 768)
        input_folder = kwargs.get("📁 Input Folder", "").strip()
        output_folder = kwargs.get("📂 Output Folder", "").strip()
        preset_prompt = kwargs.get("💭 Preset Prompt")
        custom_prompt = kwargs.get("✏️ Custom Prompt", "").strip()
        max_tokens = kwargs.get("🔢 Max Tokens")
        temperature = kwargs.get("🌡️ Temperature")
        top_p = kwargs.get("🎯 Top-p")
        keep_model_loaded = kwargs.get("🔄 Keep Model Loaded", False)
        enable_tf32 = kwargs.get("🚀 Enable TF32", False)
        seed = kwargs.get("🎲 Seed")
        seed_mode = kwargs.get("🎯 Seed Mode", "Random")
        prefix_text = kwargs.get("📝 Prefix Text", "").strip()
        suffix_text = kwargs.get("📌 Suffix Text", "").strip()
        rename_files = kwargs.get("🔄 Rename Files", False)
        filename_prefix = kwargs.get("🏷️ Filename Prefix", "image_")
        start_number = kwargs.get("🔢 Start Number", 1)
        force_overwrite = kwargs.get("🔄 Force Overwrite", False)

        # Configure TF32 acceleration
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = enable_tf32
            torch.backends.cudnn.allow_tf32 = enable_tf32
            if enable_tf32:
                print("🚀 TF32 acceleration enabled")

        # If seed mode is Random, force overwrite by default
        if seed_mode == "Random":
            force_overwrite = True

        # Qwen3VL extra options (optional)
        extra_options = kwargs.get("🎯 Qwen3VL Extra Options", None)

        # Validate input folder
        if not input_folder or not os.path.exists(input_folder):
            return ("❌ Error: Input folder path is invalid or does not exist",)

        # Set output folder
        if not output_folder:
            output_folder = input_folder
        else:
            os.makedirs(output_folder, exist_ok=True)

        # Determine the prompt to use
        base_prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt:
            base_prompt = custom_prompt

        # Apply Qwen3VL extra options to build enhanced prompt (if connected)
        if extra_options:
            # Import the static method from the Qwen3VL extra options node
            try:
                import qwen3vl_extra_options
                prompt_text = qwen3vl_extra_options.Qwen3VL_ExtraOptions.build_enhanced_prompt(base_prompt, extra_options)
            except (ImportError, AttributeError) as e:
                print(f"⚠️ Warning: Failed to import Qwen3VL Extra Options module ({e}), using base prompt")
                prompt_text = base_prompt
        else:
            prompt_text = base_prompt

        # Supported image formats
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')

        # Collect all image files
        image_files = []
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(image_extensions):
                image_files.append(filename)

        if not image_files:
            return (f"⚠️ Warning: No image files found in folder {input_folder}",)

        # Sort file list
        image_files.sort()

        print(f"\n{'='*60}")
        print(f"🚀 Starting batch caption")
        print(f"📁 Input folder: {input_folder}")
        print(f"📂 Output folder: {output_folder}")
        print(f"🖼️ Image count: {len(image_files)}")
        print(f"💭 Base prompt: {base_prompt}")

        # Show Qwen3VL extra options status
        if extra_options:
            enabled_options = [key for key, value in extra_options.items() if value]
            if enabled_options:
                print(f"🎯 Qwen3VL Extra Options: Enabled ({len(enabled_options)})")
                print(f"   Enabled options: {', '.join(enabled_options)}")
            else:
                print(f"🎯 Qwen3VL Extra Options: Connected but no options enabled")
        else:
            print(f"🎯 Qwen3VL Extra Options: Not connected")

        print(f"🎮 Seed mode: {seed_mode}")
        print(f"🔄 Overwrite: {'Yes' if force_overwrite else 'No'}")
        print(f"📋 Found image files:")
        for i, file in enumerate(image_files, 1):
            print(f"   {i}. {file}")
        print(f"{'='*60}\n")

        # Statistics
        success_count = 0
        fail_count = 0
        skip_count = 0
        start_time = time.time()

        # Detailed processing results
        success_files = []
        fail_files = []
        skip_files = []

        # Create progress bar
        pbar = comfy.utils.ProgressBar(len(image_files))

        # Pre-load the model so a missing model raises immediately (not caught by try-except)
        self.advanced_node.load_model(model_name, quantization, "auto")

        # Calculate the effective initial seed
        if seed_mode == "Fixed":
            effective_seed = seed if seed != -1 else random.randint(0, 2147483647)
        elif seed_mode == "Random":
            effective_seed = random.randint(0, 2147483647)
        elif seed_mode == "Increment":
            if self.last_seed == -1:
                effective_seed = seed if seed != -1 else random.randint(0, 2147483647)
            else:
                effective_seed = self.last_seed + 1
        else:
            effective_seed = random.randint(0, 2147483647)

        self.last_seed = effective_seed

        # Process each image
        current_number = start_number
        for idx, filename in enumerate(image_files):
            # Let errors propagate directly
            image_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]

            # Determine output filename
            if rename_files:
                new_base_name = f"{filename_prefix}{current_number:04d}"
                current_number += 1
            else:
                new_base_name = base_name
                # Add extension suffix to avoid name conflicts
                original_ext = os.path.splitext(filename)[1].lower()
                if original_ext in ['.jpeg', '.jpg']:
                    ext_suffix = '_jpg'
                elif original_ext == '.png':
                    ext_suffix = '_png'
                elif original_ext == '.bmp':
                    ext_suffix = '_bmp'
                elif original_ext == '.webp':
                    ext_suffix = '_webp'
                elif original_ext == '.gif':
                    ext_suffix = '_gif'
                else:
                    ext_suffix = original_ext.replace('.', '_')

                # Check if suffix is needed to avoid conflicts
                base_text_path = os.path.join(output_folder, f"{new_base_name}.txt")
                if os.path.exists(base_text_path) and not force_overwrite:
                    new_base_name = f"{base_name}{ext_suffix}"

            text_path = os.path.join(output_folder, f"{new_base_name}.txt")

            # Skip if caption file already exists (only when not force overwriting)
            if os.path.exists(text_path) and not force_overwrite:
                print(f"⏭️ Skipped (exists): {filename} (enable 'Force Overwrite' to overwrite)")
                skip_count += 1
                skip_files.append(filename)
                pbar.update_absolute(idx + 1, len(image_files))
                continue
            elif os.path.exists(text_path) and force_overwrite:
                print(f"🔄 Overwriting: {filename}")

            print(f"🖼️ Processing [{idx+1}/{len(image_files)}]: {filename}")
            print(f"   📁 Image path: {image_path}")
            print(f"   📝 Output path: {text_path}")

            # Process image
            caption = self.process_single_image(
                image_path,
                prompt_text,
                model_name=model_name,
                quantization=quantization,
                max_long_side=max_long_side,
                preset_prompt=preset_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_tf32=enable_tf32,
                keep_model_loaded=keep_model_loaded,
                seed=seed,
                seed_mode=seed_mode
            )

            if caption:
                # Add prefix and suffix
                if prefix_text:
                    caption = f"{prefix_text} {caption}"
                if suffix_text:
                    caption = f"{caption} {suffix_text}"

                # Save caption file
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(caption)

                print(f"✅ Success: {new_base_name}.txt")
                print(f"   Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}\n")
                success_count += 1
                success_files.append(filename)
            else:
                print("❌ Failed: empty caption generated\n")
                fail_count += 1
                fail_files.append(filename)

            # Update progress bar
            pbar.update_absolute(idx + 1, len(image_files))

        # Unload model (if not keeping loaded)
        if not keep_model_loaded:
            self.advanced_node.clear_model_resources()

        # Calculate total elapsed time
        total_time = time.time() - start_time
        avg_time = total_time / len(image_files) if image_files else 0

        # Build detailed result report
        detail_report = []

        if success_files:
            detail_report.append("✅ Successfully processed:")
            for file in success_files:
                detail_report.append(f"   • {file}")

        if fail_files:
            detail_report.append("❌ Failed:")
            for file in fail_files:
                detail_report.append(f"   • {file}")

        if skip_files:
            detail_report.append("⏭️ Skipped:")
            for file in skip_files:
                detail_report.append(f"   • {file}")

        detail_info = "\n".join(detail_report) if detail_report else ""

        # Generate result report
        result_report = f"""
{'='*60}
📊 Batch caption completed
{'='*60}
✅ Success: {success_count}
❌ Failed: {fail_count}
⏭️ Skipped: {skip_count}
📁 Total: {len(image_files)}
⏱️ Total time: {total_time:.2f} s
⚡ Avg time: {avg_time:.2f} s/image
📂 Output: {output_folder}
{'='*60}

{detail_info}

{'='*60}
"""

        print(result_report)

        return (result_report,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "Qwen3VL_Batch_Caption": Qwen3VL_Batch_Caption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL_Batch_Caption": "Qwen3VL Batch Caption",
}
