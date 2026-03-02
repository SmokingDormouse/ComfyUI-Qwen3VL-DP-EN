"""
Qwen3VL compare caption node.
Compares an original image (A) and an AI-edited result image (B), then generates a reverse-engineered edit prompt.

Features:
- Dual-folder input (original A folder + result B folder)
- Automatically checks filename pairing (A ↔ B)
- Switchable built-in prompts (Chinese/English) or custom prompt
- Output location selection (default B folder or custom folder)
- Designed for AI photo-editing workflows (e.g., Kontext/Qwen-edit prompt reverse engineering)
- Based on the proven batch caption node structure

Use cases:
- Reverse-engineering prompts used by AI photo-editing tools
- Automated description of differences between two images
- Textual logging of edit outcomes
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

# Built-in compare caption prompts
COMPARE_PROMPTS = {
    "Chinese": "第二张图片是根据第一张图片经过AI的修图得来的，你现在需要分析第二张图片相比于第一张改动了哪些内容，我需要反推出AI的prompt。这种prompt是指令式的，我需要你用自然语言描述输出这种指令式的结果主要分析图片是哪里做了变动，直接输出你的prompt结果，不要带任何解释性的文字",
    "English": "The second image is the result of AI-based photo editing applied to the first image. You need to analyze what changes were made in the second image compared to the first one, and reverse-engineer the AI prompt. This prompt should be instructional. I need you to describe in natural language the instructional result, mainly analyzing where the image was modified. Output your prompt result directly without any explanatory text."
}


class Qwen3VL_Compare_Caption:
    """Qwen3VL compare caption node - before/after analysis for AI photo edits."""

    def __init__(self):
        # Reuse the core functionality of Qwen3VL_Advanced
        self.advanced_node = Qwen3VL_Advanced()
        self.last_seed = -1

    @classmethod
    def INPUT_TYPES(cls):
        """Define input types for the compare caption node."""
        model_names = [name for name in MODEL_CONFIGS.keys() if not name.startswith('_')]
        default_model = model_names[4] if len(model_names) > 4 else model_names[0]

        return {
            "required": {
                "🤖 Model": (model_names, {"default": default_model}),
                "⚙️ Quantization": (list(Quantization.get_values()), {"default": Quantization.NONE}),
                "🖼️ Max Long Side": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "📁 Folder A (Original)": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to the original images folder (A)"
                }),
                "📂 Folder B (Result)": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to the AI-edited results folder (B)"
                }),
                "🌍 Language": (["Chinese", "English"], {"default": "Chinese"}),
                "✏️ Custom Prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Leave empty to use the built-in prompt; any text here overrides it"
                }),
                "📍 Output Location": (["Default (Folder B)", "Custom"], {"default": "Default (Folder B)"}),
                "📂 Custom Output Folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Used when output location is set to custom"
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
                    "placeholder": "Text to prepend to the output"
                }),
                "📌 Suffix Text": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Text to append to the output"
                }),
                "🔄 Force Overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overwrite existing `.txt` files."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Result",)
    FUNCTION = "compare_process"
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

    def process_image_pair(self, image_a_path: str, image_b_path: str, prompt_text: str, **kwargs) -> str:
        """
        Process an image pair (original + edited result).

        Args:
            image_a_path: Original image file path
            image_b_path: Result image file path
            prompt_text: Prompt text
            **kwargs: Additional parameters

        Returns:
            Generated comparison analysis text
        """
        # Let errors propagate directly without try-except
        print(f"   🔍 Processing image pair: {os.path.basename(image_a_path)} vs {os.path.basename(image_b_path)}")

        # Load both images
        images = []
        for img_path in [image_a_path, image_b_path]:
            with Image.open(img_path) as img:
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
                images.append(img_tensor)

        # Call the advanced node's process function with both images
        # Note: keys must match qwen3vl_node.py's INPUT_TYPES
        result = self.advanced_node.process(
            **{
                "🤖 Model": kwargs.get("model_name"),
                "⚙️ Quantization": kwargs.get("quantization"),
                "🖼️ Max Long Side": kwargs.get("max_long_side", 768),
                "💭 Preset Prompt": "",  # Overridden by Custom Prompt below
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
                "🖼️ Image 1": images[0],  # Original image (first)
                "🖼️ Image 2": images[1],  # Result image (second)
            }
        )

        return result[0] if result else ""

    def check_file_correspondence(self, folder_a: str, folder_b: str):
        """
        Check whether files in the two folders match 1-to-1.

        Args:
            folder_a: Path to folder A
            folder_b: Path to folder B

        Returns:
            (is_matched, matched_pairs, error_msg)
        """
        # Supported image formats
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')

        # Collect image files from folder A
        files_a = []
        if os.path.exists(folder_a):
            for filename in os.listdir(folder_a):
                if filename.lower().endswith(image_extensions):
                    files_a.append(filename)

        # Collect image files from folder B
        files_b = []
        if os.path.exists(folder_b):
            for filename in os.listdir(folder_b):
                if filename.lower().endswith(image_extensions):
                    files_b.append(filename)

        # Sort file lists
        files_a.sort()
        files_b.sort()

        # Check count consistency
        if len(files_a) != len(files_b):
            return False, [], f"File count mismatch: folder A ({len(files_a)}) vs folder B ({len(files_b)})"

        if len(files_a) == 0:
            return False, [], "No image files found in either folder"

        # Check filenames match 1-to-1 (ignoring extension)
        mismatched_files = []
        matched_pairs = []

        for i, (file_a, file_b) in enumerate(zip(files_a, files_b)):
            base_a = os.path.splitext(file_a)[0]
            base_b = os.path.splitext(file_b)[0]

            if base_a == base_b:
                matched_pairs.append((file_a, file_b))
            else:
                mismatched_files.append(f"Position {i+1}: {file_a} vs {file_b}")

        if mismatched_files:
            error_msg = "Filename mismatch:\n" + "\n".join(mismatched_files)
            return False, [], error_msg

        return True, matched_pairs, ""

    @torch.no_grad()
    def compare_process(self, **kwargs):
        """Compare-process image pairs from two folders."""
        # Extract parameters
        model_name = kwargs.get("🤖 Model")
        quantization = kwargs.get("⚙️ Quantization")
        max_long_side = kwargs.get("🖼️ Max Long Side", 768)
        folder_a = kwargs.get("📁 Folder A (Original)", "").strip()
        folder_b = kwargs.get("📂 Folder B (Result)", "").strip()
        language = kwargs.get("🌍 Language", "Chinese")
        custom_prompt = kwargs.get("✏️ Custom Prompt", "").strip()
        output_location = kwargs.get("📍 Output Location", "Default (Folder B)")
        custom_output_folder = kwargs.get("📂 Custom Output Folder", "").strip()
        max_tokens = kwargs.get("🔢 Max Tokens")
        temperature = kwargs.get("🌡️ Temperature")
        top_p = kwargs.get("🎯 Top-p")
        keep_model_loaded = kwargs.get("🔄 Keep Model Loaded", False)
        enable_tf32 = kwargs.get("🚀 Enable TF32", False)
        seed = kwargs.get("🎲 Seed")
        seed_mode = kwargs.get("🎯 Seed Mode", "Random")
        prefix_text = kwargs.get("📝 Prefix Text", "").strip()
        suffix_text = kwargs.get("📌 Suffix Text", "").strip()
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

        # Validate input folders
        if not folder_a or not os.path.exists(folder_a):
            return ("❌ Error: Folder A (original) path is invalid or does not exist",)

        if not folder_b or not os.path.exists(folder_b):
            return ("❌ Error: Folder B (result) path is invalid or does not exist",)

        # Determine output folder
        if output_location == "Default (Folder B)":
            output_folder = folder_b
        else:
            if not custom_output_folder:
                return ("❌ Error: A custom output folder must be specified when using custom output location",)
            output_folder = custom_output_folder
            os.makedirs(output_folder, exist_ok=True)

        # Determine the prompt to use
        if custom_prompt:
            prompt_text = custom_prompt
        else:
            prompt_text = COMPARE_PROMPTS.get(language, COMPARE_PROMPTS["Chinese"])

        # Check file pairing
        is_matched, file_pairs, error_msg = self.check_file_correspondence(folder_a, folder_b)

        if not is_matched:
            return (f"❌ File pairing check failed: {error_msg}",)

        print(f"\n{'='*60}")
        print(f"🚀 Starting compare caption")
        print(f"📁 Folder A (original): {folder_a}")
        print(f"📂 Folder B (result): {folder_b}")
        print(f"📂 Output folder: {output_folder}")
        print(f"🖼️ Image pair count: {len(file_pairs)}")
        print(f"🌍 Language: {language}")
        print(f"💭 Prompt: {'Custom' if custom_prompt else 'Built-in'}")
        print(f"🎮 Seed mode: {seed_mode}")
        print(f"🔄 Overwrite: {'Yes' if force_overwrite else 'No'}")
        print(f"📋 Found image pairs:")
        for i, (file_a, file_b) in enumerate(file_pairs, 1):
            print(f"   {i}. {file_a} ↔ {file_b}")
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
        pbar = comfy.utils.ProgressBar(len(file_pairs))

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

        # Process each image pair
        for idx, (file_a, file_b) in enumerate(file_pairs):
            # Let errors propagate directly
            image_a_path = os.path.join(folder_a, file_a)
            image_b_path = os.path.join(folder_b, file_b)

            # Determine output filename (based on B folder filename)
            base_name = os.path.splitext(file_b)[0]
            text_path = os.path.join(output_folder, f"{base_name}.txt")

            # Skip if caption file already exists (only when not force overwriting)
            if os.path.exists(text_path) and not force_overwrite:
                print(f"⏭️ Skipped (exists): {file_b} (enable 'Force Overwrite' to overwrite)")
                skip_count += 1
                skip_files.append(f"{file_a} ↔ {file_b}")
                pbar.update_absolute(idx + 1, len(file_pairs))
                continue
            elif os.path.exists(text_path) and force_overwrite:
                print(f"🔄 Overwriting: {file_b}")

            print(f"🖼️ Processing [{idx+1}/{len(file_pairs)}]: {file_a} ↔ {file_b}")
            print(f"   📁 Original image: {image_a_path}")
            print(f"   📁 Result image: {image_b_path}")
            print(f"   📝 Output path: {text_path}")

            # Process image pair
            caption = self.process_image_pair(
                image_a_path,
                image_b_path,
                prompt_text,
                model_name=model_name,
                quantization=quantization,
                max_long_side=max_long_side,
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

                print(f"✅ Success: {base_name}.txt")
                print(f"   Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}\n")
                success_count += 1
                success_files.append(f"{file_a} ↔ {file_b}")
            else:
                print("❌ Failed: empty output generated\n")
                fail_count += 1
                fail_files.append(f"{file_a} ↔ {file_b}")

            # Update progress bar
            pbar.update_absolute(idx + 1, len(file_pairs))

        # Unload model (if not keeping loaded)
        if not keep_model_loaded:
            self.advanced_node.clear_model_resources()

        # Calculate total elapsed time
        total_time = time.time() - start_time
        avg_time = total_time / len(file_pairs) if file_pairs else 0

        # Build detailed result report
        detail_report = []

        if success_files:
            detail_report.append("✅ Successfully processed pairs:")
            for file_pair in success_files:
                detail_report.append(f"   • {file_pair}")

        if fail_files:
            detail_report.append("❌ Failed pairs:")
            for file_pair in fail_files:
                detail_report.append(f"   • {file_pair}")

        if skip_files:
            detail_report.append("⏭️ Skipped pairs:")
            for file_pair in skip_files:
                detail_report.append(f"   • {file_pair}")

        detail_info = "\n".join(detail_report) if detail_report else ""

        # Generate result report
        result_report = f"""
{'='*60}
🎯 Compare caption report
{'='*60}
📊 Summary:
   • Total pairs: {len(file_pairs)}
   • Success: {success_count}
   • Failed: {fail_count}
   • Skipped: {skip_count}

⏱️ Timing:
   • Total time: {total_time:.2f} s
   • Avg time: {avg_time:.2f} s/pair

📁 Folders:
   • A folder (original): {folder_a}
   • B folder (edited): {folder_b}
   • Output folder: {output_folder}

🌍 Settings:
   • Language: {language}
   • Prompt: {'Custom' if custom_prompt else 'Built-in'}
   • Overwrite: {'Yes' if force_overwrite else 'No'}

{detail_info}
{'='*60}
"""

        print(result_report)
        return (result_report,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "Qwen3VL_Compare_Caption": Qwen3VL_Compare_Caption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL_Compare_Caption": "Qwen3VL Compare Captions",
}
