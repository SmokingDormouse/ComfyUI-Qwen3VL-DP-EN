import torch
import time
import json
import random
import platform
import psutil
import numpy as np

from packaging import version
from PIL import Image
from enum import Enum
from pathlib import Path
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download as hf_snapshot_download
import folder_paths
import gc

# Try to import ModelScope; fall back to HuggingFace if not available
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    ms_snapshot_download = None
    MODELSCOPE_AVAILABLE = False
    print("[Qwen3VL] ⚠️ ModelScope is not installed. ModelScope models cannot be downloaded. Run: pip install modelscope")

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "config.json"
MODEL_CONFIGS = {}
SYSTEM_PROMPTS = {}

def load_model_configs():
    """Load model configuration file."""
    global MODEL_CONFIGS, SYSTEM_PROMPTS
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            MODEL_CONFIGS = json.load(f)
            SYSTEM_PROMPTS = MODEL_CONFIGS.get("_system_prompts", {})
    except FileNotFoundError:
        print(f"Error: config file not found: {CONFIG_PATH}")
        MODEL_CONFIGS, SYSTEM_PROMPTS = {}, {}
    except json.JSONDecodeError:
        print("Error: failed to parse config file")
        MODEL_CONFIGS, SYSTEM_PROMPTS = {}, {}

    # Load user-defined custom model configurations
    custom_path = NODE_DIR / "custom_models.json"
    if custom_path.exists():
        try:
            with open(custom_path, "r", encoding="utf-8") as f:
                custom_data = json.load(f) or {}

            user_models = custom_data.get("hf_models", {}) or custom_data.get("models", {})

            if user_models:
                MODEL_CONFIGS.update(user_models)
                print(f"[Qwen3VL] ✅ Loaded {len(user_models)} custom model(s)")
            else:
                print("[Qwen3VL] ⚠️ Found custom_models.json but it contains no valid model entries")
        except Exception as e:
            print(f"[Qwen3VL] ⚠️ Failed to load custom_models.json → {e}")
    else:
        print("[Qwen3VL] ℹ️ custom_models.json not found; skipping custom models")

if not MODEL_CONFIGS:
    load_model_configs()

class Quantization(str, Enum):
    """Quantization options enum."""
    Q4_BIT = "4-bit (lower VRAM)"
    Q8_BIT = "8-bit (balanced)"
    NONE = "None (FP16)"
    
    @classmethod
    def get_values(cls):
        return [item.value for item in cls]

def get_model_info(model_name: str) -> dict:
    """Get model info."""
    return MODEL_CONFIGS.get(model_name, {})

def get_device_info() -> dict:
    """Get device info."""
    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / 1024**3
        gpu_info = {
            "available": True,
            "total_memory": total_mem,
            "free_memory": total_mem - (torch.cuda.memory_allocated(0) / 1024**3)
        }
    else:
        gpu_info = {"available": False, "total_memory": 0, "free_memory": 0}

    sys_mem = psutil.virtual_memory()
    sys_mem_info = {
        "total": sys_mem.total / 1024**3,
        "available": sys_mem.available / 1024**3
    }

    device_info = {
        "gpu": gpu_info,
        "system_memory": sys_mem_info,
        "device_type": "cpu",
        "recommended_device": "cpu",
        "memory_sufficient": True,
        "warning_message": ""
    }

    if platform.system() == "Darwin" and platform.processor() == "arm":
        device_info.update({
            "device_type": "apple_silicon",
            "recommended_device": "mps"
        })
        if sys_mem_info["total"] < 16:
            device_info.update({
                "memory_sufficient": False,
                "warning_message": "Apple Silicon memory is below 16GB; performance may be impacted"
            })
    elif gpu_info["available"]:
        device_info.update({
            "device_type": "nvidia_gpu",
            "recommended_device": "cuda"
        })
        if gpu_info["total_memory"] < 8:
            device_info.update({
                "memory_sufficient": False,
                "warning_message": "GPU VRAM is below 8GB; performance may degrade"
            })
    
    return device_info

def check_memory_requirements(model_name: str, quantization: str, device_info: dict) -> str:
    """Check memory requirements and auto-adjust quantization."""
    model_info = get_model_info(model_name)
    vram_req = model_info.get("vram_requirement", {})
    
    quant_map = {
        Quantization.Q4_BIT: vram_req.get("4bit", 0),
        Quantization.Q8_BIT: vram_req.get("8bit", 0),
        Quantization.NONE: vram_req.get("full", 0)
    }
    
    base_memory = quant_map.get(quantization, 0)
    device = device_info["recommended_device"]
    use_cpu_mps = device in ["cpu", "mps"]
    
    required_mem = base_memory * (1.5 if use_cpu_mps else 1.0)
    available_mem = device_info["system_memory"]["available"] if use_cpu_mps else device_info["gpu"]["free_memory"]
    mem_type = "System RAM" if use_cpu_mps else "GPU VRAM"

    if required_mem * 1.2 > available_mem:
        print(f"Warning: insufficient {mem_type} ({available_mem:.2f}GB available). Lowering quantization...")
        if quantization == Quantization.NONE:
            return Quantization.Q8_BIT
        if quantization == Quantization.Q8_BIT:
            return Quantization.Q4_BIT
        raise RuntimeError(f"Insufficient {mem_type}; cannot run even with 4-bit quantization")
    
    return quantization

def check_flash_attention() -> bool:
    """Check whether Flash Attention 2 is supported."""
    try:
        import flash_attn
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return major >= 8
    except ImportError:
        return False
    return False

def resolve_attn_implementation(attn_mode: str) -> str:
    if attn_mode == "Flash Attention 2":
        if check_flash_attention():
            return "flash_attention_2"
        print("⚠️ Flash Attention 2 is unavailable (not installed or unsupported); falling back to SDPA")
        return "sdpa"
    return "sdpa"

class ImageProcessor:
    """Image processor."""
    def to_pil(self, image_tensor, max_side: int = 0) -> Image.Image:
        """Convert a ComfyUI image tensor to a PIL Image."""
        if image_tensor is None:
            return None

        if isinstance(image_tensor, (list, tuple)) and len(image_tensor) > 0:
            image_tensor = image_tensor[0]

        if isinstance(image_tensor, Image.Image):
            img = image_tensor.copy()
        elif torch.is_tensor(image_tensor):
            if image_tensor.dim() == 4:
                image_tensor = image_tensor[0]
            image_np = (image_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(image_np)
        elif isinstance(image_tensor, np.ndarray):
            arr = image_tensor
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255).astype(np.uint8)
            img = Image.fromarray(arr)
        else:
            raise TypeError(f"Unsupported image type: {type(image_tensor)}")

        if max_side and max_side > 0:
            w, h = img.size
            if w > max_side or h > max_side:
                resample = getattr(Image, "Resampling", Image).LANCZOS
                img.thumbnail((max_side, max_side), resample=resample)
        return img

class ModelDownloader:
    """Model downloader.

    Storage path: `ComfyUI/models/llm/Qwen-VL/`
    """
    def __init__(self, configs):
        self.configs = configs
        # Store models in the llm/Qwen-VL subfolder
        self.models_dir = Path(folder_paths.models_dir) / "llm" / "Qwen-VL"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def ensure_model_available(self, model_name):
        """Ensure the model is available; download it if missing.

        The model is downloaded to `ComfyUI/models/llm/Qwen-VL/`.
        If it already exists, it will be reused (no re-download).
        Supports both HuggingFace and ModelScope sources.
        """
        model_info = self.configs.get(model_name)
        if not model_info:
            raise ValueError(f"Model '{model_name}' not found in config")

        repo_id = model_info['repo_id']
        source = model_info.get('source', 'huggingface')  # Default to HuggingFace
        model_folder_name = repo_id.split('/')[-1]
        model_path = self.models_dir / model_folder_name
        
        # Check whether the model has already been fully downloaded
        config_file = model_path / "config.json"
        model_file = model_path / "model.safetensors"
        # Some models use sharded storage
        model_index = model_path / "model.safetensors.index.json"
        
        if model_path.exists() and config_file.exists():
            # Check for model files (full model or sharded model)
            if model_file.exists() or model_index.exists():
                print(f"✅ Model '{model_name}' already exists at {model_path}")
                print(f"📁 Model path: {model_path}")
                return str(model_path)
            else:
                print("⚠️ Model folder exists but files are incomplete; re-downloading...")
        
        # Check whether the ModelScope package is installed
        if source == 'modelscope' and not MODELSCOPE_AVAILABLE:
            raise RuntimeError(
                f"Model '{model_name}' is from ModelScope, but the ModelScope package is not installed.\n"
                "Install it with:\n"
                "pip install modelscope\n"
                f"Or manually download the model to: {model_path}"
            )
        
        print(f"📥 Downloading model '{model_name}' from {source.upper()} to {model_path}...")
        print(f"📁 Target path: {model_path}")
        print("⏳ Tip: the first download can take a while; please be patient...")
        
        # Create model directory
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Select download function based on source
        if source == 'modelscope':
            snapshot_download_func = ms_snapshot_download
            download_kwargs = {
                "model_id": repo_id,
                "cache_dir": str(model_path.parent),
                "local_dir": str(model_path),
            }
            source_url = f"https://modelscope.cn/models/{repo_id}"
        else:
            snapshot_download_func = hf_snapshot_download
            download_kwargs = {
                "repo_id": repo_id,
                "local_dir": str(model_path),
                "local_dir_use_symlinks": False,
                "ignore_patterns": ["*.md", "*.txt", ".gitattributes"],
                "resume_download": True,
                "max_workers": 4
            }
            source_url = f"https://huggingface.co/{repo_id}"
        
        # Retry loop to reduce transient network issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                downloaded_path = snapshot_download_func(**download_kwargs)
                print(f"✅ Model '{model_name}' downloaded successfully!")
                print(f"📁 Model saved to: {model_path}")
                return str(model_path)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Download failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    print("⏳ Waiting 5 seconds before retry...")
                    time.sleep(5)
                else:
                    print(f"❌ Download failed after {max_retries} attempts")
                    error_msg = f"Model download failed: {str(e)}\nSuggestions:\n"
                    if source == 'modelscope':
                        error_msg += (
                            "1. Check your network connection\n"
                            "2. Ensure ModelScope is installed: pip install modelscope\n"
                            f"3. Manually download from {source_url} to: {model_path}\n"
                        )
                    else:
                        error_msg += (
                            "1. Check your network connection\n"
                            "2. Set HF_ENDPOINT to use a mirror (e.g., https://hf-mirror.com)\n"
                            f"3. Manually download from {source_url} to: {model_path}\n"
                        )
                    raise RuntimeError(error_msg)

class Qwen3VL_Advanced:
    """Qwen3-VL advanced node - supports image and video understanding."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model_name = None
        self.current_quantization = None
        self.current_device = None
        self.current_attn_implementation = None
        self.model_device = None
        self.last_seed = -1
        self.device_info = get_device_info()
        self.downloader = ModelDownloader(MODEL_CONFIGS)
        self.image_processor = ImageProcessor()
        print(f"Qwen3VL node initialized. Device: {self.device_info['device_type']}")
        if not self.device_info["memory_sufficient"]:
            print(f"Warning: {self.device_info['warning_message']}")

    def clear_model_resources(self):
        """Release model resources."""
        if self.model is not None:
            print("Releasing model resources...")
            del self.model, self.processor, self.tokenizer
            self.model = self.processor = self.tokenizer = None
            self.current_model_name = self.current_quantization = self.current_device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_model(self, model_name: str, quantization_str: str, device: str = "auto", attn_mode: str = "SDPA"):
        """Load model.

        Args:
            model_name: Model name
            quantization_str: Quantization level string
            device: Device type (auto/cuda/cpu/mps)

        Raises:
            ValueError: When GPU does not support FP8 model or abliterated model is used
        """
        self.device_info = get_device_info()
        effective_device = self.device_info["recommended_device"] if device == "auto" else device
        attn_implementation = resolve_attn_implementation(attn_mode)

        # Skip if model is already loaded with the same configuration
        if (self.model is not None and 
            self.current_model_name == model_name and 
            self.current_quantization == quantization_str and 
            self.current_device == effective_device and
            self.current_attn_implementation == attn_implementation):
            return

        self.clear_model_resources()

        model_info = get_model_info(model_name)
        
        # Warn for abliterated models
        if model_info.get("abliterated"):
            warning_msg = model_info.get("warning", "This model has safety filters removed")
            print(f"\n⚠️  Warning: {warning_msg}\n")
        
        # Check GPU compute capability for FP8 quantized models
        if model_info.get("quantized"):
            if self.device_info["gpu"]["available"]:
                major, minor = torch.cuda.get_device_capability()
                cc = major + minor / 10
                if cc < 8.9:
                    raise ValueError(
                        f"FP8 models require a GPU with compute capability 8.9 or higher (e.g., RTX 4090)."
                        f"Your GPU compute capability is {cc}. Please choose a non-FP8 model."
                    )

        model_path = self.downloader.ensure_model_available(model_name)
        adjusted_quantization = check_memory_requirements(model_name, quantization_str, self.device_info)
        
        quant_config, load_dtype = None, torch.float16
        
        # Only apply quantization config for non-pre-quantized models
        if not get_model_info(model_name).get("quantized", False):
            if adjusted_quantization == Quantization.Q4_BIT:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                load_dtype = None
            elif adjusted_quantization == Quantization.Q8_BIT:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                load_dtype = None

        device_map = "auto"
        if effective_device == "cuda" and torch.cuda.is_available():
            device_map = {"": 0}

        # Build model load kwargs
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": load_dtype,
            "attn_implementation": attn_implementation,
            "use_safetensors": True,
            "trust_remote_code": True  # Required for abliterated models
        }
        
        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        print(f"Loading model '{model_name}'...")
        # Load model, processor, and tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.current_model_name = model_name
        self.current_quantization = quantization_str
        self.current_device = effective_device
        self.current_attn_implementation = attn_implementation
        try:
            self.model_device = str(next(self.model.parameters()).device)
        except StopIteration:
            self.model_device = effective_device
        print("Model loaded successfully")

    @classmethod
    def INPUT_TYPES(cls):
        """Define node input types."""
        model_names = [name for name in MODEL_CONFIGS.keys() if not name.startswith('_')]
        default_model = model_names[4] if len(model_names) > 4 else model_names[0]
        preset_prompts = MODEL_CONFIGS.get("_preset_prompts", ["Describe this image in detail"])

        return {
            "required": {
                "🤖 Model": (model_names, {"default": default_model}),
                "⚙️ Quantization": (list(Quantization.get_values()), {"default": Quantization.NONE}),
                "🧠 Attention Mode": (["SDPA", "Flash Attention 2"], {"default": "SDPA"}),
                "🖼️ Max Long Side": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "💭 Preset Prompt": (preset_prompts, {"default": preset_prompts[2]}),
                "✏️ Custom Prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Pick a preset prompt or type a custom one"
                }),
                "🔢 Max Tokens": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "🌡️ Temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "🎯 Top-p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "🔍 Num Beams": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "🚫 Repetition Penalty": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.01}),
                "🎬 Video Frames": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "💻 Device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "🚀 Enable TF32": ("BOOLEAN", {"default": False, "tooltip": "Enable TF32 acceleration (Ampere+ GPUs only; can significantly improve speed)."}),
                "🔄 Keep Model Loaded": ("BOOLEAN", {"default": False}),
                "🧪 Performance Diagnostics": ("BOOLEAN", {"default": False, "tooltip": "Print key environment and inference info once (useful for debugging slow runs)."}),
                "🎲 Seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (-1 means random)."
                }),
                "🎯 Seed Mode": (["Random", "Fixed", "Increment"], {"default": "Random"}),
            },
            "optional": {
                "🖼️ Image 1": ("IMAGE",),
                "🖼️ Image 2": ("IMAGE",),
                "🖼️ Image 3": ("IMAGE",),
                "🖼️ Image 4": ("IMAGE",),
                "🎥 Video": ("IMAGE",),
                "🎯 Qwen3VL Extra Options": ("QWEN3VL_EXTRA_OPTIONS", {
                    "tooltip": "Optional Qwen3VL extra options (connect the Qwen3VL Extra Options node)."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Text",)
    FUNCTION = "process"
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

    @torch.no_grad()
    def process(self, **kwargs):
        """Process image/video inputs and generate text (uses kwargs for emoji parameter names)."""
        # Extract parameters
        model_name = kwargs.get("🤖 Model")
        quantization = kwargs.get("⚙️ Quantization")
        attn_mode = kwargs.get("🧠 Attention Mode", "SDPA")
        max_long_side = kwargs.get("🖼️ Max Long Side", 768)
        perf_diag = kwargs.get("🧪 Performance Diagnostics", False)
        preset_prompt = kwargs.get("💭 Preset Prompt")
        max_tokens = kwargs.get("🔢 Max Tokens")
        temperature = kwargs.get("🌡️ Temperature")
        top_p = kwargs.get("🎯 Top-p")
        repetition_penalty = kwargs.get("🚫 Repetition Penalty")
        num_beams = kwargs.get("🔍 Num Beams")
        video_frames = kwargs.get("🎬 Video Frames")
        device = kwargs.get("💻 Device")
        seed = kwargs.get("🎲 Seed")
        custom_prompt = kwargs.get("✏️ Custom Prompt", "")
        image1 = kwargs.get("🖼️ Image 1")
        image2 = kwargs.get("🖼️ Image 2")
        image3 = kwargs.get("🖼️ Image 3")
        image4 = kwargs.get("🖼️ Image 4")
        video = kwargs.get("🎥 Video")
        keep_model_loaded = kwargs.get("🔄 Keep Model Loaded", False)
        enable_tf32 = kwargs.get("🚀 Enable TF32", False)
        seed_mode = kwargs.get("🎯 Seed Mode", "Random")
        extra_options = kwargs.get("🎯 Qwen3VL Extra Options", None)
        start_time = time.time()

        # Configure TF32 acceleration
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = enable_tf32
            torch.backends.cudnn.allow_tf32 = enable_tf32
            if enable_tf32:
                print("🚀 TF32 acceleration enabled")

        # Seed logic
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
        print(f"Using random seed: {effective_seed} (mode: {seed_mode})")
        torch.manual_seed(effective_seed)

        # Check transformers version
        if version.parse(transformers.__version__) < version.parse("4.57.0"):
            raise RuntimeError(f"transformers version too low: {transformers.__version__}, requires >= 4.57.0")

        load_start = time.time()
        self.load_model(model_name, quantization, device, attn_mode)
        load_time = time.time() - load_start
        effective_device = self.current_device
        device_for_log = self.model_device or effective_device
        if effective_device == "cpu" and torch.cuda.is_available():
            print("⚠️ Running on CPU; ensure torch is built with CUDA support.")
        if perf_diag:
            try:
                import sys
                model_dtype = None
                try:
                    model_dtype = str(next(self.model.parameters()).dtype)
                except StopIteration:
                    model_dtype = "unknown"
                try:
                    sdp_info = f"sdp_flash={torch.backends.cuda.flash_sdp_enabled()} sdp_mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} sdp_math={torch.backends.cuda.math_sdp_enabled()}"
                except Exception:
                    sdp_info = "sdp_info=unknown"
                pv_shape = None
                if "pixel_values" in model_inputs and torch.is_tensor(model_inputs["pixel_values"]):
                    pv = model_inputs["pixel_values"]
                    pv_shape = f"pixel_values={tuple(pv.shape)} {str(pv.dtype).replace('torch.', '')}"
                elif "pixel_values_videos" in model_inputs and torch.is_tensor(model_inputs["pixel_values_videos"]):
                    pv = model_inputs["pixel_values_videos"]
                    pv_shape = f"pixel_values_videos={tuple(pv.shape)} {str(pv.dtype).replace('torch.', '')}"
                else:
                    pv_shape = "pixel_values=none"
                py = sys.version.split()[0]
                torch_info = f"torch={torch.__version__} cuda={torch.version.cuda} is_cuda={torch.cuda.is_available()}"
                gpu_info = "gpu=none"
                if torch.cuda.is_available():
                    gpu_info = f"gpu={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}"
                print(f"[PerfDiag] python={py} {torch_info}")
                print(f"[PerfDiag] {gpu_info} {sdp_info}")
                print(f"[PerfDiag] model_device={device_for_log} model_dtype={model_dtype} attn={self.current_attn_implementation}")
                print(f"[PerfDiag] {pv_shape} max_side={max_long_side}")
                print(f"[PerfDiag] max_new_tokens={max_tokens} num_beams={num_beams} do_sample={num_beams<=1}")
            except Exception:
                pass

        # Determine the prompt to use
        prompt_text = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt_text = custom_prompt.strip()

        # Apply Qwen3VL extra options to build enhanced prompt (if connected)
        if extra_options:
            try:
                import qwen3vl_extra_options
                prompt_text = qwen3vl_extra_options.Qwen3VL_ExtraOptions.build_enhanced_prompt(prompt_text, extra_options)
                print("✅ Qwen3VL Extra Options applied to prompt")
            except (ImportError, AttributeError) as e:
                print(f"⚠️ Warning: failed to import Qwen3VL extra options module ({e}); using base prompt")

        # Build conversation messages
        conversation = [{"role": "user", "content": []}]

        # Add images
        for image in [image1, image2, image3, image4]:
            if image is not None:
                conversation[0]["content"].append({
                    "type": "image",
                    "image": self.image_processor.to_pil(image, max_long_side)
                })

        # Add video (as a multi-frame image sequence)
        if video is not None:
            raw_frames = [
                Image.fromarray((frame.cpu().numpy() * 255).astype(np.uint8))
                for frame in video
            ]

            # Sample video frames
            if len(raw_frames) > video_frames:
                indices = np.linspace(0, len(raw_frames) - 1, video_frames, dtype=int)
                sampled_frames = [raw_frames[i] for i in indices]
            else:
                sampled_frames = raw_frames

            # Ensure at least 2 frames (Qwen3-VL requirement)
            if sampled_frames and len(sampled_frames) == 1:
                sampled_frames.append(sampled_frames[0])

            if sampled_frames:
                conversation[0]["content"].append({
                    "type": "video",
                    "video": [self.image_processor.to_pil(f, max_long_side) for f in sampled_frames]
                })

        # Add text prompt
        conversation[0]["content"].append({
            "type": "text",
            "text": prompt_text
        })

        # Apply chat template
        text_prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Extract images and video for the processor
        pil_images = [
            item['image'] for item in conversation[0]['content']
            if item['type'] == 'image'
        ]
        video_frames_list = [
            frame for item in conversation[0]['content']
            if item['type'] == 'video'
            for frame in item['video']
        ]
        videos_arg = [video_frames_list] if video_frames_list else None

        # Process inputs
        inputs = self.processor(
            text=text_prompt,
            images=pil_images if pil_images else None,
            videos=videos_arg,
            return_tensors="pt"
        )

        # Move inputs to device
        model_inputs = {
            k: v.to(effective_device)
            for k, v in inputs.items()
            if torch.is_tensor(v)
        }

        # Set stop tokens
        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, 'eot_id'):
            stop_tokens.append(self.tokenizer.eot_id)

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id
        }

        if num_beams > 1:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p
            })

        # Generate text
        gen_start = time.time()
        if "cuda" in str(effective_device) and torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.current_attn_implementation == "sdpa" and "cuda" in str(effective_device) and torch.cuda.is_available() and hasattr(torch.backends.cuda, "sdp_kernel"):
            with torch.inference_mode(), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
                outputs = self.model.generate(**model_inputs, **gen_kwargs, use_cache=True)
        else:
            with torch.inference_mode():
                outputs = self.model.generate(**model_inputs, **gen_kwargs, use_cache=True)
        if "cuda" in str(effective_device) and torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time = time.time() - gen_start

        input_ids_len = model_inputs["input_ids"].shape[1]
        text = self.tokenizer.decode(
            outputs[0, input_ids_len:],
            skip_special_tokens=True
        )

        total_time = time.time() - start_time
        generated_tokens = int(outputs.shape[1] - input_ids_len) if hasattr(outputs, "shape") else 0
        tps = (generated_tokens / gen_time) if gen_time > 0 else 0.0
        print(f"⏱️ Timing: device {device_for_log} | attn {self.current_attn_implementation} | in {input_ids_len} | out {generated_tokens} | load {load_time:.2f}s | gen {gen_time:.2f}s ({tps:.1f} tok/s) | total {total_time:.2f}s")

        if not keep_model_loaded:
            self.clear_model_resources()
        return (text.strip(),)


class Qwen3VL_Chat:
    """Qwen3-VL chat node - multimodal LLM conversation."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model_name = None
        self.current_quantization = None
        self.current_device = None
        self.current_attn_implementation = None
        self.model_device = None
        self.last_seed = -1
        self.device_info = get_device_info()
        self.downloader = ModelDownloader(MODEL_CONFIGS)
        self.image_processor = ImageProcessor()
        print(f"Qwen3VL chat node initialized. Device: {self.device_info['device_type']}")
        if not self.device_info["memory_sufficient"]:
            print(f"Warning: {self.device_info['warning_message']}")

    def clear_model_resources(self):
        """Release model resources."""
        if self.model is not None:
            print("Releasing model resources...")
            del self.model, self.processor, self.tokenizer
            self.model = self.processor = self.tokenizer = None
            self.current_model_name = self.current_quantization = self.current_device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_model(self, model_name: str, quantization_str: str, device: str = "auto", attn_mode: str = "SDPA"):
        """Load model."""
        self.device_info = get_device_info()
        effective_device = self.device_info["recommended_device"] if device == "auto" else device
        attn_implementation = resolve_attn_implementation(attn_mode)
        
        # 如果模型已加载且配置相同，则跳过
        if (self.model is not None and 
            self.current_model_name == model_name and 
            self.current_quantization == quantization_str and 
            self.current_device == effective_device and
            self.current_attn_implementation == attn_implementation):
            return

        self.clear_model_resources()

        model_info = get_model_info(model_name)
        
        # Warn for abliterated models
        if model_info.get("abliterated"):
            warning_msg = model_info.get("warning", "This model has safety filters removed")
            print(f"\n⚠️  Warning: {warning_msg}\n")
        
        # Check GPU compute capability for FP8 quantized models
        if model_info.get("quantized"):
            if self.device_info["gpu"]["available"]:
                major, minor = torch.cuda.get_device_capability()
                cc = major + minor / 10
                if cc < 8.9:
                    raise ValueError(
                        f"FP8 models require a GPU with compute capability 8.9 or higher (e.g., RTX 4090)."
                        f"Your GPU compute capability is {cc}. Please choose a non-FP8 model."
                    )

        model_path = self.downloader.ensure_model_available(model_name)
        adjusted_quantization = check_memory_requirements(model_name, quantization_str, self.device_info)
        
        quant_config, load_dtype = None, torch.float16
        
        # Only apply quantization config for non-pre-quantized models
        if not get_model_info(model_name).get("quantized", False):
            if adjusted_quantization == Quantization.Q4_BIT:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                load_dtype = None
            elif adjusted_quantization == Quantization.Q8_BIT:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                load_dtype = None

        device_map = "auto"
        if effective_device == "cuda" and torch.cuda.is_available():
            device_map = {"": 0}

        # Build model load kwargs
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": load_dtype,
            "attn_implementation": attn_implementation,
            "use_safetensors": True,
            "trust_remote_code": True
        }
        
        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        print(f"Loading model '{model_name}'...")
        # Load model, processor, and tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.current_model_name = model_name
        self.current_quantization = quantization_str
        self.current_device = effective_device
        self.current_attn_implementation = attn_implementation
        try:
            self.model_device = str(next(self.model.parameters()).device)
        except StopIteration:
            self.model_device = effective_device
        print("Model loaded successfully")

    @classmethod
    def INPUT_TYPES(cls):
        """Define chat node input types."""
        model_names = [name for name in MODEL_CONFIGS.keys() if not name.startswith('_')]
        default_model = model_names[4] if len(model_names) > 4 else model_names[0]

        return {
            "required": {
                "🤖 Model": (model_names, {"default": default_model}),
                "⚙️ Quantization": (list(Quantization.get_values()), {"default": Quantization.NONE}),
                "🧠 Attention Mode": (["SDPA", "Flash Attention 2"], {"default": "SDPA"}),
                "🖼️ Max Long Side": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "💬 User Input": ("STRING", {
                    "default": "Hi! Please introduce yourself.",
                    "multiline": True,
                    "placeholder": "Type what you want to say"
                }),
                "🎭 System Role": ("STRING", {
                    "default": "You are a professional, friendly, and helpful AI assistant.",
                    "multiline": True,
                    "placeholder": "Define the assistant's role and behavior"
                }),
                "🌡️ Temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "🎯 Top-P": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "📏 Max Length": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16, "tooltip": "Max new tokens to generate; larger values are slower."}),
                "🎲 Seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (-1 means random)."
                }),
                "🎯 Seed Mode": (["Random", "Fixed", "Increment"], {"default": "Random"}),
                "🚀 Enable TF32": ("BOOLEAN", {"default": False, "tooltip": "Enable TF32 acceleration (Ampere+ GPUs only; can significantly improve speed)."}),
                "🔄 Keep Model Loaded": ("BOOLEAN", {"default": False}),
                "🧪 Performance Diagnostics": ("BOOLEAN", {"default": False, "tooltip": "Print key environment and inference info once (useful for debugging slow runs)."}),
            },
            "optional": {
                "🖼️ Image 1": ("IMAGE",),
                "🖼️ Image 2": ("IMAGE",),
                "🖼️ Image 3": ("IMAGE",),
                "🖼️ Image 4": ("IMAGE",),
                "🎯 Qwen3VL Extra Options": ("QWEN3VL_EXTRA_OPTIONS", {
                    "tooltip": "Optional Qwen3VL extra options (connect the Qwen3VL Extra Options node)."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("AI Reply",)
    FUNCTION = "chat"
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

    @torch.no_grad()
    def chat(self, **kwargs):
        """Chat handler."""
        # Extract parameters
        model_name = kwargs.get("🤖 Model")
        quantization = kwargs.get("⚙️ Quantization")
        attn_mode = kwargs.get("🧠 Attention Mode", "SDPA")
        max_long_side = kwargs.get("🖼️ Max Long Side", 768)
        user_input = kwargs.get("💬 User Input")
        system_role = kwargs.get("🎭 System Role")
        temperature = kwargs.get("🌡️ Temperature")
        max_length = kwargs.get("📏 Max Length")
        seed = kwargs.get("🎲 Seed")
        seed_mode = kwargs.get("🎯 Seed Mode")
        keep_model_loaded = kwargs.get("🔄 Keep Model Loaded", False)
        enable_tf32 = kwargs.get("🚀 Enable TF32", False)
        perf_diag = kwargs.get("🧪 Performance Diagnostics", False)
        image1 = kwargs.get("🖼️ Image 1")
        image2 = kwargs.get("🖼️ Image 2")
        image3 = kwargs.get("🖼️ Image 3")
        image4 = kwargs.get("🖼️ Image 4")
        extra_options = kwargs.get("🎯 Qwen3VL Extra Options", None)
        top_p = kwargs.get("🎯 Top-P", 0.90)

        start_time = time.time()

        # Configure TF32 acceleration
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = enable_tf32
            torch.backends.cudnn.allow_tf32 = enable_tf32
            if enable_tf32:
                print("🚀 TF32 acceleration enabled")

        # Seed logic
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
        print(f"Using random seed: {effective_seed} (mode: {seed_mode})")
        torch.manual_seed(effective_seed)

        # Check transformers version
        if version.parse(transformers.__version__) < version.parse("4.57.0"):
            raise RuntimeError(f"transformers version too low: {transformers.__version__}, requires >= 4.57.0")

        load_start = time.time()
        self.load_model(model_name, quantization, "auto", attn_mode)
        load_time = time.time() - load_start
        effective_device = self.current_device
        device_for_log = self.model_device or effective_device
        if effective_device == "cpu" and torch.cuda.is_available():
            print("⚠️ Running on CPU; ensure torch is built with CUDA support.")
        if perf_diag:
            try:
                import sys
                model_dtype = None
                try:
                    model_dtype = str(next(self.model.parameters()).dtype)
                except StopIteration:
                    model_dtype = "unknown"
                try:
                    sdp_info = f"sdp_flash={torch.backends.cuda.flash_sdp_enabled()} sdp_mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} sdp_math={torch.backends.cuda.math_sdp_enabled()}"
                except Exception:
                    sdp_info = "sdp_info=unknown"
                pv_shape = None
                if "pixel_values" in model_inputs and torch.is_tensor(model_inputs["pixel_values"]):
                    pv = model_inputs["pixel_values"]
                    pv_shape = f"pixel_values={tuple(pv.shape)} {str(pv.dtype).replace('torch.', '')}"
                else:
                    pv_shape = "pixel_values=none"
                py = sys.version.split()[0]
                torch_info = f"torch={torch.__version__} cuda={torch.version.cuda} is_cuda={torch.cuda.is_available()}"
                gpu_info = "gpu=none"
                if torch.cuda.is_available():
                    gpu_info = f"gpu={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}"
                print(f"[PerfDiag] python={py} {torch_info}")
                print(f"[PerfDiag] {gpu_info} {sdp_info}")
                print(f"[PerfDiag] model_device={device_for_log} model_dtype={model_dtype} attn={self.current_attn_implementation}")
                print(f"[PerfDiag] {pv_shape} max_side={max_long_side}")
                print(f"[PerfDiag] max_new_tokens={max_length} do_sample=True")
            except Exception:
                pass

        # Process system role, apply extra options
        system_prompt = system_role.strip() if system_role else ""

        # Apply Qwen3VL extra options to enhance system prompt (if connected)
        if extra_options and system_prompt:
            try:
                import qwen3vl_extra_options
                system_prompt = qwen3vl_extra_options.Qwen3VL_ExtraOptions.build_enhanced_prompt(system_prompt, extra_options)
                print("✅ Qwen3VL Extra Options applied to system role")
            except (ImportError, AttributeError) as e:
                print(f"⚠️ Warning: failed to import Qwen3VL extra options module ({e}); using base system role")

        # Build conversation, prepend system role
        conversation = []

        # Add system role (if provided)
        if system_prompt:
            conversation.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

        # Add user message
        user_content = []

        # Add images
        for image in [image1, image2, image3, image4]:
            if image is not None:
                user_content.append({
                    "type": "image",
                    "image": self.image_processor.to_pil(image, max_long_side)
                })

        # Add user text input
        user_content.append({
            "type": "text",
            "text": user_input
        })

        conversation.append({
            "role": "user",
            "content": user_content
        })

        # Apply chat template
        text_prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Extract images for the processor
        pil_images = []
        for msg in conversation:
            if msg['role'] == 'user':
                pil_images.extend([
                    item['image'] for item in msg['content']
                    if item['type'] == 'image'
                ])

        # Process inputs
        inputs = self.processor(
            text=text_prompt,
            images=pil_images if pil_images else None,
            return_tensors="pt"
        )

        # Move inputs to device
        model_inputs = {
            k: v.to(effective_device)
            for k, v in inputs.items()
            if torch.is_tensor(v)
        }

        # Set stop tokens
        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, 'eot_id'):
            stop_tokens.append(self.tokenizer.eot_id)

        # Log unrecognized parameters
        remaining_kwargs = {k: v for k, v in kwargs.items() if not k.startswith(('🤖', '⚙️', '🧠', '🧪', '💬', '🎭', '🌡️', '🎯', '📏', '🎲', '🎮', '🔄', '🖼️', '🚀'))}
        if remaining_kwargs:
            print(f"[Qwen3VL_Chat] Unrecognized parameters ignored: {', '.join(remaining_kwargs.keys())}")

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_length,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id
        }

        # Generate text
        gen_start = time.time()
        if "cuda" in str(effective_device) and torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.current_attn_implementation == "sdpa" and "cuda" in str(effective_device) and torch.cuda.is_available() and hasattr(torch.backends.cuda, "sdp_kernel"):
            with torch.inference_mode(), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
                outputs = self.model.generate(**model_inputs, **gen_kwargs, use_cache=True)
        else:
            with torch.inference_mode():
                outputs = self.model.generate(**model_inputs, **gen_kwargs, use_cache=True)
        if "cuda" in str(effective_device) and torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time = time.time() - gen_start

        input_ids_len = model_inputs["input_ids"].shape[1]
        text = self.tokenizer.decode(
            outputs[0, input_ids_len:],
            skip_special_tokens=True
        )

        total_time = time.time() - start_time
        generated_tokens = int(outputs.shape[1] - input_ids_len) if hasattr(outputs, "shape") else 0
        tps = (generated_tokens / gen_time) if gen_time > 0 else 0.0
        print(f"⏱️ Timing: device {device_for_log} | attn {self.current_attn_implementation} | in {input_ids_len} | out {generated_tokens} | load {load_time:.2f}s | gen {gen_time:.2f}s ({tps:.1f} tok/s) | total {total_time:.2f}s")

        if not keep_model_loaded:
            self.clear_model_resources()
        return (text.strip(),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "Qwen3VL_Advanced": Qwen3VL_Advanced,
    "Qwen3VL_Chat": Qwen3VL_Chat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL_Advanced": "Qwen3VL-DP",
    "Qwen3VL_Chat": "Qwen3VL-DP Chat",
}
