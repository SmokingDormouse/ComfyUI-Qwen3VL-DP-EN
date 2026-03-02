"""
Qwen3VL Extra Options Node
Used for configuring Qwen3VL's advanced description options, can be connected to batch captioning node

Features:
- Provides all advanced description option configurations for Qwen3VL
- Outputs formatted options dictionary that can be connected to other nodes
- Modular design, maintains simplicity of the main node
- Supports fine-grained control over image description content and style
"""

class Qwen3VL_ExtraOptions:
    """Qwen3VL Extra Options Configuration Node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define input types for Qwen3VL extra options"""
        return {
            "required": {},
            "optional": {
                # Character information control
                "👤 Include Character Info": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If there are people/characters in the image, include relevant information (name, etc.)"
                }),
                "🚫 Exclude Immutable Features": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Don't include immutable character features (such as race, gender), but still include changeable attributes (such as hairstyle)"
                }),
                
                # Technical details
                "💡 Include Lighting Info": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include information about lighting"
                }),
                "📐 Include Camera Angle": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include camera angle information"
                }),
                "📷 Include Camera Details": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If it's a photo, must include camera information and details (such as aperture, shutter speed, ISO, etc.)"
                }),
                "💡 Mention Light Sources": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If applicable, mention artificial or natural light sources that may have been used"
                }),
                
                # Image quality assessment
                "🎨 Include Art Quality": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Must include information about the image's aesthetic/artistic quality, from very low to very high"
                }),
                "📊 Include Composition Info": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include image composition information, such as rule of thirds, leading lines, symmetry, etc."
                }),
                "🌈 Include Depth of Field": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Specify depth of field and whether the background is in focus or blurred"
                }),
                
                # Content filtering
                "🔍 Exclude Suggestive Content": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Don't include any suggestive or provocative content"
                }),
                "📝 Don't Mention Text": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Don't mention any text in the image"
                }),
                "🔇 Don't Mention Resolution": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Don't mention the image resolution"
                }),
                
                # Technical information
                "🏷️ Include Watermark Info": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include information about whether the image has a watermark"
                }),
                "🖼️ Include JPEG Artifacts": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include information about whether the image has JPEG compression artifacts"
                }),
                
                # Description style control
                "🌍 Don't Use Vague Language": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Don't use vague language"
                }),
                "⭐ Describe Important Elements": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only describe the most important elements in the image"
                }),
                "🔒 Include Safety Rating": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include information about whether the image is safe, suggestive, or unsafe"
                }),
            }
        }
    
    RETURN_TYPES = ("QWEN3VL_EXTRA_OPTIONS",)
    RETURN_NAMES = ("Qwen3VL Extra Options",)
    FUNCTION = "create_options"
    CATEGORY = "Qwen3VL-DP"
    
    def create_options(self, **kwargs):
        """
        Create Qwen3VL extra options dictionary
        
        Returns:
            Dictionary containing all options
        """
        # Extract all option parameters
        options = {
            "Include Character Info": kwargs.get("👤 Include Character Info", False),
            "Exclude Immutable Features": kwargs.get("🚫 Exclude Immutable Features", False),
            "Include Lighting Info": kwargs.get("💡 Include Lighting Info", False),
            "Include Camera Angle": kwargs.get("📐 Include Camera Angle", False),
            "Include Camera Details": kwargs.get("📷 Include Camera Details", False),
            "Mention Light Sources": kwargs.get("💡 Mention Light Sources", False),
            "Include Art Quality": kwargs.get("🎨 Include Art Quality", False),
            "Include Composition Info": kwargs.get("📊 Include Composition Info", False),
            "Include Depth of Field": kwargs.get("🌈 Include Depth of Field", False),
            "Exclude Suggestive Content": kwargs.get("🔍 Exclude Suggestive Content", False),
            "Don't Mention Text": kwargs.get("📝 Don't Mention Text", False),
            "Don't Mention Resolution": kwargs.get("🔇 Don't Mention Resolution", False),
            "Include Watermark Info": kwargs.get("🏷️ Include Watermark Info", False),
            "Include JPEG Artifacts": kwargs.get("🖼️ Include JPEG Artifacts", False),
            "Don't Use Vague Language": kwargs.get("🌍 Don't Use Vague Language", False),
            "Describe Important Elements": kwargs.get("⭐ Describe Important Elements", False),
            "Include Safety Rating": kwargs.get("🔒 Include Safety Rating", False),
        }
        
        # Count enabled options
        enabled_count = sum(1 for value in options.values() if value)
        
        print(f"🎯 Qwen3VL Extra Options configuration complete:")
        print(f"   Enabled options count: {enabled_count}")
        if enabled_count > 0:
            enabled_options = [key for key, value in options.items() if value]
            print(f"   Enabled options: {', '.join(enabled_options)}")
        
        return (options,)

    @staticmethod
    def build_enhanced_prompt(base_prompt: str, options: dict) -> str:
        """
        Build enhanced prompt based on Qwen3VL extra options
        
        Args:
            base_prompt: Base prompt
            options: Qwen3VL extra options dictionary
            
        Returns:
            Enhanced prompt
        """
        enhanced_instructions = []
        
        # Add specific instructions based on options
        if options.get("Include Character Info", False):
            enhanced_instructions.append("If there are people/characters in the image, please include relevant information (such as name, etc.).")
        
        if options.get("Exclude Immutable Features", False):
            enhanced_instructions.append("Don't include immutable character features (such as race, gender), but you can include changeable attributes (such as hairstyle).")
        
        if options.get("Include Lighting Info", False):
            enhanced_instructions.append("Please describe the lighting conditions of the image.")
        
        if options.get("Include Camera Angle", False):
            enhanced_instructions.append("Please describe the camera angle information.")
        
        if options.get("Include Camera Details", False):
            enhanced_instructions.append("If it's a photo, please include camera information and detailed parameters (such as aperture, shutter speed, ISO, etc.).")
        
        if options.get("Mention Light Sources", False):
            enhanced_instructions.append("If applicable, please mention artificial or natural light sources that may have been used.")
        
        if options.get("Include Art Quality", False):
            enhanced_instructions.append("Please rate the aesthetic/artistic quality of the image (from very low to very high).")
        
        if options.get("Include Composition Info", False):
            enhanced_instructions.append("Please describe the image composition, such as rule of thirds, leading lines, symmetry, etc.")
        
        if options.get("Include Depth of Field", False):
            enhanced_instructions.append("Please describe the depth of field and whether the background is in focus or blurred.")
        
        if options.get("Exclude Suggestive Content", False):
            enhanced_instructions.append("Don't include any description of suggestive or provocative content.")
        
        if options.get("Don't Mention Text", False):
            enhanced_instructions.append("Don't mention any text content in the image.")
        
        if options.get("Don't Mention Resolution", False):
            enhanced_instructions.append("Don't mention the image resolution.")
        
        if options.get("Include Watermark Info", False):
            enhanced_instructions.append("Please indicate whether the image has a watermark.")
        
        if options.get("Include JPEG Artifacts", False):
            enhanced_instructions.append("Please indicate whether the image has JPEG compression artifacts.")
        
        if options.get("Don't Use Vague Language", False):
            enhanced_instructions.append("Please use specific, accurate language and avoid vague expressions.")
        
        if options.get("Describe Important Elements", False):
            enhanced_instructions.append("Please focus on describing the most important elements in the image.")
        
        if options.get("Include Safety Rating", False):
            enhanced_instructions.append("Please rate whether the image is safe, suggestive, or unsafe.")
        
        # Build final prompt
        if enhanced_instructions:
            instructions_text = "\n".join([f"- {instruction}" for instruction in enhanced_instructions])
            enhanced_prompt = f"{base_prompt}\n\nPlease follow these additional requirements:\n{instructions_text}"
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt


# Node registration
NODE_CLASS_MAPPINGS = {
    "Qwen3VL_ExtraOptions": Qwen3VL_ExtraOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL_ExtraOptions": "Qwen3VL Extra Options",
}
