# Extra Options Integration Feature Usage Examples

## Feature Description
Now all main nodes (Qwen3VL_Advanced and Qwen3VL_Chat) support connecting to the "Qwen3VL Extra Options" node for more flexible prompt enhancement.

## Node Connection Examples

### Example 1: Basic Image Captioning + Extra Options

```
┌─────────────────────────┐
│  Qwen3VL Extra Options  │
│  ✓ Include Lighting Info│
│  ✓ Include Camera Angle │
│  ✓ Include Art Quality  │
└────────┬────────────────┘
         │ (Output: QWEN3VL_EXTRA_OPTIONS)
         │
         ↓
┌────────┴────────────────┐
│  Qwen3VL Main Node      │
│  Preset Prompt: Detailed│
│  🎯 Qwen3VL Extra Options ← Connect│
└─────────────────────────┘
```

**Effect**:
- Base prompt is automatically enhanced
- Generated description will include lighting, angle, and art quality information
- No need to manually modify the prompt

---

### Example 2: Intelligent Chat + Extra Options

```
┌─────────────────────────┐
│  Qwen3VL Extra Options  │
│  ✓ Include Camera Details│
│  ✓ Include Composition  │
│  ✓ Don't Use Vague Lang │
└────────┬────────────────┘
         │
         ↓
┌────────┴────────────────┐
│  Qwen3VL Chat Node      │
│  System Role: Photo Analyst│
│  🎯 Qwen3VL Extra Options ← Connect│
└─────────────────────────┘
```

**Effect**:
- AI assistant will analyze images from a professional photographer's perspective
- Automatically includes camera parameters and composition analysis
- Uses precise professional terminology

---

### Example 3: Batch Captioning (Existing Feature)

```
┌─────────────────────────┐
│  Qwen3VL Extra Options  │
│  ✓ Exclude Suggestive   │
│  ✓ Don't Mention Text   │
│  ✓ Describe Important   │
└────────┬────────────────┘
         │
         ↓
┌────────┴────────────────┐
│  Qwen3VL Batch Caption  │
│  Input Folder: ./images │
│  🎯 Qwen3VL Extra Options ← Connect│
└─────────────────────────┘
```

**Effect**:
- Apply unified extra options during batch processing
- All image descriptions follow the same rules
- Suitable for large-scale dataset processing

---

### Example 4: Multiple Nodes Sharing Extra Options

```
                    ┌─────────────────────────┐
                    │  Qwen3VL Extra Options  │
                    │  ✓ Include Lighting Info│
                    │  ✓ Include Art Quality  │
                    └────────┬────────────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ↓            ↓            ↓
        ┌───────────┐ ┌───────────┐ ┌───────────┐
        │ Main Node1│ │ Main Node2│ │ Main Node3│
        │ Image 1   │ │ Image 2   │ │ Image 3   │
        └───────────┘ └───────────┘ └───────────┘
```

**Advantages**:
- Configure once, use everywhere
- Maintain consistency across all nodes
- Easy to adjust in batch

---

## Usage Steps

### Step 1: Add Extra Options Node
1. Right-click in ComfyUI → Add Node
2. Select: `🍭Dapao-Qwen3VL` → `🍭Dapao-Qwen3VL Extra Options`
3. Configure desired options (check the corresponding checkboxes)

### Step 2: Connect to Main Node
1. Find the `🎯 Qwen3VL Extra Options` input port on the main node
2. Connect from the extra options node's output port
3. Connection line will display when successfully connected

### Step 3: Normal Usage
1. Select preset prompt or enter custom prompt in the main node
2. Run the workflow
3. Extra options will be applied automatically, no manual operation needed

---

## Prompt Enhancement Effect Examples

### Original Prompt
```
Describe this image in detail
```

### After Enabling Extra Options
```
Describe this image in detail

Please follow these additional requirements:
- Please describe the lighting conditions of the image.
- Please describe the camera angle information.
- Please rate the aesthetic/artistic quality of the image (from very low to very high).
- Please describe the image composition, such as rule of thirds, leading lines, symmetry, etc.
```

---

## FAQ

### Q1: What happens if I don't connect extra options?
**A**: No impact at all, the node will work as before. Extra options are optional.

### Q2: Can I use preset prompts and extra options together?
**A**: Yes! Extra options will be applied to any prompt, including preset and custom prompts.

### Q3: Does the batch captioning node also support this?
**A**: The batch captioning node already has this feature, now the main nodes support it too.

### Q4: Will extra options overwrite my prompt?
**A**: No! Extra options are appended after your prompt, they won't overwrite original content.

### Q5: Can one extra options node connect to multiple main nodes?
**A**: Yes! This helps maintain consistency across multiple nodes.

---

## Best Practices

### 1. Image Captioning Scenario
Recommended options:
- ✓ Include Lighting Info
- ✓ Include Camera Angle
- ✓ Include Art Quality
- ✓ Include Composition Info

### 2. Content Moderation Scenario
Recommended options:
- ✓ Exclude Suggestive Content
- ✓ Include Safety Rating
- ✓ Don't Use Vague Language

### 3. Professional Photography Analysis
Recommended options:
- ✓ Include Camera Details
- ✓ Include Lighting Info
- ✓ Mention Light Sources
- ✓ Include Depth of Field

### 4. Concise Description Scenario
Recommended options:
- ✓ Describe Important Elements
- ✓ Don't Mention Text
- ✓ Don't Mention Resolution
- ✓ Don't Use Vague Language

---

## Technical Details

### Extra Options Data Type
```python
QWEN3VL_EXTRA_OPTIONS = {
    "Include Character Info": bool,
    "Exclude Immutable Features": bool,
    "Include Lighting Info": bool,
    # ... more options
}
```

### Prompt Enhancement Function
```python
def build_enhanced_prompt(base_prompt: str, options: dict) -> str:
    """
    Build enhanced prompt based on extra options
    
    Args:
        base_prompt: Base prompt
        options: Extra options dictionary
        
    Returns:
        Enhanced prompt
    """
    # Automatically generate additional requirements
    # Append to base prompt
    # Return complete enhanced prompt
```

---

## Changelog

### 2024-11-18
- ✅ Added extra options support to Qwen3VL_Advanced node
- ✅ Added extra options support to Qwen3VL_Chat node
- ✅ Implemented automatic prompt enhancement feature
- ✅ Maintained backward compatibility

---

## Summary

The extra options integration feature allows you to:
- 🎯 More flexibly control generated content
- 🔧 Modular configuration for easy management
- 🚀 Improve work efficiency
- 🔄 Maintain consistency
- ✅ Seamless integration without affecting existing workflows

Try this new feature now to make your image descriptions more professional and precise!
