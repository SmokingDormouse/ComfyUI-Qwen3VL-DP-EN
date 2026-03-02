# ComfyUI-Qwen3VL-DP Modification Notes

## Latest Updates (November 18, 2024)

### 🌐 New ModelScope Model Support
- ✅ **Added ModelScope Download Support**: Support for downloading community models from ModelScope
- ✅ **New 4B Uncensored Model**: Huihui-Qwen3-VL-4B-Instruct-Abliterated
  - Source: ModelScope (fireicewolf/Huihui-Qwen3-VL-4B-Instruct-abliterated)
  - VRAM Requirements: 6GB (FP16) / 3.5GB (8-bit) / 2GB (4-bit)
  - Abliterated version based on Qwen3-VL-4B
- ✅ **Automatic Source Selection**: Automatically selects HuggingFace or ModelScope based on model configuration
- ✅ **Dependency Management**: Automatic detection of ModelScope library, provides friendly prompt when not installed

### Technical Implementation
```python
# Support multi-source download
if source == 'modelscope':
    from modelscope.hub.snapshot_download import snapshot_download
    # ModelScope download logic
else:
    from huggingface_hub import snapshot_download
    # HuggingFace download logic
```

---

## Modification Date
November 18, 2024

## Modification Overview
Major revisions to the entire project, mainly involving:
1. Optimization of random seed control and model loading default behavior
2. Main node integration of extra options functionality, supporting more flexible prompt enhancement
3. New ModelScope model download support

## Detailed Modifications

### 1. qwen3vl_node.py - Qwen3VL_Advanced Node

#### Modifications:
- ✅ **Added Seed Control Option**: New `🎮 Seed Control` parameter, selectable "Random" or "Fixed"
  - Default: **Random**
  - Random mode: Uses different timestamp as seed for each run
  - Fixed mode: Uses user-specified random seed value

- ✅ **Modified Keep Model Loaded Default Value**:
  - Original default: `True`
  - New default: **`False`**
  - Purpose: Avoid prolonged VRAM usage

- ✅ **Integrated Extra Options Functionality**: New `🎯 Qwen3VL Extra Options` optional input
  - Can connect to "Qwen3VL Extra Options" node
  - Automatically applies extra options to prompts
  - Supports preset prompts and custom prompts
  - Seamless integration without affecting original functionality

#### Implementation Logic:
```python
# Set random seed based on seed control setting
if seed_control == "Fixed":
    torch.manual_seed(random_seed)
else:
    torch.manual_seed(int(time.time()))

# Apply extra options to enhance prompt
if qwen3vl_extra_options:
    from .qwen3vl_extra_options import Qwen3VL_ExtraOptions
    prompt_text = Qwen3VL_ExtraOptions.build_enhanced_prompt(prompt_text, qwen3vl_extra_options)
```

---

### 2. qwen3vl_node.py - Qwen3VL_Chat Node

#### Modifications:
- ✅ **Modified Keep Model Loaded Default Value**:
  - Original default: `True`
  - New default: **`False`**

- ℹ️ **Seed Control**: This node already has seed control functionality, remains unchanged

- ✅ **Integrated Extra Options Functionality**: New `🎯 Qwen3VL Extra Options` optional input
  - Can connect to "Qwen3VL Extra Options" node
  - Automatically applies extra options to system role definition
  - Enhances AI assistant behavior control
  - Supports detail control for image understanding

---

### 3. qwen3vl_batch_caption.py - Batch Captioning Node

#### Modifications:
- ✅ **Added Seed Control Option**: New `🎮 Seed Control` parameter
  - Default: **Random**

- ✅ **Smart Overwrite Logic**:
  - **Random mode**: Automatically forces overwrite of existing txt files (ensures different results each time)
  - **Fixed mode**: Follows user's "Force Overwrite" setting

#### Implementation Logic:
```python
# If seed control is random, default to force overwrite
if seed_control == "Random":
    force_overwrite = True

# Set random seed based on seed control setting
if seed_control == "Fixed":
    torch.manual_seed(random_seed)

# Process each image
for idx, filename in enumerate(image_files):
    # If random mode, set new random seed before each processing
    if seed_control == "Random":
        torch.manual_seed(int(time.time() * 1000) + idx)
```

---

### 4. qwen3vl_compare_caption.py - Comparison Captioning Node

#### Modifications:
- ✅ **Added Seed Control Option**: New `🎮 Seed Control` parameter
  - Default: **Random**

- ✅ **Smart Overwrite Logic**:
  - **Random mode**: Automatically forces overwrite of existing txt files
  - **Fixed mode**: Follows user's "Force Overwrite" setting

#### Implementation Logic:
```python
# If seed control is random, default to force overwrite
if seed_control == "Random":
    force_overwrite = True

# Set random seed based on seed control setting
if seed_control == "Fixed":
    torch.manual_seed(random_seed)

# Process each image pair
for idx, (file_a, file_b) in enumerate(file_pairs):
    # If random mode, set new random seed before each processing
    if seed_control == "Random":
        torch.manual_seed(int(time.time() * 1000) + idx)
```

---

## Feature Summary

### 🎲 Seed Control Feature
1. **Random Mode (Default)**:
   - Generates different results each run
   - During batch processing, each image uses a different seed
   - Automatically overwrites existing caption files

2. **Fixed Mode**:
   - Uses user-specified seed value
   - Can reproduce identical results
   - Follows user's overwrite settings

### 🔄 Model Loading Optimization
- All nodes' "Keep Model Loaded" default changed to `False`
- Avoids prolonged VRAM usage
- Users can manually enable as needed

---

## Usage Recommendations

### Use Cases

#### Use Random Mode (Default):
- ✅ Need diverse description results
- ✅ Want different description styles for each image during batch captioning
- ✅ Exploring different description possibilities

#### Use Fixed Mode:
- ✅ Need reproducible results
- ✅ Debugging and testing
- ✅ Comparing effects of different parameters

---

## Technical Implementation Details

### Random Seed Generation Strategy
- Base random seed: `int(time.time())`
- Batch processing increment: `int(time.time() * 1000) + idx`
- Ensures each image has a different seed

### Smart Overwrite Logic
```python
# Batch captioning and comparison captioning nodes
if seed_control == "Random":
    force_overwrite = True  # Auto overwrite, ensures new results are generated
```

---

## Compatibility Notes
- ✅ Backward compatible: Existing workflows unaffected
- ✅ Default behavior: All nodes default to random mode
- ✅ Flexible configuration: Users can switch to fixed mode at any time

---

## Testing Recommendations
1. Test random mode: Run the same workflow multiple times, verify results are different
2. Test fixed mode: Run multiple times with the same seed, verify results are consistent
3. Test batch captioning: Verify auto-overwrite functionality in random mode
4. Test model loading: Verify default no-keep-loaded, VRAM releases normally

---

## Modified Files List
- ✅ `qwen3vl_node.py`
- ✅ `qwen3vl_batch_caption.py`
- ✅ `qwen3vl_compare_caption.py`

---

## 🎯 Extra Options Integration Feature Details

### Feature Overview
Added `🎯 Qwen3VL Extra Options` optional input to main nodes (Qwen3VL_Advanced and Qwen3VL_Chat), allowing users to connect "Qwen3VL Extra Options" node to enhance prompts.

### How It Works

#### 1. Qwen3VL_Advanced Node
```
User Input (Preset Prompt or Custom Prompt)
    ↓
Apply Extra Options Enhancement
    ↓
Generate Final Prompt
    ↓
Send to Model for Processing
```

**Example**:
- Base prompt: `"Describe this image in detail"`
- Extra options: Enable "Include Lighting Info", "Include Camera Angle"
- Final prompt:
  ```
  Describe this image in detail
  
  Please follow these additional requirements:
  - Please describe the lighting conditions of the image.
  - Please describe the camera angle information.
  ```

#### 2. Qwen3VL_Chat Node
```
System Role Definition
    ↓
Apply Extra Options Enhancement
    ↓
Generate Enhanced System Role
    ↓
Used for Conversation Generation
```

**Example**:
- Base system role: `"You are a professional, friendly, and helpful AI assistant."`
- Extra options: Enable "Include Art Quality", "Include Composition Info"
- Enhanced system role:
  ```
  You are a professional, friendly, and helpful AI assistant.
  
  Please follow these additional requirements:
  - Please evaluate the aesthetic/artistic quality of the image (from very low to very high).
  - Please describe the image composition, such as rule of thirds, leading lines, symmetry, etc.
  ```

### Usage Instructions

1. **Add Extra Options Node**:
   - Add "🍭Dapao-Qwen3VL Extra Options" node to the workflow
   - Configure desired options (such as lighting, camera angle, art quality, etc.)

2. **Connect to Main Node**:
   - Connect the extra options node output to the main node's `🎯 Qwen3VL Extra Options` input

3. **Normal Usage**:
   - Select preset prompt or enter custom prompt
   - Extra options will be applied automatically, no manual prompt modification needed

### Advantages

✅ **Flexibility**:
- Optional connection, doesn't affect original workflow
- Supports preset prompts and custom prompts
- Batch captioning nodes already have this feature, now main nodes support it too

✅ **Modularity**:
- Extra options configured independently
- Same extra options node can be reused
- Easy to manage and adjust

✅ **Smart Integration**:
- Automatically merges prompts
- Maintains integrity of original prompt
- Adds structured additional requirements

✅ **Backward Compatible**:
- Functions exactly the same when extra options not connected
- Existing workflows need no modification

### Application Scenarios

#### Scenario 1: Image Captioning
```
Main Node: Qwen3VL_Advanced
Preset Prompt: Prompt Style - Detailed
Extra Options:
  - Include Lighting Info ✓
  - Include Camera Angle ✓
  - Include Art Quality ✓
  
Result: Generates detailed description including lighting, angle, and art quality evaluation
```

#### Scenario 2: Intelligent Conversation
```
Main Node: Qwen3VL_Chat
System Role: Professional Photography Analysis Assistant
Extra Options:
  - Include Camera Details ✓
  - Include Composition Info ✓
  - Don't Use Vague Language ✓
  
Result: AI assistant analyzes images from a professional photographer's perspective
```

#### Scenario 3: Batch Processing
```
Main Node: Qwen3VL_Advanced
Extra Options Node → Connected to multiple main nodes
  
Advantage: Configure once, use everywhere
```

---

## Notes
All modifications have been completed and verified, ready for use. If you have any questions, please refer to this document or contact the developer.
```