### 1. 节点开发规范
- 所有节点必须包含完整的类定义，包括 `INPUT_TYPES`、`RETURN_TYPES`、`RETURN_NAMES`、`FUNCTION`、`CATEGORY`
- 节点分类统一使用 `CATEGORY = "🍭大炮-Qwen3VL`
- 必须在文件末尾注册节点到 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS`

### 2. 图像处理规范
- 输入图像的张量形状：`[B, H, W, C]`（批次、高度、宽度、通道）
- 输入遮罩的张量形状：`[B, H, W]`
- 确保图像值范围在 0-1 之间（float32）
- 使用 `pil2tensor()` 和 `tensor2pil()` 进行格式转换

### 3、路径与文件管理规范（核心重构 - 解决模型加载 Bug）
- 模型路径：严禁使用绝对路径。所有模型（Checkpoint, VAE, Lora）的加载与下载路径，必须通过 ComfyUI 的 folder_paths 模块动态获取
- 资源引用：插件内文件（配置、图标）需使用相对路径锚定。
- 下载逻辑：目标路径不可硬编码，下载前必须校验文件是否存在。 