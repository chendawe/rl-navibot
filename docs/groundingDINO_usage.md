这份笔记为你梳理了在离线/内网环境下，**从零安装 GroundingDINO、解决 BERT 依赖、完美避开 Rust 编译器报错**的全过程。
*(注：你上条消息提到的 `transformers 2.6.2` 应该是 `4.6.2` 的笔误，因为 HuggingFace 并没有 2.6.2 版本，且正好对应 BERT 配置文件里的 `4.6.0`。笔记中已为你修正。)*
---
# GroundingDINO 本地离线部署与避坑指南
## 1. 环境准备与基础依赖
首先安装 PyTorch 和基础视觉库（根据你的 CUDA 版本调整）：
```bash
pip install torch torchvision torchaudio
pip install opencv-python matplotlib
```
## 2. 下载模型权重与源码
1. **GroundingDINO 源码**：`git clone https://github.com/IDEA-Research/GroundingDINO.git`
2. **GroundingDINO 权重**：从 GitHub Release 下载 `groundingdino_swint_ogc.pth`，放到 `weights/` 目录。
3. **BERT 模型**：下载 `bert-base-uncased` 的全部文件，放到本地目录（如 `~/workspace/hf/bert-base-uncased`）。
   *(确保该目录下有：`config.json`, `pytorch_model.bin`, `tokenizer.json`, `vocab.txt` 等文件)*
## 3. 修改 GroundingDINO 配置文件（关键！）
打开配置文件（例如 `GroundingDINO_SwinT_OGC.py`），找到 `text_encoder_type`，将其修改为你**本地 BERT 的绝对路径**：
```python
# 原始值：text_encoder_type = "bert-base-uncased"
# 修改为：
text_encoder_type = "/home/chendawww/workspace/hf/bert-base-uncased"
```
## 4. 解决依赖地狱：Transformers 与 Tokenizers（核心踩坑点）
**背景**：
- `transformers 5.x` 版本太新，导致 BERT 模型结构不兼容（报错 `get_head_mask`）。
- 直接 `pip install transformers==4.6.2` 会触发安装 `tokenizers`，而 `tokenizers` 需要 Rust 编译器，在内网/无 Rust 环境下会直接报错 `can't find Rust compiler`。
**完美解决方案（使用 Conda 绕过编译）**：
```bash
# 1. 先用 conda 安装预编译好的 tokenizers 0.10.3（完美避开 Rust 编译！）
conda install tokenizers=0.10.3
# 2. 再安装匹配 BERT 4.6.0 的 transformers 4.6.2
pip install transformers==4.6.2
```
## 5. 安装 GroundingDINO
依赖解决后，进入 GroundingDINO 源码目录进行安装：
```bash
cd ~/workspace/GroundingDINO
pip install -e .
```
---
## 6. 推理测试代码（直接复制可用）
在 Jupyter Notebook 或 Python 脚本中运行以下代码。**注意：必须在最开头设置离线环境变量！**
```python
import os
# ⚠️ 极度重要：必须在 import transformers 之前设置，强制走本地，不联网
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
import matplotlib.pyplot as plt
from groundingdino.util.inference import load_model, predict, annotate as gd_annotate, load_image
# ================= 配置区 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GD_CONFIG = "/home/chendawww/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_CHECKPOINT = "/home/chendawww/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth"
TEXT_PROMPT = "chair . person . cup . laptop ."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
# 替换为你本地的测试图片路径列表
local_paths = ["./test1.jpg", "./test2.jpg"] 
# ==========================================
print(f"使用设备: {DEVICE}")
print("加载 Grounding DINO 模型...")
gd_model = load_model(GD_CONFIG, GD_CHECKPOINT, device=DEVICE)
print("模型加载成功！\n")
def detect_grounding_dino(image_path):
    image_source, image_tensor = load_image(image_path)
    
    boxes, logits, phrases = predict(
        model=gd_model,
        image=image_tensor,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )
    
    annotated_frame = gd_annotate(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
    )
    return annotated_frame, boxes, logits, phrases
# 批量检测并显示
for path in local_paths:
    if not os.path.exists(path): continue
        
    print(f"正在检测: {os.path.basename(path)}")
    annotated_frame, boxes, logits, phrases = detect_grounding_dino(path)
    
    n = len(boxes) if boxes is not None else 0
    print(f"  -> 检测到 {n} 个目标")
    
    if n > 0:
        for phrase, score in zip(phrases, logits):
            print(f"     - {phrase.strip('. ')} (置信度: {score:.2f})")
        
        plt.figure(figsize=(10, 6))
        plt.imshow(annotated_frame)
        plt.title(f"Result: {os.path.basename(path)}")
        plt.axis("off")
        plt.show()
    print("-" * 40)
```
### 💡 总结避坑逻辑链：
1. 报 `get_head_mask` 错 -> 说明 `transformers 5.x` 破坏了旧版 BERT 的加载逻辑。
2. 降级到 `4.6.x` -> 触发 `pip` 安装旧版 `tokenizers`，要求编译 Rust。
3. 报 `can't find Rust` -> 无法编译。
4. **最终解法** -> 用 `conda` 直接装预编译二进制包 `tokenizers=0.10.3`，再装 `transformers=4.6.2`，逻辑完美闭环。
