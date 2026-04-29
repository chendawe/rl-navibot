这里给你整理了一份可以直接复制保存的 Markdown 笔记，结构清晰，把踩过的坑都标出来了：
***
# HuggingFace 国内镜像下载配置笔记
### 1. 核心痛点
在 WSL 或内网服务器中，直接连接 `huggingface.co` 会报错 `[Errno 101] Network is unreachable`，导致模型/配置文件下载失败。
### 2. 终端命令行下载（最推荐）
使用开源镜像站 `hf-mirror.com` 替代官方源。
**⚠️ 避坑提示：** 新版 HuggingFace 工具包中，`huggingface-cli` 已废弃，必须使用 `hf` 命令，否则会报参数不识别的错误。
**操作命令：**
```bash
# 1. 设置镜像环境变量（只对当前终端有效）
export HF_ENDPOINT=https://hf-mirror.com
# 2. 执行下载（以 bert-base-uncased 为例）
hf download google-bert/bert-base-uncased \
  --local-dir /your/path/bert-base-uncased
```
*(注：如果想一行搞定，可以写成 `HF_ENDPOINT=https://hf-mirror.com hf download ...`)*
### 3. Python 脚本内下载（大坑预警）
如果在 Python 代码里下载，**必须在导入 `huggingface_hub` 之前设置环境变量**，否则镜像不生效！
```python
import os
# 【防干扰】如果有开启离线模式的变量，先清掉
for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
    if k in os.environ:
        del os.environ[k]
# 【核心】必须在 import 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 这行必须放在设置环境变量的后面
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="google-bert/bert-base-uncased",
    local_dir="/your/path/bert-base-uncased",
    local_dir_use_symlinks=False  # 建议设为 False，直接存真实文件
)
```
### 4. 下载后的验证
下载完成后，检查目标目录下是否包含完整文件（尤其是 `config.json`）：
```bash
ls /your/path/bert-base-uncased
# 期望输出：config.json  tokenizer_config.json  vocab.txt  pytorch_model.bin ...
```
### 5. 扩展应用：让 Grounding DINO 离线加载 BERT
下载好 BERT 后，为了让 Grounding DINO 不再联网请求 HuggingFace：
1. 复制一份 DINO 的配置文件（如 `GroundingDINO_SwinT_OGC.py`）。
2. 找到 `text_encoder_type` 这一行，将在线 ID 改为本地绝对路径：
   ```python
   # 改前：
   text_encoder_type = "bert-base-uncased"
   
   # 改后：
   text_encoder_type = "/your/path/bert-base-uncased"
   ```
3. 加载模型时指向这个新的配置文件即可彻底离线运行。
