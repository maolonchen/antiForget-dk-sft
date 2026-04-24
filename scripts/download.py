"""模型下载脚本。

用法:
    python scripts/download.py
"""

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-0.6B', cache_dir="/root/autodl-tmp/models/")