"""模型下载脚本。

用法:
    python scripts/download.py
"""

from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-0.6B",
    cache_dir="/usr/local/app/volume/sft_distill_mil/models",
)
print(f"Model downloaded to: {model_dir}")
