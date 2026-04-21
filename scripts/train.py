"""Block Expansion SFT 训练入口脚本。

用法:
    python scripts/train.py --model_path models/Qwen/Qwen3-0.6B --data_path data/example_messages_with_system.jsonl
"""

from sft_distill_mil.trainer import train

if __name__ == "__main__":
    train()
