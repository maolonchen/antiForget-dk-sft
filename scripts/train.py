"""Block Expansion SFT 训练入口脚本。

用法:
    python scripts/train.py --model_path models/Qwen/Qwen3-0.6B --data_path data/example_messages_with_system.jsonl
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


from src.sft_distill_mil.trainer import train

if __name__ == "__main__":
    train()
