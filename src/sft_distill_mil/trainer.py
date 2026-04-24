"""
Block Expansion 微调训练模块

包含数据集类和训练入口。
"""

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .model import BlockExpansionWrapper


class SFTDataset(Dataset):
    """SFT 数据集，支持 JSONL 格式。

    JSONL 每行格式（推荐 messages 格式，适配 Qwen3 ChatML）:
        {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    向后兼容:
        {"text": "..."}
        {"instruction": "...", "output": "..."}  → 自动转换为 messages 格式
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 512,
        train_on_responses_only: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.train_on_responses_only = train_on_responses_only
        self.samples = self._load_data(data_path)

    def _to_messages(self, item: dict) -> list[dict[str, str]]:
        """将各种数据格式统一转换为 messages 列表。"""
        if "messages" in item:
            return item["messages"]
        if "instruction" in item and "output" in item:
            messages = [{"role": "user", "content": item["instruction"]}]
            if "system" in item:
                messages.insert(0, {"role": "system", "content": item["system"]})
            messages.append({"role": "assistant", "content": item["output"]})
            return messages
        if "text" in item:
            return [{"role": "user", "content": item["text"]}]
        raise ValueError(f"无法识别的数据格式，keys: {list(item.keys())}")

    def _load_data(self, data_path: str) -> list[list[dict[str, str]]]:
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                messages = self._to_messages(item)
                samples.append(messages)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.samples[idx]

        if self.train_on_responses_only:
            # 找到最后一个 assistant 开始的位置
            # 1. 拿到去除最后一个 assistant 内容的 prompt
            prompt_messages = messages[:-1]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # 2. 拿到完整的文本
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            enc_full = self.tokenizer(
                full_text,
                max_length=self.max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            enc_prompt = self.tokenizer(
                prompt_text,
                max_length=self.max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = enc_full["input_ids"].squeeze(0)
            attention_mask = enc_full["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            # 将 prompt 部分的 labels 置为 -100
            # 比较简单的做法：看 prompt_text tokenize 后的长度
            prompt_len = enc_prompt["attention_mask"].sum().item()
            # 为了防止特殊 token (如 bos) 的差异，我们直接用 prompt_len 作为分界线
            labels[:prompt_len] = -100

        else:
            # 原有的逻辑，所有有效 token 都计算 loss
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            enc = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train():
    parser = argparse.ArgumentParser(description="Block Expansion SFT Training")
    parser.add_argument("--model_path", type=str, default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--data_path", type=str, required=True, help="JSONL 数据文件路径")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=512)
    # 插入策略
    parser.add_argument(
        "--strategy", type=str, default="second_half",
        choices=["second_half", "every_layer", "every_n", "first_half", "custom"],
        help="恒等块插入策略",
    )
    parser.add_argument("--strategy_n", type=int, default=2, help="every_n 策略的间隔 N")
    parser.add_argument(
        "--strategy_positions", type=str, default=None,
        help="custom 策略的插入位置，逗号分隔，如 '0,13,27'",
    )
    # 蒸馏参数
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda_kl", type=float, default=0.5)
    parser.add_argument("--lambda_feat", type=float, default=0.1)

    # SFT 参数
    parser.add_argument("--train_on_responses_only", action="store_true", help="如果设置，则仅在 assistant 的回复部分计算 loss（标准的 SFT 做法）")

    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 数据集
    dataset = SFTDataset(args.data_path, tokenizer, args.max_seq_length, args.train_on_responses_only)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}")

    # 构建策略参数
    strategy_kwargs = {}
    if args.strategy == "every_n":
        strategy_kwargs["n"] = args.strategy_n
    elif args.strategy == "custom" and args.strategy_positions:
        strategy_kwargs["positions"] = [int(x) for x in args.strategy_positions.split(",")]

    # 模型
    model = BlockExpansionWrapper(
        model_path=args.model_path,
        strategy=args.strategy,
        strategy_kwargs=strategy_kwargs,
        temperature=args.temperature,
        lambda_kl=args.lambda_kl,
        lambda_feat=args.lambda_feat,
    ).to(device)
    print(f"Strategy: {args.strategy}, Insert after layers: {model.insert_positions}")
    print(f"Student layers: {len(model.student.model.layers)} (original: {len(model.teacher.model.layers)})")

    # 可训练参数统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.1f}%)")

    # Optimizer（只优化 student 的参数）
    optimizer = AdamW(
        model.student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    # 训练循环
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.student.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            losses = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Backward（梯度累积）
            scaled_loss = losses["total_loss"] / args.gradient_accumulation_steps
            scaled_loss.backward()

            epoch_loss += losses["total_loss"].item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 日志
                if global_step % args.log_interval == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch+1}/{args.epochs} | "
                        f"Step {global_step} | "
                        f"Loss: {losses['total_loss'].item():.4f} "
                        f"(task: {losses['task_loss'].item():.4f}, "
                        f"kl: {losses['kl_loss'].item():.4f}, "
                        f"feat: {losses['feat_loss'].item():.4f}) | "
                        f"Avg: {avg_loss:.4f} | LR: {lr:.2e}"
                    )

                # 保存 checkpoint
                if global_step % args.save_interval == 0:
                    ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.student.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"Checkpoint saved to {ckpt_dir}")

        # Epoch 结束
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Avg loss: {avg_epoch_loss:.4f}")

        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_dir = Path(args.output_dir) / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.student.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"Best model saved to {best_dir}")

    # 最终保存
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")
