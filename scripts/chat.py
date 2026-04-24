"""与训练后的 Block Expansion 模型对话。

用法:
    python scripts/chat.py --model_path output/final
    python scripts/chat.py --model_path output/best --max_new_tokens 1024
    # 单轮测试（非交互）
    python scripts/chat.py --model_path output/final --prompt "你好，介绍一下自己"
"""
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with the fine-tuned model")
    parser.add_argument("--model_path", type=str, default="output/final")
    parser.add_argument("--system", type=str, default=None, help="可选的 system prompt")
    parser.add_argument("--prompt", type=str, default=None, help="单轮模式：直接传入一个 prompt 并退出")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--greedy", action="store_true", help="使用贪心解码（覆盖采样参数）")
    parser.add_argument("--think", action="store_true", help="开启 Qwen3 思考模式（默认关闭）")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
        })

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    @torch.inference_mode()
    def generate(msgs: list[dict[str, str]]) -> str:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.think,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # 单轮模式
    if args.prompt is not None:
        messages.append({"role": "user", "content": args.prompt})
        reply = generate(messages)
        print(f"\nUser: {args.prompt}")
        print(f"Assistant: {reply}")
        return

    # 多轮交互
    print("\n进入对话模式。命令: /reset 清空历史, /exit 退出。\n")
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/reset":
            messages = [{"role": "system", "content": args.system}] if args.system else []
            print("(对话历史已清空)\n")
            continue

        messages.append({"role": "user", "content": user_input})
        reply = generate(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
