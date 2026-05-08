"""Ruozhiba 原始数据预处理脚本。

输入: data/ruozhiba_raw/ 下三个 JSON 文件
输出: data/ruozhiba/ 下对应的 JSONL 文件（符合本项目数据格式）

处理规则:
  - ruozhiba-post-annual.json:
      只取 content 字段。去掉文本前的标号（三种格式：
        1. "数字、"   例: "151、家电下乡..."
        2. "数字."    例: "151.家电下乡..."
        3. "数字 "    例: "151 家电下乡..."
      ）。结果作为 {"text": ...}（即 assistant 内容）。

  - ruozhiba-title-good.json / ruozhiba-title-norm.json:
      title 与 abs 都非空 -> {"messages": [user=title, assistant=abs]}
      abs 空、title 非空  -> {"text": title}
      title 空、abs 非空  -> {"text": abs}
      两者都空            -> 跳过

用法:
    python scripts/process_ruozhiba.py
    python scripts/process_ruozhiba.py --merge  # 额外输出一个合并文件
"""
import argparse
import json
import re
from pathlib import Path

# 三种标号: "数字、" | "数字." | "数字 "（开头）
NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s*[、.\s]\s*")

# URL 匹配（http/https 开头，到空白或字符串结尾）
URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def strip_number_prefix(text: str) -> str:
    """去掉文本前的序号标号（如果有）。"""
    return NUMBER_PREFIX_RE.sub("", text, count=1).strip()


def remove_urls(text: str) -> str:
    """删除文本中的所有 URL。"""
    return URL_RE.sub("", text).strip()


def is_blank(value) -> bool:
    """None / 空串 / 纯空白 都视为空。"""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def process_post_annual(in_path: Path, out_path: Path) -> int:
    with in_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            content = item.get("content")
            if is_blank(content):
                continue
            cleaned = strip_number_prefix(content)
            if is_blank(cleaned):
                continue
            f.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")
            written += 1
    return written


def process_title_file(in_path: Path, out_path: Path) -> int:
    with in_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            title = item.get("title")
            abs_text = item.get("abs")

            # 清洗 URL
            if abs_text:
                abs_text = remove_urls(abs_text)

            title_blank = is_blank(title)
            abs_blank = is_blank(abs_text)

            if title_blank and abs_blank:
                continue

            if title_blank:
                record = {"text": abs_text.strip()}
            elif abs_blank:
                record = {"text": title.strip()}
            else:
                record = {
                    "messages": [
                        {"role": "user", "content": title.strip()},
                        {"role": "assistant", "content": abs_text.strip()},
                    ]
                }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ruozhiba raw data")
    parser.add_argument("--raw_dir", type=str, default="data/ruozhiba_raw")
    parser.add_argument("--out_dir", type=str, default="data/ruozhiba")
    parser.add_argument("--merge", action="store_true", help="额外生成一个合并的 ruozhiba_all.jsonl")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("ruozhiba-post-annual.json", "post-annual.jsonl", process_post_annual),
        ("ruozhiba-title-good.json", "title-good.jsonl", process_title_file),
        ("ruozhiba-title-norm.json", "title-norm.jsonl", process_title_file),
    ]

    outputs = []
    for in_name, out_name, fn in jobs:
        in_path = raw_dir / in_name
        out_path = out_dir / out_name
        if not in_path.exists():
            print(f"[skip] {in_path} not found")
            continue
        n = fn(in_path, out_path)
        outputs.append(out_path)
        print(f"[ok] {in_name} -> {out_path}  ({n} samples)")

    if args.merge and outputs:
        merge_path = out_dir / "ruozhiba_all.jsonl"
        total = 0
        with merge_path.open("w", encoding="utf-8") as fout:
            for p in outputs:
                with p.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                        total += 1
        print(f"[merged] -> {merge_path}  ({total} samples)")


if __name__ == "__main__":
    main()
