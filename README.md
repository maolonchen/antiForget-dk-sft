<div align="center">

# SFT-Distill-MIL

**Block Expansion + Knowledge Distillation for LLM Fine-tuning**

Mitigating catastrophic forgetting during Qwen3 fine-tuning via block expansion and knowledge distillation.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-5.5%2B-yellow.svg)](https://huggingface.co/docs/transformers)

</div>

---

## Why Block Expansion?

A core challenge in fine-tuning large models is **catastrophic forgetting** — learning new tasks overwrites old knowledge. Traditional approaches (L2 regularization, EWC, etc.) try to constrain parameters, but their effectiveness is limited in billion-parameter spaces.

Block Expansion takes a different approach:

> **Instead of constraining old parameters, insert new ones.** New knowledge gets its own storage space, and old knowledge is naturally preserved.

The idea is simple — insert **identity blocks** between Transformer layers. Initially, identity blocks are transparent (input = output), so the expanded model behaves identically to the original. During training, identity blocks gradually grow from zero into meaningful layers.

```
 Original (28 layers)              Expanded (42 layers, second_half)
 ──────────────────────            ──────────────────────────────────
 Layer 0                           Layer 0  (original)
 Layer 1                           Layer 1  (original)
 ...                               ...
 Layer 13                          Layer 13 (original)
                                   Layer 14 (original)
 Layer 14                   ──→    [ID-14]  (identity block) ← new!
 Layer 15                          Layer 15 (original)
 ...                               [ID-15]  (identity block) ← new!
 Layer 27                          ...
                                   Layer 27 (original)
                                   [ID-27]  (identity block) ← new!
```

### How Identity Blocks Work

Qwen3's DecoderLayer uses a Pre-Norm residual structure. By zeroing out the weights of `o_proj` (the last layer in attention) and `down_proj` (the last layer in MLP):

```python
# Before zeroing: normal Transformer layer
x = residual + Attention(x)   # o_proj output is non-zero
x = residual + MLP(x)         # down_proj output is non-zero

# After zeroing: exact identity mapping
x = residual + 0 = residual   # attention branch is zero, residual passes through
x = residual + 0 = residual   # MLP branch is zero, residual passes through
```

During training, gradients update these zero-initialized weights, and identity blocks gradually learn new feature transformations.

---

## Triple Loss Design

Block expansion alone is not enough — original layer weights can still be indirectly perturbed by gradients from the new task. Therefore, a triple loss is introduced:

$$
\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_{kl} \cdot \mathcal{L}_{kl} + \lambda_{feat} \cdot \mathcal{L}_{feat}
$$

| Loss | Formula | Purpose |
|------|---------|---------|
| **Task** | Cross-Entropy on labels | Learn the new task |
| **KL Distillation** | $\tau^2 \cdot KL(\text{softmax}(S/\tau) \;\|\|\; \text{softmax}(T/\tau))$ | Match Student output distribution to Teacher |
| **Feature Distillation** | $\frac{1}{N}\sum_i \text{MSE}(S_{hidden[i]},\; T_{hidden[i]})$ | Align intermediate representations layer by layer |

- **KL distillation** protects output-level knowledge (syntax, commonsense, reasoning preferences)
- **Feature distillation** protects internal representation-level knowledge (feature space structure)
- Together they cover the full knowledge chain from bottom layers to output

<details>
<summary>Architecture Diagram</summary>

```
┌─────────────────────────────────────────────────────────┐
│                  BlockExpansionWrapper                  │
│                                                         │
│  ┌──────────────────┐     ┌──────────────────────────┐  │
│  │ Teacher (frozen) │     │     Student (trainable)  │  │
│  │  ┌──────────────┐│     │  ┌──────────────────────┐│  │
│  │  │ Layer 0      ││     │  │ Layer 0 (original)   ││  │
│  │  │ Layer 1      ││     │  │ Layer 1 (original)   ││  │
│  │  │ ...          ││     │  │ ...                  ││  │
│  │  │ Layer 14     ││────→│  │ Layer 14 (original)  ││  │
│  │  │ Layer 15     ││ MSE │  │ [ID-14] (identity)   ││  │
│  │  │ ...          ││     │  │ Layer 15 (original)  ││  │
│  │  │ Layer 27     ││     │  │ [ID-15] (identity)   ││  │
│  │  └──────┬───────┘│     │  │ ...                  ││  │
│  │         │ logits │     │  │ Layer 27 (original)  ││  │
│  │         ↓        │     │  │ [ID-27] (identity)   ││  │
│  │    KL div  ←─────│────→│  └──────────┬───────────┘│  │
│  └──────────────────┘     │             │ logits     │  │
│                           │             ↓            │  │
│                           │      CE Loss (labels)    │  │
│                           └──────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

</details>

---

## Getting Started

### Installation

```bash
# Clone & install
git clone <repo-url>
cd sft_distill_mil
pip install -r requirements.txt

# Or with uv
uv sync
```

Dependencies: `torch>=2.11`, `transformers>=5.5`, `modelscope>=1.36`

### Download Model

```bash
python dl.py  # Downloads Qwen3-0.6B to models/
```

### Prepare Dataset

Training data uses **JSONL** format (one JSON object per line). Three formats are supported:

**Format 1 — Messages (Recommended, Qwen3 ChatML compatible)**

```jsonl
{"messages": [{"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is a subfield of AI..."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Write a poem about spring"}, {"role": "assistant", "content": "Spring breeze brushes the face..."}]}
```

> All formats are ultimately converted to Qwen3's ChatML format (`<|im_start|>user\n...<|im_end|>`) via `tokenizer.apply_chat_template()`.

**Format 2 — Instruction-Response (backward compatible)**

```jsonl
{"instruction": "Translate the following sentence to English", "output": "Hello, how are you today?"}
{"instruction": "Summarize the main idea of the article", "output": "This article mainly discusses the prospects of AI in healthcare."}
```

> Automatically converted to `[{"role": "user", "content": instruction}, {"role": "assistant", "content": output}]`. An optional `"system"` field is also supported.

**Format 3 — Plain Text (backward compatible)**

```jsonl
{"text": "Artificial Intelligence (AI) is a branch of computer science..."}
{"text": "In recent years, large language models (LLMs) have achieved breakthrough progress..."}
```

> Converted to `[{"role": "user", "content": text}]`.

<details>
<summary>Sample Files</summary>

Four sample files are provided in the `data/` directory:

| File | Format |
|------|--------|
| `example_messages_with_system.jsonl` | Messages with system prompt |
| `example_messages_without_system.jsonl` | Messages without system prompt |
| `example_instruction_response.jsonl` | Instruction-Response |
| `example_plain_text.jsonl` | Plain text |

</details>

### Train

```bash
# Quick start (default: second_half strategy)
python scripts/train.py \
    --model_path models/Qwen/Qwen3-0.6B \
    --data_path data/example_messages_with_system.jsonl

# Every layer gets an identity block
python scripts/train.py \
    --model_path models/Qwen/Qwen3-0.6B \
    --data_path data/example_messages_with_system.jsonl \
    --strategy every_layer

# Insert every 4 layers
python scripts/train.py \
    --model_path models/Qwen/Qwen3-0.6B \
    --data_path data/example_messages_with_system.jsonl \
    --strategy every_n --strategy_n 4

# Custom positions
python scripts/train.py \
    --model_path models/Qwen/Qwen3-0.6B \
    --data_path data/example_messages_with_system.jsonl \
    --strategy custom --strategy_positions "0,13,27"
```

### Use in Code

```python
from sft_distill_mil import BlockExpansionWrapper

wrapper = BlockExpansionWrapper(
    model_path="models/Qwen/Qwen3-0.6B",
    strategy="second_half",
    temperature=2.0,
    lambda_kl=0.5,
    lambda_feat=0.1,
)

losses = wrapper(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
)

losses["total_loss"].backward()
```

---

## Insertion Strategies

| Strategy | Description | 0.6B (28L) | 32B (64L) | When to Use |
|----------|-------------|------------|-----------|-------------|
| `second_half` | Insert after every layer in the second half | 28→42 | 64→96 | Default; high-level features benefit most |
| `every_layer` | Insert after every layer | 28→56 | 64→128 | Maximum capacity growth |
| `every_n` | Insert every N layers | +14 (n=2) | +32 (n=2) | Balance growth & efficiency |
| `first_half` | Insert after every layer in the first half | 28→42 | 64→96 | Modify low-level representations |
| `custom` | Specify exact layer indices | custom | custom | Full control |

**General layer mapping formula:**

$$\mathrm{student\_idx}(i) = i + |\{p \in P : p < i\}|$$

where $P$ is the set of insert positions. This formula applies to any model size and strategy.

---

## All Parameters

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | `models/Qwen/Qwen3-0.6B` | Path to pretrained model |
| `--data_path` | *(required)* | JSONL training data path |
| `--output_dir` | `output` | Output directory |
| `--epochs` | `3` | Number of epochs |
| `--batch_size` | `4` | Batch size |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation steps |
| `--lr` | `2e-5` | Learning rate |
| `--weight_decay` | `0.01` | Weight decay |
| `--warmup_ratio` | `0.1` | Warmup ratio |
| `--max_seq_length` | `512` | Max sequence length |

| `--train_on_responses_only` | `False` | (Flag) Only compute loss on the assistant's responses (Standard SFT practice) |

### Strategy

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--strategy` | `second_half` | Insertion strategy |
| `--strategy_n` | `2` | Interval N for `every_n` |
| `--strategy_positions` | `None` | Comma-separated indices for `custom` |

### Distillation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | `2.0` | Softmax temperature τ for KL distillation |
| `--lambda_kl` | `0.5` | Weight for KL divergence loss |
| `--lambda_feat` | `0.1` | Weight for feature distillation loss |

### Logging & Checkpoints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--log_interval` | `10` | Logging interval (steps) |
| `--save_interval` | `500` | Checkpoint save interval (steps) |

---

## Notes

### GPU Memory

Both Teacher and Student must be loaded simultaneously. Estimated memory for Qwen3-0.6B:

| Component | Memory (bf16) |
|-----------|---------------|
| Teacher (28L, frozen) | ~1.2 GB |
| Student (42L, with grads) | ~3.6 GB |
| Optimizer (AdamW) | ~3.6 GB |
| Activations & temp | ~2 GB |
| **Total** | **~10 GB** |

For larger models (e.g. Qwen3-32B), use DeepSpeed ZeRO or FSDP for distributed training.

### Hyperparameter Tuning Tips

| Parameter | Range | Too Low | Too High |
|-----------|-------|---------|----------|
| `temperature` | 1.0 – 4.0 | Soft labels → one-hot, distillation fails | Probabilities flatten, gradients vanish |
| `lambda_kl` | 0.1 – 1.0 | Insufficient forgetting protection | Hinders new task learning |
| `lambda_feat` | 0.01 – 0.5 | Weak feature alignment | Over-constrains internal representations |
| `lr` | 1e-5 – 5e-5 | Slow convergence | Overwrites original knowledge |

### bfloat16 Precision

Training runs in bfloat16. Due to lower mantissa precision (7 bit vs float32's 23 bit), floating-point errors accumulate in deep layers:

- Student's initial output differs slightly from Teacher (max logits difference ~0.3)
- Feature distillation loss is non-zero at initialization

This is expected and does not affect training. KL divergence and feature loss are computed in float32 for numerical stability.

### Output Structure

```
output/
├── best/                  # Best model (lowest avg epoch loss)
├── final/                 # Final model after all epochs
├── checkpoint-500/        # Intermediate checkpoints
├── checkpoint-1000/
└── ...
```

Each directory contains the Student model weights and tokenizer, loadable via:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("output/best")
```

---

## File Structure

```
sft_distill_mil/
├── src/sft_distill_mil/  # Core package
│   ├── __init__.py
│   ├── model.py          # Insertion strategies, model creation, distillation losses
│   └── trainer.py        # SFTDataset, training loop
├── scripts/              # Entry point scripts
│   ├── train.py          # python scripts/train.py ...
│   └── download.py       # Model download utility
├── tests/                # Unit tests
├── data/                 # Sample datasets
│   ├── example_messages_with_system.jsonl
│   ├── example_messages_without_system.jsonl
│   ├── example_instruction_response.jsonl
│   └── example_plain_text.jsonl
├── models/               # Local model files (gitignored)
│   └── Qwen/Qwen-{...}/
├── pyproject.toml
├── README.md
└── README_CN.md
```

## License

MIT

![Visitor Count](https://komarev.com/ghpvc/?username=maolonchen&color=blue)



useful:
without-think:         python scripts/chat.py --model_path output/best
with-think:            python scripts/chat.py --model_path output/best --think
train-only-addmoudle:  python scripts/train.py --model_path /root/autodl-tmp/models/Qwen/Qwen3-0.6B --data_path data/example_messages_without_system.jsonl --lambda_kl 0 --lambda_feat 0 --epochs 10 --batch_size 2 --gradient_accumulation_steps 1 --lr 2e-4 --warmup_ratio 0.03