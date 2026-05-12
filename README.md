<div align="center">

# antiForget-dk-sft

**Anti-Catastrophic Forgetting via Local Distributional Anchoring at Identity Blocks**

Block Expansion provides the architecture. Distributional anchoring provides the protection.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-5.5%2B-yellow.svg)](https://huggingface.co/docs/transformers)

</div>

---

## Core Innovation: Local Distributional Anchoring

Block Expansion (inserting identity-mapping layers into a pretrained model) is a known technique for adding capacity without modifying existing parameters. However, prior work leaves a critical gap: **the inserted blocks can distort the internal representations flowing through frozen layers, causing cascading degradation of the model's original capabilities.**

We introduce **Local Distributional Anchoring** — at every identity block insertion point, we anchor the block's output distribution to its input distribution, creating a self-referential constraint that preserves the model's predictive behavior while allowing new knowledge to be absorbed:

> **The model's own pre-block distribution serves as an in-situ reference.** At each insertion point, the identity block faces a distributional tug-of-war: task loss pushes it toward new knowledge, while local KL divergence and feature distillation anchor it to the original distribution.

```
 At each identity block insertion point:

   Frozen Layer i output (h_in)
          │
          ↓  ┌─────────────────────────────────────────┐
          │  │         Identity Block (trainable)        │
          │  └────────────────────┬────────────────────┘
          │                       │ output (h_out)
          │                       │
          │    ┌──────────────────┤
          │    │  Distributional  │
          │    │    Anchoring     │
          │    │                  │
          │    │  MSE(h_in, h_out)        ← hidden-space anchor
          │    │  KL(P_in ‖ P_out)        ← output-space anchor
          │    │  where P = softmax(lm_head(h) / τ)
          │    └──────────────────┤
          │                       │
          ↓                       ↓
   → Frozen Layer i+1 → ... → Final Output → CE Loss (task learning)
```

This creates a **dual-space constraint** at every insertion point:
- **Hidden space (MSE)**: keeps numerical representations close — structural preservation
- **Output space (KL)**: keeps the predicted token distribution close — behavioral preservation

The two spaces are complementary. MSE is dimension-agnostic and cheap, but blind to which dimensions matter. KL projects through `lm_head` into vocabulary space, automatically focusing on the directions that actually shift predictions. A perturbation that MSE considers negligible but that would flip the model's top prediction is caught by KL — this is the key to preventing **cascading distortion** through subsequent frozen layers.

### Why This Works: The Distributional Tug-of-War

During training, each identity block is pulled by opposing forces:

| Force | Direction | Effect |
|-------|-----------|--------|
| Task Loss (CE) | Push h_out away from h_in | Absorb new task knowledge |
| Local KL | Pull output distribution back toward input distribution | Preserve predictive behavior |
| Local MSE | Pull h_out back toward h_in | Preserve representational structure |

The balance is controlled by $\lambda_{kl}$ and $\lambda_{feat}$. The identity block grows from zero, learning to satisfy both the new task and the distributional anchor — absorbing new knowledge without corrupting the knowledge already encoded in frozen layers.

```
 Original (28 layers)              Expanded (42 layers, second_half)
 ──────────────────────            ──────────────────────────────────
 Layer 0                           Layer 0  (frozen)
 Layer 1                           Layer 1  (frozen)
 ...                               ...
 Layer 13                          Layer 13 (frozen)
                                   Layer 14 (frozen)
 Layer 14                   ──→    [ID-14]  (trainable) ← new!
 Layer 15                          Layer 15 (frozen)
 ...                               [ID-15]  (trainable) ← new!
 Layer 27                          ...
                                   Layer 27 (frozen)
                                   [ID-27]  (trainable) ← new!
```

### Identity Block Initialization

Qwen's DecoderLayer uses Pre-Norm residual structure. Zeroing `o_proj` and `down_proj` weights:

```python
# Normal layer: output ≠ input
x = residual + Attention(x)
x = residual + MLP(x)

# Identity block: output = input (exact)
x = residual + 0 = residual   # attention branch zeroed
x = residual + 0 = residual   # MLP branch zeroed
```

At initialization, the expanded model behaves identically to the original. Training gradually grows identity blocks from zero into meaningful layers.

---

## Loss Formulation

$$
\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_{kl} \cdot \mathcal{L}_{kl}^{local} + \lambda_{feat} \cdot \mathcal{L}_{feat}^{local}
$$

| Loss | Formula | Space |
|------|---------|-------|
| **Task Loss** | Cross-entropy on labels | Output |
| **Local KL** | $\frac{1}{K}\sum_{k} KL(\text{softmax}(\text{lm}\_\text{head}(h_{in}^{(k)})/\tau) \;\|\|\; \text{softmax}(\text{lm}\_\text{head}(h_{out}^{(k)})/\tau))$ | Output distribution |
| **Local MSE** | $\frac{1}{K}\sum_{k} \text{MSE}(h_{in}^{(k)},\; h_{out}^{(k)})$ | Hidden state |

### Implementation Details

- `h_in` is **detached** at each identity block — each block trains independently, no gradient cross-talk
- `P_in` is computed under `torch.no_grad()` — zero backward overhead for the reference side
- Both KL and MSE are averaged over all $K$ identity blocks and computed in **float32**

<details>
<summary>Architecture Diagram</summary>

```
┌───────────────────────────────────────────────────────────┐
│               Single Expanded Model                        │
│                                                           │
│  Layer 0 (frozen) ──→ ... ──→ Layer 14 (frozen)          │
│                                    │                      │
│                                    ↓ h_in (detached)      │
│                              ┌──────────────┐             │
│                              │  ID-14       │ (trainable) │
│                              └──────┬───────┘             │
│                                     │ h_out               │
│                          ┌──────────┼──────────┐          │
│                          │  MSE(h_in, h_out)   │          │
│                          │  KL(lm(h_in)‖lm(h_out))│       │
│                          └──────────┴──────────┘          │
│                                     │                     │
│                                     ↓                     │
│  Layer 15 (frozen) ──→ ... ──→ [Final Output] ──→ CE Loss │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

</details>

---

## Getting Started

### Installation

```bash
git clone <repo-url>
cd sft_distill_mil
uv sync
```

Dependencies: `torch>=2.11`, `transformers>=5.5`, `modelscope>=1.36`

### Download Model

```bash
python dl.py  # Downloads Qwen3-... to models/
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

**Format 3 — Plain Text (recommended for pretrain-style corpora)**

```jsonl
{"text": "Artificial Intelligence (AI) is a branch of computer science..."}
{"text": "In recent years, large language models (LLM) have achieved breakthrough progress..."}
```

> Converted to `[{"role": "assistant", "content": text}]`. The entire text is used as the training target (assistant content), aligning with the convention of pretrain-style corpora (wikitext, ruozhiba, etc.). When `--train_on_responses_only` is enabled, the entire text still contributes to the loss.

**Format 4 — Messages with Thinking (Qwen3 reasoning mode)**

```jsonl
{"messages": [{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "<think\>\nBasic arithmetic.\n</think\>\n\n1+1 equals 2."}]}
```

> When the assistant content contains `<think\>...</think\>`, the chat template preserves it as-is. Data with and without thinking can be mixed freely in the same file — no extra flags required.

### Mixing Multiple Formats / Files

The `_to_messages` logic dispatches per-line based on which keys exist, so **a single JSONL file can mix `messages`, `text`, and `instruction/output` rows freely**.

For multiple files, just concatenate them:

```bash
cat data/ruozhiba.jsonl data/example_messages_with_system.jsonl data/facts.jsonl \
    > data/_merged.jsonl
python scripts/train.py --data_path data/_merged.jsonl ...
```

<details>
<summary>Sample Files</summary>

Sample files are provided in the `data/` directory:

| File | Format |
|------|--------|
| `example_messages_with_system.jsonl` | Messages with system prompt |
| `example_messages_without_system.jsonl` | Messages without system prompt |
| `example_messages_with_think.jsonl` | Messages with `<think\>` reasoning mode |
| `example_instruction_response.jsonl` | Instruction-Response |
| `example_plain_text.jsonl` | Plain text (treated as assistant content) |

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

# Only identity block parameters get gradients
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

$$\mathrm{expanded\_idx}(i) = i + |\{p \in P : p < i\}|$$

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
| `--train_on_responses_only` | `False` | (Flag) Only compute loss on the assistant's responses |
| `--gradient_checkpointing` | `False` | (Flag) Enable gradient checkpointing to save memory (~25% slower) |

### Strategy

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--strategy` | `second_half` | Insertion strategy |
| `--strategy_n` | `2` | Interval N for `every_n` |
| `--strategy_positions` | `None` | Comma-separated indices for `custom` |

### Local Distillation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | `2.0` | Softmax temperature τ for local KL |
| `--lambda_kl` | `0.5` | Weight for local KL divergence loss |
| `--lambda_feat` | `0.1` | Weight for local feature distillation loss |

### Logging & Checkpoints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--log_interval` | `10` | Logging interval (steps) |
| `--save_interval` | `500` | Checkpoint save interval (steps) |
| `--save_total_limit` | `3` | Max checkpoints to keep |

---

## Notes

### GPU Memory

Only **one** model is loaded. Estimated memory for Qwen3-0.6B:

| Component | Memory (bf16) |
|-----------|---------------|
| Expanded model (42L, only ID blocks with grads) | ~2.4 GB |
| Optimizer (AdamW, identity blocks only) | ~1.2 GB |
| Activations (output_hidden_states) | ~1.5 GB |
| **Total** | **~5 GB** |

For larger models, use DeepSpeed ZeRO or FSDP.

### Hyperparameter Tuning Tips

| Parameter | Range | Too Low | Too High |
|-----------|-------|---------|----------|
| `temperature` | 1.0 – 4.0 | Soft labels → one-hot, distillation fails | Probabilities flatten, gradients vanish |
| `lambda_kl` | 0.1 – 1.0 | Insufficient output-distribution protection | Hinders identity blocks from learning |
| `lambda_feat` | 0.01 – 0.5 | Weak hidden-state alignment | Over-constrains, blocks can't grow |
| `lr` | 1e-5 – 5e-5 | Slow convergence | Identity blocks overfit to training data |

### bfloat16 Precision

Training runs in bfloat16. KL divergence and feature loss are computed in float32 for numerical stability.

### Output Structure

```
output/
├── best/                  # Best model (lowest avg epoch loss)
├── final/                 # Final model after all epochs
├── checkpoint-500/        # Intermediate checkpoints
├── checkpoint-1000/
└── ...
```

Each directory contains the full expanded model and tokenizer, loadable via:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("output/best")
```

### Quick Reference

Train:
```bash
python scripts/train.py --model_path xxx --data_path xxx \
    --train_on_responses_only --lambda_kl 0.5 --lambda_feat 0.1 \
    --epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
    --lr 2e-5 --gradient_checkpointing
```

Inference:
```bash
python scripts/chat.py --model_path output/best --think
```

---

## File Structure

```
sft_distill_mil/
├── src/sft_distill_mil/  # Core package
│   ├── __init__.py
│   ├── model.py          # Insertion strategies, model creation, local distillation
│   └── trainer.py        # SFTDataset, training loop
├── scripts/              # Entry point scripts
│   ├── train.py          # python scripts/train.py ...
│   └── download.py       # Model download utility
├── data/                 # Sample datasets
├── models/               # Local model files (gitignored)
├── pyproject.toml
├── README.md
└── README_CN.md
```

## License

MIT
