<div align="center">

# antiForget-dk-sft

**通过恒等块局部分布锚定实现无遗忘的 LLM 微调**

Block Expansion 提供架构，分布锚定提供保护。

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-5.5%2B-yellow.svg)](https://huggingface.co/docs/transformers)

</div>

---

## 核心创新：局部分布锚定

Block Expansion（向预训练模型插入恒等映射层以增加容量）是已有的技术。但现有工作存在一个关键缺陷：**插入的恒等块会扭曲流经冻结层的内部表示，导致模型原有能力级联退化。**

我们提出**局部分布锚定（Local Distributional Anchoring）**——在每个恒等块插入点，将块的输出分布锚定到输入分布，构建一个自指的约束，在吸收新知识的同时保护模型的预测行为：

> **模型自身的块前分布就是就地参考。** 在每个插入点，恒等块面临一场分布拉扯：任务损失将它推向新知识，而局部 KL 散度和特征蒸馏将它锚定回原始分布。

```
 在每个恒等块插入点：

   冻结层 i 的输出 (h_in)
          │
          ↓  ┌─────────────────────────────────────────┐
          │  │         恒等块（可训练）                    │
          │  └────────────────────┬────────────────────┘
          │                       │ 输出 (h_out)
          │                       │
          │    ┌──────────────────┤
          │    │   分布锚定约束    │
          │    │                  │
          │    │  MSE(h_in, h_out)        ← 隐空间锚定
          │    │  KL(P_in ‖ P_out)        ← 输出空间锚定
          │    │  其中 P = softmax(lm_head(h) / τ)
          │    └──────────────────┤
          │                       │
          ↓                       ↓
   → 冻结层 i+1 → ... → 最终输出 → CE Loss（任务学习）
```

这在每个插入点构建了**双空间约束**：
- **隐空间（MSE）**：保持数值表示接近——结构性保护
- **输出空间（KL）**：保持预测的 token 分布接近——行为性保护

两个空间互补。MSE 对维度一视同仁、计算量小，但不知道哪些维度重要。KL 通过 `lm_head` 投射到词表空间，自动聚焦于真正影响预测的方向。一个被 MSE 视为可忽略、但会翻转模型 top 预测的扰动，会被 KL 精准捕捉——这就是防止后续冻结层**级联失真**的关键。

### 为什么有效：分布拉扯的动态平衡

训练过程中，每个恒等块被相反的力拉扯：

| 力 | 方向 | 效果 |
|---|------|------|
| 任务损失（CE） | 将 h_out 推离 h_in | 吸收新任务知识 |
| 局部 KL | 将输出分布拉回输入分布 | 保护预测行为 |
| 局部 MSE | 将 h_out 拉回 h_in | 保护表示结构 |

平衡由 $\lambda_{kl}$ 和 $\lambda_{feat}$ 控制。恒等块从零开始增长，在满足新任务和分布锚定之间找到平衡——在不破坏冻结层已有知识的前提下吸收新知识。

```
 原始模型 (28 层)                  扩展后模型 (42 层, second_half 策略)
 ──────────────────────            ──────────────────────────────────
 Layer 0                           Layer 0  (冻结)
 Layer 1                           Layer 1  (冻结)
 ...                               ...
 Layer 13                          Layer 13 (冻结)
                                   Layer 14 (冻结)
 Layer 14                   ──→    [ID-14]  (可训练) ← 新增!
 Layer 15                          Layer 15 (冻结)
 ...                               [ID-15]  (可训练) ← 新增!
 Layer 27                          ...
                                   Layer 27 (冻结)
                                   [ID-27]  (可训练) ← 新增!
```

### 恒等块的工作原理

Qwen3 的 DecoderLayer 采用 Pre-Norm 残差结构。将 `o_proj`（注意力最后一层）和 `down_proj`（MLP 最后一层）的权重置零：

```python
# 正常层：输出 ≠ 输入
x = residual + Attention(x)
x = residual + MLP(x)

# 恒等块：输出 = 输入（精确）
x = residual + 0 = residual   # 注意力分支置零，残差直传
x = residual + 0 = residual   # MLP 分支置零，残差直传
```

初始化时扩展模型行为与原始模型完全一致。训练过程中恒等块从零逐渐增长为有意义的层。

---

## 损失公式

$$
\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_{kl} \cdot \mathcal{L}_{kl}^{local} + \lambda_{feat} \cdot \mathcal{L}_{feat}^{local}
$$

| 损失 | 公式 | 空间 |
|------|------|------|
| **任务损失** | 交叉熵（labels） | 输出 |
| **局部 KL** | $\frac{1}{K}\sum_{k} KL(\text{softmax}(\text{lm}\_\text{head}(h_{in}^{(k)})/\tau) \;\|\|\; \text{softmax}(\text{lm}\_\text{head}(h_{out}^{(k)})/\tau))$ | 输出分布 |
| **局部 MSE** | $\frac{1}{K}\sum_{k} \text{MSE}(h_{in}^{(k)},\; h_{out}^{(k)})$ | 隐空间 |

### 实现细节

- `h_in` 在每个恒等块处 **detach**——各块独立训练，无梯度交叉干扰
- `P_in` 在 `torch.no_grad()` 下计算——参考侧零反向开销
- KL 和 MSE 均对所有 $K$ 个恒等块取平均，并在 **float32** 下计算

<details>
<summary>架构图</summary>

```
┌───────────────────────────────────────────────────────────┐
│               单一扩展模型                                  │
│                                                           │
│  Layer 0 (冻结) ──→ ... ──→ Layer 14 (冻结)              │
│                                    │                      │
│                                    ↓ h_in (detach)        │
│                              ┌──────────────┐             │
│                              │  ID-14       │ (可训练)     │
│                              └──────┬───────┘             │
│                                     │ h_out               │
│                          ┌──────────┼──────────┐          │
│                          │  MSE(h_in, h_out)   │          │
│                          │  KL(lm(h_in)‖lm(h_out))│       │
│                          └──────────┴──────────┘          │
│                                     │                     │
│                                     ↓                     │
│  Layer 15 (冻结) ──→ ... ──→ [最终输出] ──→ CE Loss        │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

</details>

---

## 快速开始

### 安装

```bash
git clone <repo-url>
cd sft_distill_mil
uv sync
```

依赖：`torch>=2.11`、`transformers>=5.5`、`modelscope>=1.36`

### 下载模型

```bash
python dl.py  # 下载 Qwen3-... 到 models/ 目录
```

### 准备数据集

训练数据使用 **JSONL** 格式（每行一个 JSON 对象）。支持三种格式：

**格式 1 — Messages（推荐，适配 Qwen3 ChatML）**

```jsonl
{"messages": [{"role": "user", "content": "什么是机器学习？"}, {"role": "assistant", "content": "机器学习是人工智能的一个子领域..."}]}
{"messages": [{"role": "system", "content": "你是一个有用的助手"}, {"role": "user", "content": "写一首关于春天的诗"}, {"role": "assistant", "content": "春风拂面柳丝长..."}]}
```

> 所有格式最终都会通过 `tokenizer.apply_chat_template()` 转换为 Qwen3 的 ChatML 格式（`<|im_start|>user\n...<|im_end|>`）。

**格式 2 — Instruction-Response（向后兼容）**

```jsonl
{"instruction": "将以下句子翻译为英文", "output": "Hello, how are you today?"}
{"instruction": "总结以下文章的主旨", "output": "本文主要讨论了人工智能在医疗领域的应用前景。"}
```

> 自动转换为 `[{"role": "user", "content": instruction}, {"role": "assistant", "content": output}]`，支持可选的 `"system"` 字段。

**格式 3 — Plain Text（推荐用于预训练风格语料）**

```jsonl
{"text": "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支..."}
{"text": "近年来，大语言模型（LLM）在自然语言处理领域取得了突破性进展..."}
```

> 转换为 `[{"role": "assistant", "content": text}]`。整段文本作为训练目标（assistant 内容），符合 wikitext、ruozhiba 等预训练风格语料的惯例。开启 `--train_on_responses_only` 时整段仍然参与 loss 计算。

**格式 4 — Messages with Thinking（Qwen3 思考模式）**

```jsonl
{"messages": [{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "ächwen\n基础算术。\n羚羊\n\n1+1 等于 2。"}]}
```

> 当 assistant 内容含 `ächwen...羚羊` 时，chat template 会原样保留。带 think 与不带 think 的数据可以在同一个文件中自由混合，**不需要额外参数控制**。

### 混合多种格式 / 多个文件

`_to_messages` 按行内字段判断格式，因此 **单个 JSONL 文件可以自由混合 `messages`、`text`、`instruction/output` 三种格式**。

需要混合多个文件时，直接 cat 即可：

```bash
cat data/ruozhiba.jsonl data/example_messages_with_system.jsonl data/facts.jsonl \
    > data/_merged.jsonl
python scripts/train.py --data_path data/_merged.jsonl ...
```

<details>
<summary>样例文件</summary>

`data/` 目录下提供了样例文件：

| 文件 | 格式 |
|------|------|
| `example_messages_with_system.jsonl` | Messages（含 system） |
| `example_messages_without_system.jsonl` | Messages（无 system） |
| `example_messages_with_think.jsonl` | Messages（含 `ächwen` 思考模式） |
| `example_instruction_response.jsonl` | Instruction-Response |
| `example_plain_text.jsonl` | Plain Text（视为 assistant 内容） |

</details>

### 训练

```bash
# 快速开始（默认 second_half 策略）
python scripts/train.py \
    --model_path models/Qwen/Qwen3-0.6B \
    --data_path data/example_messages_with_system.jsonl

# 每层都插入恒等块
python scripts/train.py \
    --model_path models/Qwen/Qwen3-0.6B \
    --data_path data/example_messages_with_system.jsonl \
    --strategy every_layer

# 每隔 4 层插入
python scripts/train.py \
    --model_path models/Qwen/Qwen3-0.6B \
    --data_path data/example_messages_with_system.jsonl \
    --strategy every_n --strategy_n 4

# 自定义位置
python scripts/train.py \
    --model_path models/Qwen/Qwen3-0.6B \
    --data_path data/example_messages_with_system.jsonl \
    --strategy custom --strategy_positions "0,13,27"
```

### 代码调用

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

# 只有恒等块参数会获得梯度
losses["total_loss"].backward()
```

---

## 插入策略

| 策略 | 描述 | 0.6B (28层) | 32B (64层) | 适用场景 |
|------|------|-------------|------------|----------|
| `second_half` | 后半部分每层后插入 | 28→42 | 64→96 | 默认策略；高层特征受益最大 |
| `every_layer` | 每层后都插入 | 28→56 | 64→128 | 最大化容量增长 |
| `every_n` | 每隔 N 层插入 | +14 (n=2) | +32 (n=2) | 平衡增长与效率 |
| `first_half` | 前半部分每层后插入 | 28→42 | 64→96 | 修改底层表示 |
| `custom` | 自定义层索引 | 自定义 | 自定义 | 完全控制 |

**通用层映射公式：**

$$\mathrm{expanded\_idx}(i) = i + |\{p \in P : p < i\}|$$

其中 $P$ 是插入位置集合。此公式适用于任意模型大小和策略。

---

## 全部参数

### 训练参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--model_path` | `models/Qwen/Qwen3-0.6B` | 预训练模型路径 |
| `--data_path` | *(必填)* | JSONL 训练数据路径 |
| `--output_dir` | `output` | 输出目录 |
| `--epochs` | `3` | 训练轮数 |
| `--batch_size` | `4` | 批大小 |
| `--gradient_accumulation_steps` | `4` | 梯度累积步数 |
| `--lr` | `2e-5` | 学习率 |
| `--weight_decay` | `0.01` | 权重衰减 |
| `--warmup_ratio` | `0.1` | 预热比例 |
| `--max_seq_length` | `512` | 最大序列长度 |
| `--train_on_responses_only` | `False` | （布尔标签）仅在 assistant 的回复部分计算 loss |
| `--gradient_checkpointing` | `False` | （布尔标签）启用梯度检查点，节省显存（约慢 25%） |

### 策略参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--strategy` | `second_half` | 插入策略 |
| `--strategy_n` | `2` | `every_n` 策略的间隔 N |
| `--strategy_positions` | `None` | `custom` 策略的层索引，逗号分隔 |

### 局部蒸馏参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--temperature` | `2.0` | 局部 KL 的 softmax 温度 τ |
| `--lambda_kl` | `0.5` | 局部 KL 散度损失权重 |
| `--lambda_feat` | `0.1` | 局部特征蒸馏损失权重 |

### 日志与检查点

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--log_interval` | `10` | 日志打印间隔（步） |
| `--save_interval` | `500` | 检查点保存间隔（步） |
| `--save_total_limit` | `3` | 最多保留检查点数量 |

---

## 注意事项

### 显存占用

只需加载**一个**模型。Qwen3-0.6B 的显存估算：

| 组件 | 显存 (bf16) |
|------|-------------|
| 扩展模型 (42层, 仅恒等块有梯度) | ~2.4 GB |
| 优化器 (AdamW, 仅恒等块) | ~1.2 GB |
| 激活值 (output_hidden_states) | ~1.5 GB |
| **合计** | **~5 GB** |

更大模型请使用 DeepSpeed ZeRO 或 FSDP。

### 超参调优建议

| 参数 | 范围 | 过低 | 过高 |
|------|------|------|------|
| `temperature` | 1.0 – 4.0 | 软标签退化为 one-hot，蒸馏失效 | 概率分布趋于平坦，梯度消失 |
| `lambda_kl` | 0.1 – 1.0 | 输出分布保护不足 | 恒等块学不动 |
| `lambda_feat` | 0.01 – 0.5 | 特征对齐较弱 | 过度约束，恒等块无法增长 |
| `lr` | 1e-5 – 5e-5 | 收敛缓慢 | 恒等块过拟合训练数据 |

### bfloat16 精度

训练使用 bfloat16 精度。KL 散度和特征损失均在 float32 下计算以保证数值稳定性。

### 输出结构

```
output/
├── best/                  # 最佳模型（最低平均 epoch 损失）
├── final/                 # 所有 epoch 结束后的最终模型
├── checkpoint-500/        # 中间检查点
├── checkpoint-1000/
└── ...
```

每个目录包含完整扩展模型权重和 tokenizer，可通过以下方式加载：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("output/best")
```

### 实践

训练：
```bash
python scripts/train.py --model_path xxx --data_path xxx \
    --train_on_responses_only --lambda_kl 0.5 --lambda_feat 0.1 \
    --epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
    --lr 2e-5 --gradient_checkpointing
```

推理：
```bash
python scripts/chat.py --model_path output/best --think
```

---

## 文件结构

```
sft_distill_mil/
├── src/sft_distill_mil/  # 核心代码包
│   ├── __init__.py
│   ├── model.py          # 插入策略、模型创建、局部蒸馏损失
│   └── trainer.py        # SFTDataset、训练循环
├── scripts/              # 可执行脚本入口
│   ├── train.py          # python scripts/train.py ...
│   └── download.py       # 模型下载工具
├── data/                 # 样例数据集
├── models/               # 本地模型文件（gitignored）
├── pyproject.toml
├── README.md
└── README_CN.md
```

## 许可证

MIT
