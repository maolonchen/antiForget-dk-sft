<div align="center">

# SFT-Distill-MIL

**避免仅新知识微调造成的灾难性遗忘：Block Expansion + 蒸馏**

通过块扩展（Block Expansion）和知识蒸馏，在微调 Qwen 系列模型时有效缓解灾难性遗忘。

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-5.5%2B-yellow.svg)](https://huggingface.co/docs/transformers)

</div>

---

## 为什么用 Block Expansion？💡

大模型微调面临一个核心矛盾：**学新忘旧**。传统做法（L2 正则、EWC 等）试图约束参数不动，但在数十亿参数的空间里收效甚微。

Block Expansion 换了个思路：

> **不约束旧参数，而是插入新参数。** 让新知识有独立的存储空间，旧知识自然保留。

在 Transformer 层之间插入**恒等块**。初始状态下恒等块是透明的（输入=输出），扩展后的模型行为与原始模型完全一致。训练时，恒等块逐渐从零增长为有意义的层。

```
 原始模型 (28 层)                  扩展后模型 (42 层, second_half 策略)
 ──────────────────────            ──────────────────────────────────
 Layer 0                           Layer 0  (原始层)
 Layer 1                           Layer 1  (原始层)
 ...                               ...
 Layer 13                          Layer 13 (原始层)
                                   Layer 14 (原始层)
 Layer 14                   ──→    [ID-14]  (恒等块) ← 新增!
 Layer 15                          Layer 15 (原始层)
 ...                               [ID-15]  (恒等块) ← 新增!
 Layer 27                          ...
                                   Layer 27 (原始层)
                                   [ID-27]  (恒等块) ← 新增!
```

### 恒等块的工作原理

Qwen3 的 DecoderLayer 采用 Pre-Norm 残差结构。将 `o_proj`（注意力最后一层）和 `down_proj`（MLP 最后一层）的权重置零：

```python
# 置零前：正常 Transformer 层
x = residual + Attention(x)   # o_proj 输出非零
x = residual + MLP(x)         # down_proj 输出非零

# 置零后：精确恒等映射
x = residual + 0 = residual   # 注意力分支为零，残差直传
x = residual + 0 = residual   # MLP 分支为零，残差直传
```

训练过程中梯度从零开始更新这两个权重，恒等块逐渐学出新的特征变换。

---

## 三重损失设计

单靠块扩展不够——原始层的权重仍会被新任务的梯度间接扰动。因此引入三重损失：

$$
\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_{kl} \cdot \mathcal{L}_{kl} + \lambda_{feat} \cdot \mathcal{L}_{feat}
$$

| 损失 | 公式 | 作用 |
|------|------|------|
| **任务损失** | 交叉熵（labels） | 学习新任务 |
| **KL 蒸馏** | $\tau^2 \cdot KL(\text{softmax}(S/\tau) \;\|\|\; \text{softmax}(T/\tau))$ | 让 Student 输出分布模仿 Teacher |
| **特征蒸馏** | $\frac{1}{N}\sum_i \text{MSE}(S_{hidden[i]},\; T_{hidden[i]})$ | 逐层对齐中间表征 |

- **KL 蒸馏**保护的是输出层面的知识（语法、常识、推理偏好）
- **特征蒸馏**保护的是内部表征层面的知识（特征空间结构）
- 两者互补，覆盖了从底层到输出的完整知识链路

<details>
<summary>架构图</summary>

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

## 快速开始

### 安装

```bash
# 克隆并安装
git clone <repo-url>
cd sft_distill_mil

# 使用 uv
uv sync
```

依赖：`torch>=2.11`、`transformers>=5.5`、`modelscope>=1.36`

### 下载模型

```bash
python dl.py  # 下载 Qwen3-0.6B 到 models/ 目录
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
{"messages": [{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "<think>\n基础算术。\n</think>\n\n1+1 等于 2。"}]}
```

> 当 assistant 内容含 `<think>...</think>` 时，chat template 会原样保留。带 think 与不带 think 的数据可以在同一个文件中自由混合，**不需要额外参数控制**。

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

`data/` 目录下提供了四个样例文件：

| 文件 | 格式 |
|------|------|
| `example_messages_with_system.jsonl` | Messages（含 system） |
| `example_messages_without_system.jsonl` | Messages（无 system） |
| `example_messages_with_think.jsonl` | Messages（含 `<think>` 思考模式） |
| `example_instruction_response.jsonl` | Instruction-Response |
| `example_plain_text.jsonl` | Plain Text（视为 assistant 内容） |

</details>

### 训练

```bash
# 快速开始（默认 second_half 策略）
python scripts/train.py \
    --model_path /root/autodl-tmp/models/Qwen/Qwen3-0.6B \
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

$$\mathrm{student\_idx}(i) = i + |\{p \in P : p < i\}|$$

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

| `--train_on_responses_only` | `False` | （布尔标签）如果设置，则仅在 assistant 的回复部分计算 loss（标准的 SFT 做法） |

### 策略参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--strategy` | `second_half` | 插入策略 |
| `--strategy_n` | `2` | `every_n` 策略的间隔 N |
| `--strategy_positions` | `None` | `custom` 策略的层索引，逗号分隔 |

### 蒸馏参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--temperature` | `2.0` | KL 蒸馏的 softmax 温度 τ |
| `--lambda_kl` | `0.5` | KL 散度损失权重 |
| `--lambda_feat` | `0.1` | 特征蒸馏损失权重 |

### 日志与检查点

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--log_interval` | `10` | 日志打印间隔（步） |
| `--save_interval` | `500` | 检查点保存间隔（步） |

---

## 注意事项

### 显存占用

Teacher 和 Student 需要同时加载。Qwen3-0.6B 的显存估算：

| 组件 | 显存 (bf16) |
|------|-------------|
| Teacher (28层, 冻结) | ~1.2 GB |
| Student (42层, 含梯度) | ~3.6 GB |
| 优化器 (AdamW) | ~3.6 GB |
| 激活值与临时变量 | ~2 GB |
| **合计** | **~10 GB** |

对于更大的模型（如 Qwen3-32B），请使用 DeepSpeed ZeRO 或 FSDP 进行分布式训练。

### 超参调优建议

| 参数 | 范围 | 过低 | 过高 |
|------|------|------|------|
| `temperature` | 1.0 – 4.0 | 软标签退化为 one-hot，蒸馏失效 | 概率分布趋于平坦，梯度消失 |
| `lambda_kl` | 0.1 – 1.0 | 遗忘保护不足 | 阻碍新任务学习 |
| `lambda_feat` | 0.01 – 0.5 | 特征对齐较弱 | 过度约束内部表示 |
| `lr` | 1e-5 – 5e-5 | 收敛缓慢 | 覆盖原始知识 |

### bfloat16 精度

训练使用 bfloat16 精度。由于尾数精度较低（7 位 vs float32 的 23 位），深层会累积浮点误差：

- Student 初始输出与 Teacher 存在微小差异（最大 logits 差异约 0.3）
- 初始化时特征蒸馏损失不为零

这是正常现象，不影响训练。KL 散度和特征损失均在 float32 下计算以保证数值稳定性。

### 输出结构

```
output/
├── best/                  # 最佳模型（最低平均 epoch 损失）
├── final/                 # 所有 epoch 结束后的最终模型
├── checkpoint-500/        # 中间检查点
├── checkpoint-1000/
└── ...
```

每个目录包含 Student 模型权重和 tokenizer，可通过以下方式加载：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("output/best")
```

### 实践

训练模型：
```python

python scripts/train.py --model_path xxx --data_path xxx --train_on_responses_only --lambda_kl xxx --lambda_feat xxx --epochs xxx --batch_size xxx --gradient_accumulation_steps xxx --lr xxx --warmup_ratio xxx --save_interval xxx --save_total_limit xxx --gradient_checkpointing
```

推理：
```python

python scripts/chat.py --model_path output/best --think
```

---

## 文件结构

```
sft_distill_mil/
├── src/sft_distill_mil/  # 核心代码包
│   ├── __init__.py
│   ├── model.py          # 插入策略、模型创建、蒸馏损失
│   └── trainer.py        # SFTDataset、训练循环
├── scripts/              # 可执行脚本入口
│   ├── train.py          # python scripts/train.py ...
│   └── download.py       # 模型下载工具
├── tests/                # 单元测试
├── data/                 # 样例数据集
│   ├── example_messages_with_system.jsonl
│   ├── example_messages_without_system.jsonl
│   ├── example_instruction_response.jsonl
│   └── example_plain_text.jsonl
├── models/               # 本地模型文件（gitignored）
│   └── Qwen/Qwen-{...}/
├── pyproject.toml
├── README.md
└── README_CN.md
```

## 🎉🥚

这个项目用来做“后门攻击”效果其实意外地好。
（⚠️ 仅限授权测试环境 😉）

## 许可证

MIT
