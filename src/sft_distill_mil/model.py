"""
Block Expansion + Knowledge Distillation for Qwen

在原始模型的指定 Transformer 层后插入恒等块，
并通过 KL 散度 + 逐层特征蒸馏来缓解灾难性遗忘。

支持任意 Qwen3 模型（0.6B / 1.7B / 4B / 8B / 32B 等）
和多种插入策略。
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# 插入策略
# ---------------------------------------------------------------------------

def get_insert_positions(
    strategy: str,
    num_layers: int,
    **kwargs,
) -> list[int]:
    """根据策略生成需要在哪些原始层之后插入恒等块。

    Args:
        strategy: 插入策略名称
            - "second_half": 后半部分每层后插入（默认）
            - "every_layer": 每层后都插入
            - "every_n": 每隔 N 层后插入（需传 n=N）
            - "first_half": 前半部分每层后插入
            - "custom": 自定义位置（需传 positions=[...]）
        num_layers: 原始模型的总层数
        **kwargs: 策略参数

    Returns:
        排序后的原始层索引列表，表示在这些层之后插入恒等块

    Examples:
        >>> get_insert_positions("second_half", 28)
        [14, 15, 16, ..., 27]

        >>> get_insert_positions("every_layer", 4)
        [0, 1, 2, 3]

        >>> get_insert_positions("every_n", 8, n=2)
        [1, 3, 5, 7]

        >>> get_insert_positions("custom", 28, positions=[0, 13, 27])
        [0, 13, 27]
    """
    if strategy == "second_half":
        start = num_layers // 2
        return list(range(start, num_layers))

    if strategy == "every_layer":
        return list(range(num_layers))

    if strategy == "first_half":
        return list(range(0, num_layers // 2))

    if strategy == "every_n":
        n = kwargs.get("n", 2)
        return list(range(n - 1, num_layers, n))

    if strategy == "custom":
        positions = kwargs.get("positions", [])
        return sorted(set(positions))

    raise ValueError(
        f"Unknown strategy: {strategy!r}. "
        f"Choose from: second_half, every_layer, every_n, first_half, custom"
    )


# ---------------------------------------------------------------------------
# 层映射
# ---------------------------------------------------------------------------

def _build_layer_mapping(
    num_original: int,
    insert_positions: list[int],
) -> tuple[list[int], list[int]]:
    """构建 teacher→student 的层索引映射和恒等块位置。

    核心公式：对于 teacher 层 i，
        student_idx = i + count(p < i for p in insert_positions)
    即前面插了多少个恒等块，就往后偏移多少。

    Args:
        num_original: 原始模型层数
        insert_positions: 排序后的插入位置列表

    Returns:
        (layer_mapping, identity_indices)
        - layer_mapping[teacher_idx] = student_idx
        - identity_indices: 恒等块在 student 中的层索引列表
    """
    insert_set = set(insert_positions)

    # teacher → student 映射
    layer_mapping = []
    offset = 0
    insert_iter = iter(sorted(insert_positions))
    next_insert = next(insert_iter, None)

    for i in range(num_original):
        while next_insert is not None and next_insert < i:
            offset += 1
            next_insert = next(insert_iter, None)
        layer_mapping.append(i + offset)

    # 恒等块在 student 中的索引 = mapping[p] + 1
    identity_indices = []
    offset = 0
    insert_sorted = sorted(insert_positions)
    # 重新计算：每个插入位置 p 对应 student 中的 mapping[p] + 1
    for p in insert_sorted:
        student_idx = layer_mapping[p] + 1
        identity_indices.append(student_idx)

    return layer_mapping, identity_indices


# ---------------------------------------------------------------------------
# 权重复制
# ---------------------------------------------------------------------------

def _copy_decoder_layer(src: nn.Module, dst: nn.Module):
    """逐个子模块复制 DecoderLayer 权重。

    适用于所有 Qwen3 模型（0.6B ~ 32B），因为架构相同。
    """
    # self_attn
    dst.self_attn.q_proj.weight.data.copy_(src.self_attn.q_proj.weight.data)
    dst.self_attn.k_proj.weight.data.copy_(src.self_attn.k_proj.weight.data)
    dst.self_attn.v_proj.weight.data.copy_(src.self_attn.v_proj.weight.data)
    dst.self_attn.o_proj.weight.data.copy_(src.self_attn.o_proj.weight.data)
    dst.self_attn.q_norm.weight.data.copy_(src.self_attn.q_norm.weight.data)
    dst.self_attn.k_norm.weight.data.copy_(src.self_attn.k_norm.weight.data)

    # mlp
    dst.mlp.gate_proj.weight.data.copy_(src.mlp.gate_proj.weight.data)
    dst.mlp.up_proj.weight.data.copy_(src.mlp.up_proj.weight.data)
    dst.mlp.down_proj.weight.data.copy_(src.mlp.down_proj.weight.data)

    # layer norms
    dst.input_layernorm.weight.data.copy_(src.input_layernorm.weight.data)
    dst.post_attention_layernorm.weight.data.copy_(
        src.post_attention_layernorm.weight.data
    )


# ---------------------------------------------------------------------------
# 模型创建
# ---------------------------------------------------------------------------

def create_expanded_model(
    model_path: str,
    insert_positions: list[int],
) -> AutoModelForCausalLM:
    """创建含恒等块的学生模型。

    在 insert_positions 指定的每个原始层后插入一个恒等块。
    恒等块通过将 o_proj 和 down_proj 权重置零来实现输入=输出。

    Args:
        model_path: 原始模型路径（支持任意 Qwen3 模型）
        insert_positions: 排序后的原始层索引列表，在这些层之后插入恒等块

    Returns:
        扩展后的模型
    """
    insert_positions = sorted(insert_positions)

    # 加载原始 config 和模型
    original_config = AutoConfig.from_pretrained(model_path)
    original_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    num_original = original_config.num_hidden_layers

    # 构建层映射
    layer_mapping, identity_indices = _build_layer_mapping(
        num_original, insert_positions
    )
    num_total = num_original + len(insert_positions)

    # 构建扩展后的 config
    new_config = copy.deepcopy(original_config)
    new_config.num_hidden_layers = num_total

    # 扩展 layer_types
    old_layer_types = list(original_config.layer_types)
    insert_set = set(insert_positions)
    new_layer_types = []
    for i in range(num_original):
        new_layer_types.append(old_layer_types[i])
        if i in insert_set:
            new_layer_types.append("full_attention")
    new_config.layer_types = new_layer_types

    # 用新 config 创建模型（随机初始化）
    new_model = AutoModelForCausalLM.from_config(new_config)
    new_model = new_model.to(torch.bfloat16)

    # --- 权重复制 ---
    new_model.model.embed_tokens.weight.data.copy_(
        original_model.model.embed_tokens.weight.data
    )
    new_model.model.norm.weight.data.copy_(original_model.model.norm.weight.data)
    new_model.lm_head.weight.data.copy_(original_model.lm_head.weight.data)

    for teacher_idx, student_idx in enumerate(layer_mapping):
        _copy_decoder_layer(
            original_model.model.layers[teacher_idx],
            new_model.model.layers[student_idx],
        )

    # --- 恒等块初始化 ---
    for identity_idx in identity_indices:
        layer = new_model.model.layers[identity_idx]
        nn.init.zeros_(layer.self_attn.o_proj.weight)
        nn.init.zeros_(layer.mlp.down_proj.weight)

    del original_model
    return new_model


# ---------------------------------------------------------------------------
# 蒸馏训练封装
# ---------------------------------------------------------------------------

class BlockExpansionWrapper(nn.Module):
    """封装 Teacher + Student 的联合模型，计算三部分损失。

    支持任意 Qwen3 模型和插入策略。
    """

    def __init__(
        self,
        model_path: str,
        strategy: str = "second_half",
        strategy_kwargs: Optional[dict] = None,
        temperature: float = 2.0,
        lambda_kl: float = 0.5,
        lambda_feat: float = 0.1,
    ):
        """
        Args:
            model_path: 原始模型路径
            strategy: 插入策略（见 get_insert_positions）
            strategy_kwargs: 策略参数，如 {"n": 2} 或 {"positions": [0, 13, 27]}
            temperature: KL 蒸馏温度
            lambda_kl: KL 损失权重
            lambda_feat: 特征蒸馏损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.lambda_kl = lambda_kl
        self.lambda_feat = lambda_feat

        # Teacher: 冻结的原始模型
        self.teacher = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 计算插入位置
        num_layers = self.teacher.config.num_hidden_layers
        strategy_kwargs = strategy_kwargs or {}
        self.insert_positions = get_insert_positions(strategy, num_layers, **strategy_kwargs)

        # 构建层映射
        self.layer_mapping, self.identity_indices = _build_layer_mapping(
            num_layers, self.insert_positions
        )

        # Student: 块扩展后的模型
        self.student = create_expanded_model(model_path, self.insert_positions)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """前向传播，计算总损失。

        Returns:
            dict with keys: total_loss, task_loss, kl_loss, feat_loss
        """
        # Teacher forward（不计算梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Student forward
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        # 1. Task loss（交叉熵）
        task_loss = student_outputs.loss

        # 2. KL 散度损失
        kl_loss = self._compute_kl_loss(
            student_outputs.logits, teacher_outputs.logits
        )

        # 3. 逐层特征蒸馏损失
        feat_loss = self._compute_feature_loss(
            student_outputs.hidden_states, teacher_outputs.hidden_states
        )

        # 总损失
        total_loss = task_loss + self.lambda_kl * kl_loss + self.lambda_feat * feat_loss

        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "kl_loss": kl_loss,
            "feat_loss": feat_loss,
        }

    def _compute_kl_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """KL 散度蒸馏损失（float32 计算避免 bf16 精度问题）。"""
        teacher_logits_aligned = teacher_logits[:, : student_logits.shape[1], :]

        s_logits = student_logits.float() / self.temperature
        t_logits = teacher_logits_aligned.float() / self.temperature

        student_log_probs = F.log_softmax(s_logits, dim=-1)
        teacher_probs = F.softmax(t_logits, dim=-1)

        kl_loss = F.kl_div(
            student_log_probs, teacher_probs, reduction="batchmean"
        ) * (self.temperature**2)
        return kl_loss.clamp(min=0.0)

    def _compute_feature_loss(
        self,
        student_hidden: tuple[torch.Tensor, ...],
        teacher_hidden: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """逐层特征蒸馏损失（MSE）。

        对 teacher 的每一层，找到 student 对应层，计算 MSE。
        """
        total_mse = torch.tensor(0.0, device=student_hidden[0].device)
        num_layers = len(self.layer_mapping)

        for teacher_idx, student_idx in enumerate(self.layer_mapping):
            s_hidden = student_hidden[student_idx + 1].float()
            t_hidden = teacher_hidden[teacher_idx + 1].float()
            total_mse = total_mse + F.mse_loss(s_hidden, t_hidden)

        return total_mse / num_layers
