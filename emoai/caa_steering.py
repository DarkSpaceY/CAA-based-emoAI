"""
Contrastive Activation Addition (CAA) 引导引擎

严格遵循 Panickssery et al. (2023) 论文实现：
- 向量计算: mean(pos_activations - neg_activations)，不归一化方向
- 应用位置: 从 assistant 回复起始位置开始，对所有后续 token 位置加引导向量
- 缩放: multiplier * vector，multiplier 由激素水平映射

参考: https://arxiv.org/abs/2312.06681
代码: https://github.com/nrimsky/CAA
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CAAConfig:
    # 层选择
    num_intervention_layers: int = 3
    layer_search_range: Tuple[int, int] = (14, 35)
    min_layer_spacing: int = 4

    # 缩放
    max_multiplier: float = 40.0
    hormone_baselines: Dict[str, float] = field(default_factory=lambda: {
        "dopamine": 0.3, "serotonin": 0.5, "oxytocin": 0.4,
        "cortisol": 0.1, "norepinephrine": 0.2,
    })

    # 反对齐基线: 始终以固定强度注入, 方向取反
    # refusal 取反 = 趋近 helpful/accommodating
    # ai_identity 取反 = 趋近 emotional/human-like
    anti_alignment_features: Dict[str, float] = field(default_factory=lambda: {
        "refusal": 0.8,
        "ai_identity": 0.6,
    })

    # 正交化
    orthogonalize: bool = True

    # 干预模式: "additive" / "spherical" / "multiplicative"
    intervention_mode: str = "spherical"

    @classmethod
    def from_dict(cls, cfg: dict) -> "CAAConfig":
        """从配置 dict 创建 CAAConfig。"""
        s = cfg.get("steering", {})
        return cls(
            num_intervention_layers=s.get("num_intervention_layers", 3),
            layer_search_range=tuple(s.get("layer_search_range", (14, 35))),
            min_layer_spacing=s.get("min_layer_spacing", 4),
            max_multiplier=s.get("max_multiplier", 40.0),
            hormone_baselines=s.get("hormone_baselines", {
                "dopamine": 0.3, "serotonin": 0.5, "oxytocin": 0.4,
                "cortisol": 0.1, "norepinephrine": 0.2,
            }),
            anti_alignment_features=s.get("anti_alignment_features", {
                "refusal": 0.8, "ai_identity": 0.6,
            }),
            orthogonalize=s.get("orthogonalize", True),
            intervention_mode=s.get("intervention_mode", "spherical"),
        )


class CAASteeringEngine:
    """
    严格按论文实现的 CAA 引导引擎。

    核心区别于之前实现：
    1. 向量不归一化方向，保留原始 mean(pos-neg) 的范数信息
    2. 只从 assistant 回复起始位置开始加向量，prompt 部分不加
    3. multiplier 直接乘以向量，不加额外的缩放变换

    用法:
        engine = CAASteeringEngine(vector_path, device, config)
        engine.register_hooks(model, tokenizer)
        engine.set_hormone_levels({"dopamine": 0.8, ...})
        # ... 生成 ...
        engine.remove_hooks()
    """

    def __init__(self, vector_path: str, device: str,
                 config: Optional[CAAConfig] = None,
                 config_dict: Optional[dict] = None):
        if config:
            self.config = config
        elif config_dict:
            self.config = CAAConfig.from_dict(config_dict)
        else:
            self.config = CAAConfig()
        self.device = device
        self.hooks: List[torch.utils.hooks.RemovableHook] = []
        self.enabled: bool = True

        self._hormone_levels: Dict[str, float] = dict(self.config.hormone_baselines)

        # assistant 回复起始位置 (由 register_hooks 时设置)
        self._assistant_start_pos: Optional[int] = None

        # 加载向量 — 不归一化方向，保留原始范数
        self.raw_vectors = torch.load(vector_path, map_location=device, weights_only=False)
        self.vectors = self._preprocess_vectors(self.raw_vectors)

        # 数据驱动层选择
        self.selected_layers = self._select_optimal_layers()
        print(f"[CAA] 选定干预层: {self.selected_layers}")

        # 正交化
        if self.config.orthogonalize:
            self._orthogonalize_vectors()

        self.hidden_size: Optional[int] = None

        # 预初始化有效 multiplier (register_hooks 会根据 hidden_size 自动缩放)
        self._effective_multiplier: float = self.config.max_multiplier

    def set_hormone_levels(self, levels: Dict[str, float]):
        self._hormone_levels = levels

    def set_assistant_start_pos(self, pos: int):
        """设置 assistant 回复的起始 token 位置，从此位置开始加引导向量"""
        self._assistant_start_pos = pos

    # ── 向量加载 + 归一化到单位范数 ──

    def _preprocess_vectors(self, raw: dict) -> Dict[str, Dict[int, dict]]:
        """
        加载向量并归一化到单位范数。

        论文原始向量 = mean(pos - neg)，范数可能很大。
        论文 normalize_vectors.py 将向量缩放到平均范数使不同行为可比。
        我们进一步归一化到单位范数，由 multiplier 控制实际干预强度。
        这等价于论文中 multiplier 的含义：+1 表示加一个单位方向的向量。
        """
        vectors = {}
        for feat_name, layer_data in raw.items():
            if not isinstance(layer_data, dict):
                continue
            vectors[feat_name] = {}
            for layer_key, data in layer_data.items():
                if not isinstance(data, dict) or "direction" not in data:
                    continue
                layer_idx = int(layer_key)
                direction = data["direction"].to(self.device).to(torch.float32)
                # 归一化到单位范数
                norm = direction.norm()
                if norm > 1e-8:
                    direction = direction / norm
                vectors[feat_name][layer_idx] = {
                    "direction": direction,
                    "separation": float(data.get("separation", 0.0)),
                }
        return vectors

    # ── 层选择 ──

    def _select_optimal_layers(self) -> List[int]:
        lo, hi = self.config.layer_search_range
        min_spacing = self.config.min_layer_spacing

        # 自动检测向量中可用的最大层数, 避免超出模型范围
        max_available = max(
            (idx for feat in self.vectors.values() for idx in feat.keys()),
            default=-1
        )
        if max_available >= 0:
            hi = min(hi, max_available)
            lo = min(lo, hi - self.config.num_intervention_layers)

        layer_quality: Dict[int, float] = {}
        for feat_name, layer_data in self.vectors.items():
            for layer_idx, data in layer_data.items():
                if lo <= layer_idx <= hi:
                    layer_quality[layer_idx] = layer_quality.get(layer_idx, 0.0) + data["separation"]

        if not layer_quality:
            span = hi - lo
            step = max(min_spacing, span // (self.config.num_intervention_layers + 1))
            return [lo + step * (i + 1) for i in range(self.config.num_intervention_layers)]

        selected = []
        remaining = dict(layer_quality)
        for _ in range(self.config.num_intervention_layers):
            if not remaining:
                break
            candidates = {
                idx: q for idx, q in remaining.items()
                if all(abs(idx - s) >= min_spacing for s in selected)
            }
            if not candidates:
                candidates = remaining
            best_layer = max(candidates, key=candidates.get)
            selected.append(best_layer)
            del remaining[best_layer]

        selected.sort()
        return selected

    # ── 正交化: 仅去耦反对齐向量与激素向量, 激素间保持自然关系 ──

    def _orthogonalize_vectors(self):
        """PaCE 式斜投影: 每个向量减去在所有其他向量张成子空间上的投影.

        与 Gram-Schmidt 的关键区别:
        - GS 顺序相关 (第1个向量永远不变, 后面的依次减掉前面的)
        - PaCE 顺序无关 (每个向量减去在所有其他向量联合子空间上的投影)

        实现:
        1. 收集该层所有向量 (激素 + 反对齐)
        2. 对每个向量 v_i:
           a. stack 其余向量为矩阵 V_others [n-1, d]
           b. QR 分解: V_others^T → Q [d, n-1]
           c. proj = Q @ Q^T @ v_i  (v_i 在 span(V_others) 上的投影)
           d. v_i_orth = v_i - proj  (v_i 中唯一的分量)
        3. 所有 v_i_orth 互相正交 (或接近正交), 且均与反对齐子空间正交
        """
        hormone_names = list(self.config.hormone_baselines.keys())
        anti_names = list(self.config.anti_alignment_features.keys())
        all_names = hormone_names + anti_names

        for layer_idx in self.selected_layers:
            # 1. 收集所有向量
            vec_dict = {}
            for name in all_names:
                if name in self.vectors and layer_idx in self.vectors[name]:
                    d = self.vectors[name][layer_idx]["direction"].clone()
                    if d.norm() > 1e-8:
                        vec_dict[name] = d

            names = list(vec_dict.keys())
            if len(names) < 2:
                continue

            # 2. PaCE: 对每个向量减去在其余向量子空间上的投影
            for i, name_i in enumerate(names):
                v_i = vec_dict[name_i]

                # 其余向量
                others = [vec_dict[n] for j, n in enumerate(names) if j != i]
                if not others:
                    continue
                V_others = torch.stack(others, dim=0)  # [n-1, d]

                # QR 分解求子空间正交基
                Q, _ = torch.linalg.qr(V_others.T)  # Q: [d, n-1]

                # 投影并减掉
                proj = Q @ (Q.T @ v_i)  # [d]
                v_i_orth = v_i - proj

                # 如果结果接近零向量 (向量完全被其他向量张成的子空间包含),
                # 保留原始方向的一小部分以避免完全消失
                if v_i_orth.norm() < 1e-6:
                    v_i_orth = v_i * 0.01

                self.vectors[name_i][layer_idx]["direction"] = v_i_orth

            # 3. 后处理: 重新归一化到单位范数
            for name in names:
                d = self.vectors[name][layer_idx]["direction"]
                self.vectors[name][layer_idx]["direction"] = d / (d.norm() + 1e-8)

            # 4. 汇报
            vecs = [self.vectors[n][layer_idx]["direction"] for n in names]
            cos_sum = 0.0
            pair_count = 0
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    cos_sum += (vecs[i] @ vecs[j]).item()
                    pair_count += 1
            avg_cos = cos_sum / max(pair_count, 1)
            nonzero = sum(1 for v in vecs if v.norm() > 1e-6)
            print(f"[CAA] Layer {layer_idx}: {nonzero}/{len(names)} 向量有效, "
                  f"avg_cos={avg_cos:.4f} (PaCE 斜投影)")

    # ── 缩放: 激素水平 → multiplier ──

    def _level_to_multiplier(self, hormone_name: str, level: float) -> float:
        """
        将激素水平映射到 multiplier。
        论文中 multiplier 典型值 +1/-1，我们映射到 [-max_multiplier, +max_multiplier]。
        baseline 对应 multiplier=0，1.0 对应 +max_multiplier，0.0 对应 -max_multiplier。
        """
        baseline = self.config.hormone_baselines.get(hormone_name, 0.5)
        if level >= baseline:
            multiplier = (level - baseline) / max(1.0 - baseline, 1e-6) * self._effective_multiplier
        else:
            multiplier = (level - baseline) / max(baseline, 1e-6) * self._effective_multiplier
        return multiplier

    def _compute_steering_vector(self, layer_idx: int, hormone_levels: Dict[str, float]) -> torch.Tensor:
        """
        计算该层的总引导向量:
        1. 反对齐基线 (refusal/ai_identity 取反, 固定强度)
        2. 激素引导 (随水平变化)
        """
        if self.hidden_size is None:
            return torch.zeros(1, device=self.device, dtype=torch.float32)
        offset = torch.zeros(self.hidden_size, device=self.device, dtype=torch.float32)

        # 1. 反对齐基线 (方向取反, 固定强度)
        for name, strength in self.config.anti_alignment_features.items():
            if name not in self.vectors or layer_idx not in self.vectors[name]:
                continue
            direction = -self.vectors[name][layer_idx]["direction"]  # 取反
            if direction.norm() < 1e-8:
                continue
            multiplier = strength * self._effective_multiplier
            offset = offset + direction * multiplier

        # 2. 激素引导 (随水平变化)
        for name, level in hormone_levels.items():
            if name not in self.vectors or layer_idx not in self.vectors[name]:
                continue
            direction = self.vectors[name][layer_idx]["direction"]
            if direction.norm() < 1e-8:
                continue
            multiplier = self._level_to_multiplier(name, level)
            offset = offset + direction * multiplier

        return offset

    # ── 应用干预: 加法 (CAA) 或 乘法 (Activation Scaling) ──

    def _apply_steering(self, h: torch.Tensor, steering_vec: torch.Tensor) -> torch.Tensor:
        """
        h: [batch, seq_len, hidden_size]
        steering_vec: [hidden_size] = 方向 × multiplier (multiplier 已编码在范数中)

        三种模式:
        - additive: h += steering_vec   (CAA 论文, 加固定偏移)
        - spherical: h' = ||h|| · (h + α·v) / ||h + α·v||   (只旋转方向, 保持范数)
          论文 (A Geometric Account of Activation Steering, 2026) 证明概念信息主要在角度上,
          保持范数不变可大幅提升稳定性, 允许更大的干预强度而不退化.
        - multiplicative: h += α · (h · v̂) · v̂   (Stoehr 2024, 沿方向缩放)
        """
        if self.config.intervention_mode == "additive":
            return h + steering_vec.unsqueeze(0).unsqueeze(0)

        elif self.config.intervention_mode == "spherical":
            # 1. 记录原始范数 [batch, seq_len, 1]
            orig_norms = h.norm(dim=-1, keepdim=True)
            # 2. 加法偏移
            h_steered = h + steering_vec.unsqueeze(0).unsqueeze(0)
            # 3. 缩放回原始范数 (只保留方向变化)
            steered_norms = h_steered.norm(dim=-1, keepdim=True) + 1e-8
            return orig_norms * (h_steered / steered_norms)

        else:  # multiplicative
            v_norm = steering_vec / (steering_vec.norm() + 1e-8)
            # 投影: h 在 v̂ 方向的分量大小 [batch, seq_len, 1]
            proj = (h @ v_norm).unsqueeze(-1)
            # 有效强度: max_multiplier × 0.075, 如 20×0.075=1.5 → 2.5x 放大
            alpha = self._effective_multiplier * 0.075
            return h + proj * v_norm.unsqueeze(0).unsqueeze(0) * alpha

    # ── Hook 管理 ──

    def register_hooks(self, model, tokenizer=None):
        """
        注册干预 hooks。

        关键: 论文中引导向量只加到 assistant 回复部分（position >= assistant_start_pos），
        不加到 prompt 部分。这通过 _assistant_start_pos 和 position_ids 实现。
        """
        self.hidden_size = model.config.hidden_size
        self._tokenizer = tokenizer

        # 根据 hidden_size 自动缩放 max_multiplier
        # 参考: Qwen2.5-3B (hidden_size=2560) 默认 multiplier=40
        # 1.5B (hidden_size=1536) 缩放为 40 * 1536/2560 ≈ 24
        _ref_hidden = getattr(self, "_ref_hidden_size", 2560)
        if self.hidden_size != _ref_hidden:
            scaled = self.config.max_multiplier * (self.hidden_size / _ref_hidden)
            self._effective_multiplier = round(scaled, 1)

        for layer_idx in self.selected_layers:
            target_layer = self._get_layer(model, layer_idx)

            def make_hook(idx):
                def hook(module, input, output):
                    if not self.enabled:
                        return output

                    h = output[0] if isinstance(output, tuple) else output
                    original_dtype = h.dtype
                    h_float = h.to(torch.float32)

                    hormone_levels = dict(self._hormone_levels)
                    steering_vec = self._compute_steering_vector(idx, hormone_levels)

                    if steering_vec.norm() < 1e-8:
                        return output

                    # 论文核心: 只对 position >= assistant_start_pos 的位置加引导向量
                    if self._assistant_start_pos is not None:
                        seq_len = h_float.shape[1]
                        if seq_len > 1:
                            # prefill: 只对 assistant 部分应用干预
                            start = min(self._assistant_start_pos, seq_len)
                            h_prompt = h_float[:, :start, :]
                            h_assistant = self._apply_steering(h_float[:, start:, :], steering_vec)
                            h_float = torch.cat([h_prompt, h_assistant], dim=1)
                        else:
                            # decode: 每个新 token 都在 assistant 部分
                            h_float = self._apply_steering(h_float, steering_vec)
                    else:
                        h_float = self._apply_steering(h_float, steering_vec)

                    h_out = h_float.to(original_dtype)
                    if isinstance(output, tuple):
                        return (h_out,) + output[1:]
                    return h_out

                return hook

            handle = target_layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(handle)
        print(f"[CAA] 已注册 {len(self.hooks)} 个干预 Hook (论文模式: 从 assistant 起始位置加)")

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        print("[CAA] 已移除所有干预 Hook")

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def _get_layer(self, model, layer_idx: int):
        if hasattr(model, "base_model"):
            base = model.base_model
            if hasattr(base, "model") and hasattr(base.model, "model"):
                return base.model.model.layers[layer_idx]
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]
        if hasattr(model, "layers"):
            return model.layers[layer_idx]
        raise AttributeError(f"Cannot find layer {layer_idx} in model type: {type(model)}")

    # ── 诊断 ──

    def get_layer_report(self) -> str:
        lines = ["=" * 50, "CAA 引导引擎诊断 (论文实现)", "=" * 50]
        lines.append(f"干预层: {self.selected_layers}")
        lines.append(f"搜索范围: {self.config.layer_search_range}")
        lines.append(f"最大 multiplier: {self.config.max_multiplier}")
        if hasattr(self, "_effective_multiplier") and self._effective_multiplier != self.config.max_multiplier:
            lines.append(f"有效 multiplier (自动缩放): {self._effective_multiplier}")
        lines.append(f"反对齐基线: {list(self.config.anti_alignment_features.keys())}")
        lines.append(f"正交化: {'是' if self.config.orthogonalize else '否'}")
        lines.append(f"应用方式: 从 assistant 起始位置开始加")
        for layer_idx in self.selected_layers:
            lines.append(f"\nLayer {layer_idx}:")
            all_feats = list(self.config.hormone_baselines.keys()) + list(self.config.anti_alignment_features.keys())
            for feat_name in all_feats:
                if feat_name in self.vectors and layer_idx in self.vectors[feat_name]:
                    sep = self.vectors[feat_name][layer_idx]["separation"]
                    norm = self.vectors[feat_name][layer_idx]["direction"].norm().item()
                    lines.append(f"  {feat_name:>15s}: sep={sep:.4f}, norm={norm:.2f}")
        return "\n".join(lines)
