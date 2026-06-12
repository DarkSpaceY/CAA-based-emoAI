"""
向量加载与正交化工具。

提供 load_vectors() 和 orthonormalize_pace() 两个核心函数,
供 CAA 引擎和向量计算脚本共用。
"""
import torch


def model_id_to_filename(model_id: str) -> str:
    """将模型 ID 转换为文件名友好的短名称。

    Examples:
        Qwen/Qwen2.5-3B-Instruct → qwen2.5-3b-instruct
        google/gemma-2-2b-it → gemma-2-2b-it
    """
    name = model_id.split("/")[-1]  # 取最后一段
    return name.lower()


def load_vectors(path: str, device: str = "cpu") -> dict:
    """加载 .pt 格式的向量文件, 返回 {feature_name: {layer_idx: {direction, separation, ...}}}"""
    data = torch.load(path, map_location=device, weights_only=True)
    return data


def orthonormalize_pace(vectors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    PaCE (Parallel Concept Editing) 斜投影正交化。

    用 SVD 一次性分解所有引导方向, 只保留唯一的方向成分。
    相比逐对 Gram-Schmidt:
    - 不会把近似平行的向量清零 (如 COR/NE 互补)
    - 所有激素向量保持完整的秩
    - 输出严格正交但不破坏空间覆盖

    Args:
        vectors: {name: direction_vec} 字典, 每个 vec shape [hidden_size]

    Returns:
        {name: orthogonalized_vec} 字典, 形状不变
    """
    names = list(vectors.keys())
    stacked = torch.stack([vectors[n] for n in names], dim=0)  # [k, d]

    # SVD 分解概念空间
    U, S, Vt = torch.linalg.svd(stacked, full_matrices=False)

    # 用右奇异向量重构: 每对概念只保留独立方向成分
    # 等价于在概念空间内做 Gram-Schmidt 但不丢失秩
    ortho = torch.zeros_like(stacked)
    for i in range(len(names)):
        # 从第 i 个方向中减去前 i-1 个奇异方向上的投影
        v = stacked[i].clone()
        for j in range(i):
            v = v - (v @ Vt[j]) * Vt[j]
        # 如果被减光了 (概念完全冗余), 保留极小量避免零向量
        if v.norm() < 1e-6:
            v = torch.randn_like(v) * 1e-6
        ortho[i] = v / (v.norm() + 1e-8) * stacked[i].norm()

    return {names[i]: ortho[i] for i in range(len(names))}


def select_best_layers(
    vectors_data: dict,
    num_layers: int = 5,
    min_sep: float = 5.0,
) -> tuple[list[int], float]:
    """
    从全层级向量数据中选出分离度最高的连续层区间。

    Args:
        vectors_data: {feature_name: {layer_idx: {direction, separation, ...}}}
        num_layers: 期望的层数
        min_sep: 最小接受分离度

    Returns:
        (selected_layers, avg_separation) 元组
    """
    # 收集所有特征在各层的平均分离度
    feature_names = list(vectors_data.keys())
    num_total_layers = len(vectors_data[feature_names[0]])

    layer_scores = []
    for lidx in range(num_total_layers):
        sep_sum = 0.0
        for fn in feature_names:
            sep_sum += vectors_data[fn][lidx].get("separation", 0.0)
        layer_scores.append((lidx, sep_sum / len(feature_names)))

    # 滑动窗口找最高平均分离度的连续区间
    best_start, best_score = 0, 0.0
    for start in range(num_total_layers - num_layers + 1):
        score = sum(s for _, s in layer_scores[start: start + num_layers])
        if score > best_score:
            best_score = score
            best_start = start

    selected = list(range(best_start, best_start + num_layers))
    avg_sep = best_score / num_layers

    if avg_sep < min_sep:
        raise ValueError(
            f"所选层 [{selected[0]}, {selected[-1]}] 平均分离度 {avg_sep:.1f} "
            f"低于阈值 {min_sep}。请检查向量文件或增大 num_layers。"
        )

    return selected, avg_sep
