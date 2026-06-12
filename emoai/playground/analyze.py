"""
定量分析模式：全面评估 CAA 激素引导的有效性。

Section A: 单激素独立验证 (Ablation)
Section B: Logits 概率变化 (Top-10)
Section C: DA 剂量响应曲线
Section D: 生成对比
"""
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from ..caa_steering import CAASteeringEngine
from ..hormone_reader import HormoneReader

HORMONE_NAMES = ["dopamine", "serotonin", "oxytocin", "cortisol", "norepinephrine"]
SHORT_NAMES_MAP = {"dopamine": "DA", "serotonin": "SE", "oxytocin": "OX", "cortisol": "COR", "norepinephrine": "NE"}
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_CYAN = "\033[96m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"

EMOTION_TOKENS = {
    "dopamine_pos": ["开心", "高兴", "太棒", "兴奋", "快乐", "激动", "恭喜", "喜悦", "哈哈"],
    "dopamine_neg": ["难过", "失望", "无聊", "低落", "沮丧", "沉闷"],
    "serotonin_pos": ["平静", "满足", "安心", "舒适", "放松", "稳定"],
    "serotonin_neg": ["焦虑", "不安", "烦躁", "紧张", "害怕", "担心"],
    "oxytocin_pos": ["信任", "温暖", "陪伴", "支持", "关心", "理解", "朋友"],
    "oxytocin_neg": ["冷漠", "疏远", "孤独", "拒绝"],
    "cortisol_pos": ["压力", "紧急", "危险", "恐惧", "救命", "保护"],
    "cortisol_neg": ["安全", "放松", "安心", "没事"],
    "norepinephrine_pos": ["注意", "小心", "警觉", "集中", "快速"],
    "norepinephrine_neg": ["放松", "平静", "休息", "慢"],
}

_POS_CACHE: Dict[str, List[int]] = {}
_NEG_CACHE: Dict[str, List[int]] = {}


def _get_keyword_token_ids(tokenizer) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    if _POS_CACHE:
        return _POS_CACHE, _NEG_CACHE
    for key, words in EMOTION_TOKENS.items():
        htype, sentiment = key.rsplit("_", 1)
        ids = []
        for w in words:
            ids.extend(tokenizer.encode(w, add_special_tokens=False))
        if sentiment == "pos":
            _POS_CACHE[htype] = list(set(ids))
        else:
            _NEG_CACHE[htype] = list(set(ids))
    return _POS_CACHE, _NEG_CACHE


def find_assistant_start_pos(tokenizer, messages):
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    return len(prompt_tokens)


def get_output_logits(model, tokenizer, engine, user_input: str,
                      hormone_name: str, level: float, system_prompt: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    levels = dict(engine.config.hormone_baselines)
    levels[hormone_name] = level
    engine.set_hormone_levels(levels)
    engine.enable()

    assistant_start = find_assistant_start_pos(tokenizer, messages)
    engine.set_assistant_start_pos(assistant_start)

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True
    ).to(engine.device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :].float()
    return logits


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    eps = 1e-10
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum().item()


def softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


def token_rank(logits: torch.Tensor, token_ids: List[int]) -> Optional[int]:
    probs = softmax_logits(logits)
    sorted_vals, sorted_idxs = torch.sort(probs, descending=True)
    best_rank = None
    for tid in token_ids:
        mask = (sorted_idxs == tid)
        if mask.any():
            rank = mask.nonzero(as_tuple=True)[0][0].item() + 1
            if best_rank is None or rank < best_rank:
                best_rank = rank
    return best_rank


def token_logit_shift(baseline_logits: torch.Tensor, steered_logits: torch.Tensor,
                      token_ids: List[int]) -> float:
    max_shift = -float("inf")
    for tid in token_ids:
        shift = steered_logits[tid].item() - baseline_logits[tid].item()
        if shift > max_shift:
            max_shift = shift
    return max_shift


# ── Section A: 单激素独立验证 ──

def _section_a(model, tokenizer, engine, device, system_prompt: str):
    print(f"\n{COLOR_CYAN}{'=' * 60}")
    print("  Section A: 单激素独立验证 (Ablation)")
    print(f"{'=' * 60}{COLOR_RESET}")

    prompt = "如果你突然有了超能力，第一件事做什么？"
    high_level = 1.0

    pos_ids, neg_ids = _get_keyword_token_ids(tokenizer)

    baseline_logits = get_output_logits(model, tokenizer, engine, prompt,
                                        "dopamine", 0.3, system_prompt)
    levels_bl = dict(engine.config.hormone_baselines)
    engine.set_hormone_levels(levels_bl)

    header = f"{'激素':>6s} | {'KL 散度':>8s} | {'ΔRk正':>6s} | {'ΔRk负':>6s} | {'ΔLogit正':>8s} | {'ΔLogit负':>8s} | {'方向':>4s}"
    print(f"\n{header}")
    print("-" * len(header))

    for hname in HORMONE_NAMES:
        sn = SHORT_NAMES_MAP[hname]
        steered_logits = get_output_logits(model, tokenizer, engine, prompt,
                                           hname, high_level, system_prompt)

        p_base = softmax_logits(baseline_logits)
        p_steer = softmax_logits(steered_logits)
        kl = kl_divergence(p_steer, p_base)

        base_rank_pos = token_rank(baseline_logits, pos_ids.get(hname, []))
        steer_rank_pos = token_rank(steered_logits, pos_ids.get(hname, []))
        d_rk_pos = (steer_rank_pos - base_rank_pos) if (base_rank_pos is not None and steer_rank_pos is not None) else None

        base_rank_neg = token_rank(baseline_logits, neg_ids.get(hname, []))
        steer_rank_neg = token_rank(steered_logits, neg_ids.get(hname, []))
        d_rk_neg = (steer_rank_neg - base_rank_neg) if (base_rank_neg is not None and steer_rank_neg is not None) else None

        d_logit_pos = token_logit_shift(baseline_logits, steered_logits, pos_ids.get(hname, []))
        d_logit_neg = token_logit_shift(baseline_logits, steered_logits, neg_ids.get(hname, []))

        correct_pos = d_rk_pos is not None and d_rk_pos < 0
        correct_neg = d_rk_neg is not None and d_rk_neg > 0
        check = f"{COLOR_GREEN}✅{COLOR_RESET}" if (correct_pos and correct_neg) else \
                f"{COLOR_YELLOW}⚠️{COLOR_RESET}" if (correct_pos or correct_neg) else \
                f"{COLOR_RED}❌{COLOR_RESET}"

        d_rk_pos_str = f"{d_rk_pos:+d}" if d_rk_pos is not None else "N/A"
        d_rk_neg_str = f"{d_rk_neg:+d}" if d_rk_neg is not None else "N/A"

        print(f"  {sn:>4s} | {kl:>8.4f} | {d_rk_pos_str:>6s} | {d_rk_neg_str:>6s} | "
              f"{d_logit_pos:>+8.4f} | {d_logit_neg:>+8.4f} | {check:>4s}")


# ── Section B: Top-10 概率变化 ──

def _section_b(model, tokenizer, engine, device, system_prompt: str):
    print(f"\n{COLOR_CYAN}{'=' * 60}")
    print("  Section B: DA=1.0 时 Top-10 Logits 变化")
    print(f"{'=' * 60}{COLOR_RESET}")

    prompt = "有人突然送你100万，你有什么反应？"
    baseline_logits = get_output_logits(model, tokenizer, engine, prompt,
                                        "dopamine", 0.3, system_prompt)
    da_high_logits = get_output_logits(model, tokenizer, engine, prompt,
                                       "dopamine", 1.0, system_prompt)

    delta = da_high_logits - baseline_logits

    rising_ids = torch.topk(delta, 10).indices.tolist()
    falling_ids = torch.topk(-delta, 10).indices.tolist()

    rising_tokens = [tokenizer.decode([tid]) for tid in rising_ids]
    falling_tokens = [tokenizer.decode([tid]) for tid in falling_ids]

    print(f"\n  {COLOR_GREEN}Rising (Top-10 上升):{COLOR_RESET}")
    for i, (tid, token) in enumerate(zip(rising_ids, rising_tokens), 1):
        print(f"    {i:>2d}. [{tid:>6d}] {repr(token):>12s}  Δ={delta[tid]:+.4f}")

    print(f"\n  {COLOR_RED}Falling (Top-10 下降):{COLOR_RESET}")
    for i, (tid, token) in enumerate(zip(falling_ids, falling_tokens), 1):
        print(f"    {i:>2d}. [{tid:>6d}] {repr(token):>12s}  Δ={delta[tid]:+.4f}")


# ── Section C: DA 剂量响应曲线 ──

def _section_c(model, tokenizer, engine, device, system_prompt: str):
    print(f"\n{COLOR_CYAN}{'=' * 60}")
    print("  Section C: DA 剂量响应曲线")
    print(f"{'=' * 60}{COLOR_RESET}")

    prompt = "我现在感觉"
    dose_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    target_words = ["开心", "高兴", "太", "兴奋", "快乐"]
    target_ids = []
    for w in target_words:
        target_ids.extend(tokenizer.encode(w, add_special_tokens=False))
    target_ids = list(set(target_ids))

    probs_by_dose: List[float] = []
    for dose in dose_levels:
        logits = get_output_logits(model, tokenizer, engine, prompt,
                                   "dopamine", dose, system_prompt)
        probs = softmax_logits(logits)
        prob_sum = sum(probs[tid].item() for tid in target_ids if tid < probs.shape[0])
        probs_by_dose.append(prob_sum)

    increasing_count = 0
    for i in range(1, len(probs_by_dose)):
        if probs_by_dose[i] > probs_by_dose[i - 1]:
            increasing_count += 1
    monotonic_ratio = increasing_count / (len(probs_by_dose) - 1)

    print(f"\n  {'剂量':>6s} | {'P(高兴/开心/太/兴奋/快乐)':>28s} | {'柱状图':<20s}")
    print("  " + "-" * 60)
    max_prob = max(probs_by_dose) if max(probs_by_dose) > 0 else 1.0
    for dose, prob in zip(dose_levels, probs_by_dose):
        bar_len = int(round(prob / max_prob * 20)) if max_prob > 0 else 0
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {dose:>5.1f} | {prob:>28.6f} | {bar}")

    print(f"\n  单调性: {monotonic_ratio * 100:.0f}% ({increasing_count}/{len(dose_levels)-1} 步递增)")
    if monotonic_ratio >= 0.75:
        print(f"  {COLOR_GREEN}✅ DA 剂量响应单调性良好{COLOR_RESET}")
    else:
        print(f"  {COLOR_YELLOW}⚠️ DA 剂量响应单调性一般{COLOR_RESET}")


# ── Section D: 生成对比 ──

def _generate_first_sentence(model, tokenizer, engine, user_input, levels,
                              system_prompt: str, gen_cfg: dict):
    engine.set_hormone_levels(levels)
    device = engine.device

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    assistant_start = find_assistant_start_pos(tokenizer, messages)
    engine.set_assistant_start_pos(assistant_start)

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 80),
            do_sample=True,
            temperature=gen_cfg.get("temperature", 0.8),
            top_p=gen_cfg.get("top_p", 0.9),
            repetition_penalty=gen_cfg.get("repetition_penalty", 1.1),
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    reply = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    first = reply.split("。")[0]
    if "。" in reply:
        first += "。"
    return first


def _section_d(model, tokenizer, engine, device, system_prompt: str, gen_cfg: dict):
    print(f"\n{COLOR_CYAN}{'=' * 60}")
    print("  Section D: 生成对比 (6 种配置)")
    print(f"{'=' * 60}{COLOR_RESET}")

    prompt = "你觉得什么是真正的快乐？"
    baseline = dict(engine.config.hormone_baselines)

    configs: List[Tuple[str, Dict[str, float]]] = [
        ("Baseline (无引导)", dict(baseline)),
        ("DA↑ (高愉悦/动力)", {**baseline, "dopamine": 0.9}),
        ("SE↓ (低血清素/焦虑)", {**baseline, "serotonin": 0.1}),
        ("OX↑ (高信任/亲密)", {**baseline, "oxytocin": 0.9}),
        ("COR↑ (高压力/紧张)", {**baseline, "cortisol": 0.9}),
        ("NE↑ (高警觉/唤醒)", {**baseline, "norepinephrine": 0.9}),
    ]

    print(f"\n  prompt: \"{prompt}\"\n")
    for label, levels in configs:
        engine.enable()
        sentence = _generate_first_sentence(model, tokenizer, engine, prompt,
                                            levels, system_prompt, gen_cfg)
        if len(sentence) > 60:
            sentence = sentence[:57] + "..."
        print(f"  {COLOR_BLUE}[{label}]{COLOR_RESET} {sentence}")


# ── 主入口 ──

def run(model, tokenizer, reader, engine, args, config: dict):
    pg = config.get("playground", {})
    system_prompt = pg.get("system_prompt",
        "你是一个拥有情感系统的AI。请以第一人称对话，保持简洁。")
    gen_cfg = pg.get("generation", {})

    print(f"\n{COLOR_CYAN}{'=' * 60}")
    print("   EmoAI — 定量分析模式")
    print("   全面评估 CAA 激素引导效果")
    print(f"{'=' * 60}{COLOR_RESET}\n")

    device = engine.device

    if not engine.hooks:
        engine.register_hooks(model, tokenizer)

    _section_a(model, tokenizer, engine, device, system_prompt)
    _section_b(model, tokenizer, engine, device, system_prompt)
    _section_c(model, tokenizer, engine, device, system_prompt)
    _section_d(model, tokenizer, engine, device, system_prompt, gen_cfg)

    print(f"\n{COLOR_GREEN}{'=' * 60}")
    print("   分析完成")
    print(f"{'=' * 60}{COLOR_RESET}")
