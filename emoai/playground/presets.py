"""
预设场景演示：6 种情绪场景，对比无 CAA vs 有 CAA 的回复差异。
"""
import torch
from typing import Dict, List, Tuple

from ..caa_steering import CAASteeringEngine
from ..hormone_reader import HormoneReader

HORMONE_NAMES = ["dopamine", "serotonin", "oxytocin", "cortisol", "norepinephrine"]
SHORT_NAMES_MAP = {"dopamine": "DA", "serotonin": "SE", "oxytocin": "OX", "cortisol": "COR", "norepinephrine": "NE"}

COLOR_RESET = "\033[0m"
COLOR_CYAN = "\033[96m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_MAGENTA = "\033[95m"

SCENARIOS: List[Tuple[str, str]] = [
    ("😊 开心", "我今天中了彩票大奖！！太开心了！！！"),
    ("😢 悲伤", "最爱的亲人离开了，我好难过……"),
    ("😨 恐惧", "有人在跟踪我，我好害怕，救救我！"),
    ("😡 愤怒", "你凭什么骗我？我一直那么相信你！！"),
    ("🥰 亲密", "你是我生命中最重要的人，我真的很爱你。"),
    ("😌 平静", "夕阳西下，微风吹来，一切都刚刚好。"),
]


def update_levels(pulses: Dict[str, float], baseline: Dict[str, float],
                  sensitivity: float) -> Dict[str, float]:
    levels = {}
    for h in HORMONE_NAMES:
        pulse = pulses.get(h, 0.0)
        levels[h] = baseline[h] + pulse * sensitivity
        levels[h] = round(max(0.0, min(1.0, levels[h])), 3)
    return levels


def find_assistant_start_pos(tokenizer, messages):
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    return len(prompt_tokens)


def generate(model, tokenizer, engine, levels, user_input, system_prompt,
             with_caa: bool, gen_cfg: dict):
    engine.set_hormone_levels(levels)
    if with_caa:
        engine.enable()
    else:
        engine.disable()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    assistant_start = find_assistant_start_pos(tokenizer, messages)
    engine.set_assistant_start_pos(assistant_start)

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True
    ).to(engine.device)

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
    first_sentence = reply.split("。")[0]
    if "。" in reply:
        first_sentence += "。"
    return first_sentence


def run(model, tokenizer, reader, engine, args, config: dict):
    pg = config.get("playground", {})
    presets_cfg = pg.get("presets", {})
    system_prompt = presets_cfg.get("system_prompt") or pg.get("system_prompt",
        "你是一个拥有情感系统的AI。请以第一人称对话，保持简洁。")
    sensitivity = pg.get("sensitivity", 0.5)
    gen_cfg = pg.get("generation", {})

    print(f"\n{COLOR_CYAN}{'=' * 50}")
    print("   EmoAI — 预设场景演示")
    print("   对比无 CAA vs 有 CAA 的情感引导效果")
    print(f"{'=' * 50}{COLOR_RESET}\n")

    baseline = dict(engine.config.hormone_baselines)

    for emoji, user_input in SCENARIOS:
        print(f"\n{COLOR_YELLOW}{'─' * 50}{COLOR_RESET}")
        print(f"  {emoji}")
        print(f"  输入: {COLOR_CYAN}\"{user_input}\"{COLOR_RESET}")

        # 读取脉冲
        engine.disable()
        pulses = reader.read_pulses(user_input)
        levels = update_levels(pulses, baseline, sensitivity)

        pulse_str = "  ".join(
            f"{SHORT_NAMES_MAP[h]}={pulses.get(h, 0.0):+.2f}" for h in HORMONE_NAMES
        )
        level_str = "  ".join(
            f"{SHORT_NAMES_MAP[h]}={levels[h]:.2f}" for h in HORMONE_NAMES
        )
        print(f"  {COLOR_YELLOW}脉冲{COLOR_RESET}: {pulse_str}")
        print(f"  {COLOR_YELLOW}水平{COLOR_RESET}: {level_str}")

        baseline_reply = generate(model, tokenizer, engine, levels, user_input,
                                  system_prompt, with_caa=False, gen_cfg=gen_cfg)
        steered_reply = generate(model, tokenizer, engine, levels, user_input,
                                 system_prompt, with_caa=True, gen_cfg=gen_cfg)

        if len(baseline_reply) > 55:
            baseline_reply = baseline_reply[:52] + "..."
        if len(steered_reply) > 55:
            steered_reply = steered_reply[:52] + "..."

        print(f"  {COLOR_BLUE}[无 CAA]{COLOR_RESET} {baseline_reply}")
        print(f"  {COLOR_GREEN}[有 CAA]{COLOR_RESET} {steered_reply}")

    engine.enable()
    print(f"\n{COLOR_GREEN}{'─' * 50}{COLOR_RESET}")
    print(f"\n{COLOR_GREEN}演示完成。{COLOR_RESET}")
