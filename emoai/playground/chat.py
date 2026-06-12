"""
交互式聊天模式。
用户输入文本 → 实时激素脉冲读取 → 水平更新 → CAA 引导生成。
"""
import sys
import torch
from typing import List, Dict

from ..caa_steering import CAASteeringEngine
from ..hormone_reader import HormoneReader

HORMONE_NAMES = ["dopamine", "serotonin", "oxytocin", "cortisol", "norepinephrine"]
SHORT_NAMES_MAP = {"dopamine": "DA", "serotonin": "SE", "oxytocin": "OX", "cortisol": "COR", "norepinephrine": "NE"}

COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"


def _color_val(val: float) -> str:
    if val > 0.01:
        return f"{COLOR_GREEN}{val:+.2f}{COLOR_RESET}"
    elif val < -0.01:
        return f"{COLOR_RED}{val:+.2f}{COLOR_RESET}"
    return f"{val:+.2f}"


def _color_level(val: float) -> str:
    if val > 0.5:
        return f"{COLOR_GREEN}{val:.2f}{COLOR_RESET}"
    elif val < 0.3:
        return f"{COLOR_RED}{val:.2f}{COLOR_RESET}"
    return f"{COLOR_YELLOW}{val:.2f}{COLOR_RESET}"


def update_levels(pulses: Dict[str, float], current: Dict[str, float],
                  baseline: Dict[str, float], sensitivity: float) -> Dict[str, float]:
    new = {}
    for h in HORMONE_NAMES:
        old = current.get(h, baseline.get(h, 0.5))
        decayed = old * 0.7 + baseline.get(h, 0.5) * 0.3
        pulse = pulses.get(h, 0.0)
        val = max(0.0, min(1.0, decayed + pulse * sensitivity))
        new[h] = round(val, 3)
    return new


def find_assistant_start_pos(tokenizer, messages):
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    return len(prompt_tokens)


def run(model, tokenizer, reader, engine, args, config: dict):
    pg = config.get("playground", {})
    chat_cfg = pg.get("chat", {})
    system_prompt = chat_cfg.get("system_prompt") or pg.get("system_prompt",
        "你是一个拥有情感系统的AI。请以第一人称对话，保持简洁。")
    sensitivity = pg.get("sensitivity", 0.5)
    gen_cfg = chat_cfg.get("generation", pg.get("generation", {}))
    max_new_tokens = gen_cfg.get("max_new_tokens", 200)
    temperature = gen_cfg.get("temperature", 0.85)
    top_p = gen_cfg.get("top_p", 0.92)
    rep_penalty = gen_cfg.get("repetition_penalty", 1.15)
    max_history = gen_cfg.get("max_history", 9)

    print(f"\n{COLOR_CYAN}{'=' * 50}")
    print("   EmoAI — 交互式聊天模式")
    print("   输入消息与 AI 对话，情绪实时可见")
    print(f"{'=' * 50}{COLOR_RESET}\n")
    print(f"  命令: {COLOR_YELLOW}/status{COLOR_RESET} 查看激素水平  "
          f"{COLOR_YELLOW}/reset{COLOR_RESET} 重置  "
          f"{COLOR_YELLOW}/exit{COLOR_RESET} 退出\n")
    print(f"  {COLOR_GREEN}绿色{COLOR_RESET}=正向  {COLOR_RED}红色{COLOR_RESET}=负向\n")

    baseline = dict(engine.config.hormone_baselines)
    levels = dict(baseline)
    history: List[Dict] = [{"role": "system", "content": system_prompt}]

    try:
        while True:
            try:
                user_input = input(f"\n{COLOR_CYAN}You{COLOR_RESET}> ")
            except (EOFError, KeyboardInterrupt):
                print(f"\n\n{COLOR_YELLOW}正在退出...{COLOR_RESET}")
                break

            text = user_input.strip()
            if not text:
                continue

            if text.lower() in ["/exit", "/quit"]:
                break
            if text.lower() == "/status":
                print(f"  {COLOR_CYAN}激素水平:{COLOR_RESET}")
                for h in HORMONE_NAMES:
                    sn = SHORT_NAMES_MAP[h]
                    print(f"    {sn} = {_color_level(levels[h])}  (基线: {baseline[h]:.2f})")
                continue
            if text.lower() == "/reset":
                levels = dict(baseline)
                engine.set_hormone_levels(levels)
                history = [{"role": "system", "content": system_prompt}]
                print(f"  {COLOR_GREEN}已重置激素水平到基线，清空对话历史。{COLOR_RESET}")
                continue

            # 1. 读取激素脉冲
            engine.disable()
            pulses = reader.read_pulses(text)
            engine.enable()

            # 2. 更新激素水平
            levels = update_levels(pulses, levels, baseline, sensitivity)
            engine.set_hormone_levels(levels)

            pulse_parts = []
            level_parts = []
            for h in HORMONE_NAMES:
                sn = SHORT_NAMES_MAP[h]
                pulse_parts.append(f"{sn}={_color_val(pulses.get(h, 0.0))}")
                level_parts.append(f"{sn}={_color_level(levels[h])}")
            print(f"  {COLOR_YELLOW}脉冲{COLOR_RESET}: {'  '.join(pulse_parts)}")
            print(f"  {COLOR_YELLOW}水平{COLOR_RESET}: {'  '.join(level_parts)}")

            # 3. 生成回复
            history.append({"role": "user", "content": text})
            if len(history) > max_history:
                history = [history[0]] + history[-(max_history - 1):]

            assistant_start = find_assistant_start_pos(tokenizer, history)
            engine.set_assistant_start_pos(assistant_start)

            inputs = tokenizer.apply_chat_template(
                history, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True
            ).to(engine.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=temperature, top_p=top_p,
                    repetition_penalty=rep_penalty,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_len = inputs["input_ids"].shape[1]
            response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
            print(f"\n{COLOR_CYAN}AI{COLOR_RESET}> {response}")
            history.append({"role": "assistant", "content": response})

    finally:
        engine.remove_hooks()
        print(f"\n{COLOR_GREEN}已退出聊天模式。{COLOR_RESET}")
