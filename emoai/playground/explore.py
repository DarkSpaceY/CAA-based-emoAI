"""
激素沙盘模式：交互式调整激素水平，实时观察 AI 回复变化。
"""
import sys
import tty
import termios
import torch
from typing import Dict, List

from ..caa_steering import CAASteeringEngine
from ..hormone_reader import HormoneReader

HORMONE_NAMES = ["dopamine", "serotonin", "oxytocin", "cortisol", "norepinephrine"]
SHORT_NAMES = ["DA", "SE", "OX", "COR", "NE"]
SHORT_NAMES_MAP = {"dopamine": "DA", "serotonin": "SE", "oxytocin": "OX", "cortisol": "COR", "norepinephrine": "NE"}
COLOR_RESET = "\033[0m"
COLOR_CYAN = "\033[96m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_MAGENTA = "\033[95m"
COLOR_DIM = "\033[2m"

BAR_WIDTH = 20
HORMONE_COLORS = {
    "DA": COLOR_GREEN, "SE": COLOR_BLUE, "OX": COLOR_MAGENTA,
    "COR": COLOR_RED, "NE": COLOR_YELLOW,
}

DIRECTION_LABELS = {
    "DA": ("愉悦/动力 ↑", "低落/无趣 ↓"),
    "SE": ("满足/平静 ↑", "焦虑/不安 ↓"),
    "OX": ("信任/亲密 ↑", "疏离/冷漠 ↓"),
    "COR": ("紧张/恐惧 ↑", "放松/安全 ↓"),
    "NE": ("警觉/专注 ↑", "疲倦/放松 ↓"),
}


def _make_bar(val: float) -> str:
    filled = int(round(val * BAR_WIDTH))
    filled = max(0, min(BAR_WIDTH, filled))
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def _hormone_line(name: str, val: float, marker: str) -> str:
    """返回格式统一的单行激素显示，长度固定。"""
    color = HORMONE_COLORS.get(name, COLOR_RESET)
    bar = _make_bar(val)
    pct = f"{val:.2f}".rjust(5)
    if val > 0.55:
        direction = DIRECTION_LABELS[name][0]
    elif val < 0.45:
        direction = DIRECTION_LABELS[name][1]
    else:
        direction = "⚌  平衡  ⚌"
    return (
        f"  {marker} "
        f"[{color}{name}{COLOR_RESET}] "
        f"{color}{bar}{COLOR_RESET} "
        f"{pct}  "
        f"{COLOR_DIM}{direction}{COLOR_RESET}"
    )


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def find_assistant_start_pos(tokenizer, messages):
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    return len(prompt_tokens)


def _generate_reply(model, tokenizer, engine, history, gen_cfg: dict):
    assistant_start = find_assistant_start_pos(tokenizer, history)
    engine.set_assistant_start_pos(assistant_start)
    device = engine.device

    inputs = tokenizer.apply_chat_template(
        history, tokenize=True, add_generation_prompt=True,
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
    response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    return response


def run(model, tokenizer, reader, engine, args, config: dict):
    pg = config.get("playground", {})
    system_prompt = pg.get("system_prompt",
        "你是一个拥有情感系统的AI。请以第一人称对话，保持简洁。")
    gen_cfg = pg.get("generation", {})

    print(f"\n{COLOR_CYAN}{'=' * 50}")
    print("   EmoAI — 激素沙盘模式")
    print("   交互式调整激素水平，观察 AI 情感变化")
    print(f"{'=' * 50}{COLOR_RESET}\n")

    levels = dict(engine.config.hormone_baselines)
    selected_idx = 0
    prompt_text = ""

    engine.disable()
    try:
        prompt_text = input(f"{COLOR_CYAN}输入一个话题或问题:{COLOR_RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        engine.remove_hooks()
        return

    if not prompt_text:
        prompt_text = "周末打算做什么？"

    engine.enable()

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]

    engine.set_hormone_levels(levels)
    response = _generate_reply(model, tokenizer, engine, history, gen_cfg)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def _rl(text: str):
        """raw mode 下 print，用 \\r\\n 替代 \\n 防止列偏移"""
        sys.stdout.write(text + '\r\n')
        sys.stdout.flush()

    try:
        tty.setraw(sys.stdin)

        while True:
            clear_screen()
            _rl(f"  {COLOR_CYAN}EmoAI 激素沙盘{COLOR_RESET}")
            _rl(f"  {COLOR_DIM}Prompt:{COLOR_RESET} \"{prompt_text}\"")
            _rl('')

            for i, (short, full) in enumerate(zip(SHORT_NAMES, HORMONE_NAMES)):
                marker = "▸" if i == selected_idx else " "
                line = _hormone_line(short, levels[full], marker)
                _rl(line)

            # 汇总方向
            active_dirs = []
            for short, full in zip(SHORT_NAMES, HORMONE_NAMES):
                v = levels[full]
                if v > 0.55:
                    active_dirs.append(f"{short}↑")
                elif v < 0.45:
                    active_dirs.append(f"{short}↓")
            if active_dirs:
                dir_str = "  ".join(active_dirs)
                _rl('')
                _rl(f"  {COLOR_DIM}主导方向:{COLOR_RESET} {dir_str}")

            _rl('')
            _rl(f"{COLOR_GREEN}> {response}{COLOR_RESET}")

            _rl('')
            _rl(f"{COLOR_DIM}Controls: [1-5] 选激素  [↑/↓] +-0.1  [Enter] 重生成  [q] 退出{COLOR_RESET}")

            ch = sys.stdin.read(1)

            if ch == "q":
                _rl('')
                _rl(f"{COLOR_GREEN}退出沙盘模式。{COLOR_RESET}")
                break
            elif ch == "\r" or ch == "\n":
                engine.set_hormone_levels(levels)
                response = _generate_reply(model, tokenizer, engine, history, gen_cfg)
            elif ch in "12345":
                selected_idx = int(ch) - 1
            elif ch == "\x1b":
                next1 = sys.stdin.read(1)
                next2 = sys.stdin.read(1)
                if next1 == "[" and next2 == "A":
                    full = HORMONE_NAMES[selected_idx]
                    levels[full] = min(1.0, levels[full] + 0.1)
                    engine.set_hormone_levels(levels)
                    response = _generate_reply(model, tokenizer, engine, history, gen_cfg)
                elif next1 == "[" and next2 == "B":
                    full = HORMONE_NAMES[selected_idx]
                    levels[full] = max(0.0, levels[full] - 0.1)
                    engine.set_hormone_levels(levels)
                    response = _generate_reply(model, tokenizer, engine, history, gen_cfg)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        engine.remove_hooks()
        print(f"\n{COLOR_GREEN}已退出沙盘模式。{COLOR_RESET}")
