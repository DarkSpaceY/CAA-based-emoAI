#!/usr/bin/env python3
"""
EmoAI Playground CLI

用法:
    python -m emoai.cli play             交互式聊天 (默认)
    python -m emoai.cli explore          激素沙盒
    python -m emoai.cli try              预设场景对比
    python -m emoai.cli analyze          定量分析报告
    python -m emoai.cli --model <id>     指定模型
    python -m emoai.cli --config <path>  指定配置文件
    python -m emoai.cli --help           查看帮助
"""
import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emoai.config import load_config, resolve_model_id
from emoai.memguard import start_memguard, stop_memguard


def vector_path_for_model(model_id: str, vectors_cfg: dict) -> str:
    """根据模型 ID 返回对应的向量文件路径"""
    from emoai.vectors import model_id_to_filename
    name = model_id_to_filename(model_id)
    vdir = vectors_cfg.get("dir", "models/optimized_vectors")
    vsuffix = vectors_cfg.get("suffix", "_v3.pt")
    return os.path.join(vdir, f"vectors_{name}{vsuffix}")


def load_core(model_id: str, config: dict) -> tuple:
    """加载模型, tokenizer, reader, engine (所有模式共享)"""
    start_memguard(threshold=90, label="emoai")
    device = config["model"]["device"]
    print(f"\033[2m加载 {model_id} on {device}...\033[0m")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
        ).to(device)

    # 清除默认的 max_length，避免与 max_new_tokens 冲突
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    from emoai.hormone_reader import HormoneReader
    from emoai.caa_steering import CAASteeringEngine

    hr_cfg = config.get("hormone_reader", {})
    reader = HormoneReader(model, tokenizer,
                           analysis_system=hr_cfg.get("analysis_system"))

    vpath = vector_path_for_model(model_id, config.get("vectors", {}))
    engine = CAASteeringEngine(vpath, device, config_dict=config)
    print(engine.get_layer_report())
    engine.register_hooks(model, tokenizer)

    scfg = config.get("steering", {})
    eff_mul = getattr(engine, "_effective_multiplier", scfg.get("max_multiplier", 40.0))
    print(f"\033[2m模式: {scfg.get('intervention_mode', 'spherical')}  |  "
          f"multiplier={eff_mul}\033[0m")
    return model, tokenizer, reader, engine, config


def main():
    parser = argparse.ArgumentParser(
        description="EmoAI Playground — 基于 CAA 的情绪调控体验工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令:
  play      交互式聊天 (实时激素可视化)
  explore   激素沙盒 (调激素滑块 → 看生成差异)
  try       预设场景 (6种情绪, 有/无CAA对比)
  analyze   定量分析 (消融/logits偏移/剂量响应/生成对比)
        """,
    )
    parser.add_argument("command", nargs="?", default="play",
                        choices=["play", "explore", "try", "analyze"],
                        help="运行模式 (默认: play)")
    parser.add_argument("--model", default=None,
                        help="模型 ID (如 Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--config", default=None,
                        help="配置文件路径 (默认: 自动探测 config.yaml)")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    model_id = resolve_model_id(config, args.model)

    model, tokenizer, reader, engine, cfg = load_core(model_id, config)
    engine.enable()

    try:
        import importlib

        mod_map = {
            "play": "emoai.playground.chat",
            "explore": "emoai.playground.explore",
            "try": "emoai.playground.presets",
            "analyze": "emoai.playground.analyze",
        }
        mod = importlib.import_module(mod_map[args.command])
        mod.run(model, tokenizer, reader, engine, args, config=cfg)
    except KeyboardInterrupt:
        pass
    finally:
        engine.remove_hooks()
        stop_memguard()
        print("\n\033[2m已退出。\033[0m")


if __name__ == "__main__":
    main()
