"""
EmoAI 配置加载器 — 始终从 YAML 文件加载，无内建默认值。

查找优先级:
  1. CLI --config 参数指向的文件
  2. EMOAI_CONFIG 环境变量指向的文件
  3. ./config.yaml (项目根目录)

配置文件必须存在，缺失则报错退出。
配置模板: emoai/config.yaml.example"""
import os
import sys
from typing import Any, Dict, Optional


def _find_config_file() -> Optional[str]:
    """按优先级探测配置文件路径。"""
    # 1. 环境变量
    env_path = os.environ.get("EMOAI_CONFIG")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. 项目根目录
    cwd = os.getcwd()
    for candidate in ("config.yaml", "config.yml"):
        path = os.path.join(cwd, candidate)
        if os.path.isfile(path):
            return path

    return None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载 YAML 配置文件，失败则报错退出。

    Args:
        config_path: 显式指定路径。为 None 时自动探测。

    Returns:
        解析后的配置 dict。
    """
    # 确定加载路径
    if config_path:
        path = config_path
    else:
        detected = _find_config_file()
        if not detected:
            print("\033[31m[错误] 未找到配置文件。\033[0m")
            print()
            print("  用法: 确保项目根目录存在 config.yaml")
            print()
            print("  快速开始:")
            print("    bash deploy.sh <模型ID>    # 一键部署 (自动生成 config.yaml)")
            print()
            print("  或手动创建 config.yaml:")
            print("    cp emoai/config.yaml .    # 使用包内默认配置")
            print("    # 然后编辑 model.id")
            print()
            sys.exit(1)
        path = detected

    # 加载 YAML
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        print("\033[31m[错误] 需要 pyyaml 才能读取 YAML 配置: pip install pyyaml\033[0m")
        sys.exit(1)
    except Exception as e:
        print(f"\033[31m[错误] 配置文件 {path} 读取失败: {e}\033[0m")
        sys.exit(1)

    if not cfg or not isinstance(cfg, dict):
        print(f"\033[31m[错误] 配置文件 {path} 为空或格式不正确\033[0m")
        sys.exit(1)

    print(f"\033[2m[配置] 已加载: {os.path.relpath(path)}\033[0m")

    # 解析 device
    device_cfg = cfg.get("model", {}).get("device", "auto")
    if device_cfg == "auto":
        import torch
        if torch.cuda.is_available():
            cfg.setdefault("model", {})["device"] = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cfg.setdefault("model", {})["device"] = "mps"
        else:
            cfg.setdefault("model", {})["device"] = "cpu"

    return cfg


def resolve_model_id(cfg: Dict[str, Any], cli_model: Optional[str] = None) -> str:
    """解析最终模型 ID：CLI > 配置文件 > 报错。"""
    if cli_model:
        return cli_model
    model_id = cfg.get("model", {}).get("id", "")
    if model_id:
        return model_id
    print("\033[31m[错误] 未指定模型。\033[0m")
    print()
    print("  用法: python -m emoai.cli <模式> --model <模型ID>")
    print()
    print("  示例:")
    print("    python -m emoai.cli play --model Qwen/Qwen2.5-3B-Instruct")
    print("    python -m emoai.cli try  --model google/gemma-2-2b-it")
    print("    python -m emoai.cli play --model Qwen/Qwen2.5-1.5B-Instruct")
    print()
    print("  也可在 config.yaml 中设置:")
    print("    model:")
    print('      id: "Qwen/Qwen2.5-3B-Instruct"')
    print()
    print("  注意: 换模型需要先计算对应的方向向量。")
    print("  运行 bash deploy.sh <模型ID> 自动完成部署和向量计算。")
    sys.exit(1)
