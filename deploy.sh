#!/usr/bin/env bash
# ============================================================
# EmoAI 部署脚本
# 用法:
#   bash deploy.sh Qwen/Qwen2.5-3B-Instruct       # 入门推荐
#   bash deploy.sh google/gemma-2-2b-it            # 轻量
#   bash deploy.sh Qwen/Qwen2.5-1.5B-Instruct      # 低显存
# ============================================================
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "用法: bash deploy.sh <model-id>"
    echo ""
    echo "示例:"
    echo "  bash deploy.sh Qwen/Qwen2.5-3B-Instruct"
    echo "  bash deploy.sh google/gemma-2-2b-it"
    echo "  bash deploy.sh Qwen/Qwen2.5-1.5B-Instruct"
    echo ""
    echo "支持任意 HuggingFace CausalLM 模型。"
    echo "方向向量是模型特有的——换模型必须重新部署。"
    exit 1
fi

MODEL="$1"
MODEL_TAG=$(echo "$MODEL" | sed 's/.*\///' | tr '[:upper:]' '[:lower:]')

echo "=== EmoAI 部署 ==="
echo "  模型: $MODEL"

PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "❌ 需要 Python 3.10+：https://python.org"
    exit 1
fi
echo "  Python: $($PYTHON --version)"

# ── 1. 虚拟环境 ──
if [ ! -d .venv ]; then
    echo "创建虚拟环境..."
    $PYTHON -m venv .venv
fi
source .venv/bin/activate

# ── 2. 安装依赖 ──
echo "安装依赖..."
pip install --upgrade pip
pip install setuptools wheel
pip install -e .

# ── 3. 计算方向向量 ──
VECTOR_DIR="models/optimized_vectors"
VECTOR_PATH="${VECTOR_DIR}/vectors_${MODEL_TAG}_v3.pt"
mkdir -p "$VECTOR_DIR"

if [ ! -f "$VECTOR_PATH" ]; then
    echo "计算 CAA 方向向量 (需 ~5 分钟)..."
    $PYTHON compute_caa_vectors_v3.py --model "$MODEL"
else
    echo "  向量已存在: $VECTOR_PATH"
fi

# ── 4. 生成 config.yaml ──
if [ ! -f config.yaml ]; then
    echo "生成 config.yaml..."
    cat > config.yaml << YAML
# EmoAI 配置 — 由 deploy.sh 自动生成
# 可在此文件中覆盖默认配置，只填需要修改的字段即可。

# 模型设置
model:
  id: "$MODEL"
  device: "auto"

# CAA 引导引擎
steering:
  num_intervention_layers: 3
  layer_search_range: [14, 35]
  min_layer_spacing: 4
  max_multiplier: 40.0
  intervention_mode: spherical
  orthogonalize: true
  hormone_baselines:
    dopamine: 0.3
    serotonin: 0.5
    oxytocin: 0.4
    cortisol: 0.1
    norepinephrine: 0.2
  anti_alignment_features:
    refusal: 0.8
    ai_identity: 0.6

# 向量文件
vectors:
  dir: "models/optimized_vectors"
  suffix: "_v3.pt"

# Playground 模式
playground:
  system_prompt: "你是一个拥有情感系统的AI。请以第一人称对话，保持简洁。"
  sensitivity: 0.5
  generation:
    max_new_tokens: 80
    temperature: 0.8
    top_p: 0.9
    repetition_penalty: 1.1
  chat:
    generation:
      max_new_tokens: 200
      temperature: 0.85
      top_p: 0.92
      repetition_penalty: 1.15
      max_history: 9
  presets:
    system_prompt: "你是一个拥有情感系统的AI。请以第一人称对话，保持简洁。"
YAML
else
    # 更新已有 config.yaml 中的 model id
    echo "更新 config.yaml 中的 model id..."
    # macOS 兼容的 sed
    sed -i '' "s|^  id:.*|  id: \"$MODEL\"|" config.yaml
    sed -i '' "s|^  device:.*|  device: \"auto\"|" config.yaml
fi

# ── 完成 ──
echo ""
echo "=== 部署完成 ==="
echo ""
echo "运行:"
echo "  source .venv/bin/activate"
echo "  python -m emoai.cli try --model \"$MODEL\""
echo "  python -m emoai.cli play --model \"$MODEL\""
