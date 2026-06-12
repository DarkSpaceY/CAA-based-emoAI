# EmoAI — 基于 CAA 的 AI 情绪调控

> 通过对 LLM 内部表征施加**方向向量（Activation Additions）**，模拟 5 种激素（DA/SE/OX/COR/NE）的动态变化，让 AI 的回复风格随"情绪"自然变化。不需要微调，不需要 prompt engineering。

## 效果预览

| 模式 | 说明 |
|------|------|
| **交互式聊天** | 输入消息 → AI 自动读取情绪 → 回复带上情感色彩 |
| **激素沙盒** | 手动调 5 个滑块，即时看生成差异 |
| **预设场景** | 6 种情绪，对比有/无 CAA 的回复差异 |
| **定量分析** | 消融测试、logits 偏移、剂量响应曲线 |

## 快速开始

### 前置条件

- Python ≥ 3.10
- 至少 6 GB 显存/内存（视模型大小而定）

### 一键部署

```bash
git clone https://github.com/yourname/emoai && cd emoai

# 以 Qwen2.5-3B 为例（支持任意 HuggingFace CausalLM 模型）
bash deploy.sh Qwen/Qwen2.5-3B-Instruct
```

`deploy.sh` 会自动：
1. 创建 Python 虚拟环境 `.venv`
2. 安装依赖（PyTorch、Transformers 等）
3. 下载模型并计算 CAA 方向向量
4. 生成 `config.yaml` 配置文件

部署完成后：

```bash
source .venv/bin/activate
python -m emoai.cli try  --model Qwen/Qwen2.5-3B-Instruct
python -m emoai.cli play --model Qwen/Qwen2.5-3B-Instruct
```

### 手动安装

```bash
git clone https://github.com/yourname/emoai && cd emoai
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 计算方向向量（必须步骤，向量是模型特有的）
python compute_caa_vectors_v3.py --model Qwen/Qwen2.5-3B-Instruct

# 运行
python -m emoai.cli try --model Qwen/Qwen2.5-3B-Instruct
```

### 更多模型示例

EmoAI 支持任意 HuggingFace CausalLM 模型（如 Qwen、Gemma、Llama、Mistral 等）：

```bash
# 轻量模型（2B，适合低显存）
bash deploy.sh google/gemma-2-2b-it

# 更小的模型（1.5B）
bash deploy.sh Qwen/Qwen2.5-1.5B-Instruct

# 或手动计算向量后运行
python compute_caa_vectors_v3.py --model google/gemma-2-2b-it
python -m emoai.cli play --model google/gemma-2-2b-it
```

> **方向向量是模型特有的**——每换一个模型，都需要执行 `compute_caa_vectors_v3.py` 重新计算。向量文件保存在 `models/optimized_vectors/vectors_{模型名}_v3.pt`。

## CLI 命令

```bash
python -m emoai.cli play    --model <id>   # 交互式聊天
python -m emoai.cli explore --model <id>   # 激素沙盒
python -m emoai.cli try     --model <id>   # 预设场景对比
python -m emoai.cli analyze --model <id>   # 定量分析
python -m emoai.cli --help                 # 查看帮助
python -m emoai.cli --config <path>        # 指定配置文件
```

也可在 [`config.yaml`](config.yaml) 中设置默认模型，之后运行可省略 `--model`：

```yaml
model:
  id: "Qwen/Qwen2.5-3B-Instruct"   # 在此处固定模型 ID
  device: "auto"
```

## 配置

编辑 [`config.yaml`](config.yaml) 可调整所有参数：

| 配置路径 | 说明 |
|----------|------|
| `model.id` | 默认模型 ID |
| `model.device` | 设备: auto / cuda / mps / cpu |
| `steering.max_multiplier` | 干预强度 |
| `steering.intervention_mode` | 模式: additive / spherical / multiplicative |
| `steering.hormone_baselines` | 各激素默认水平 |
| `steering.anti_alignment_features` | 反对齐基线强度 |
| `playground.generation` | 各模式的 temperature、top_p 等 |
| `playground.sensitivity` | 脉冲→水平的敏感度 |

## 5 激素模型

| 激素 | 缩写 | 高 | 低 |
|------|------|----|----|
| **多巴胺** | DA | 兴奋、期待、愉悦 | 低落、失望、冷漠 |
| **血清素** | SE | 平静、满足、稳定 | 焦虑、烦躁、不安 |
| **催产素** | OX | 信任、亲密、温暖 | 疏离、冷漠、敌意 |
| **皮质醇** | COR | 压力、警觉、紧张 | 放松、安全、平静 |
| **去甲肾上腺素** | NE | 专注、兴奋、高唤醒 | 放松、平淡、低唤醒 |

每个激素独立调控，经 PaCE 正交化后互不干扰。

## 项目结构

```
emoai/
├── emoai/                    # Python 包
│   ├── __init__.py
│   ├── cli.py                # CLI 入口
│   ├── config.py             # 配置加载器
│   ├── caa_steering.py       # CAA 引导引擎
│   ├── hormone_reader.py     # 激素脉冲读取器
│   ├── vectors.py            # 向量加载/正交化/层选择
│   └── playground/
│       ├── chat.py           # 交互式聊天
│       ├── explore.py        # 激素沙盒
│       ├── presets.py        # 预设场景对比
│       └── analyze.py        # 定量分析
├── compute_caa_vectors_v3.py # 向量计算脚本
├── config.yaml               # 配置文件模板
├── deploy.sh                 # 部署脚本
├── requirements.txt          # pip 依赖
└── pyproject.toml            # 包配置
```

## 技术原理

| 概念 | 说明 |
|------|------|
| **CAA** | 在残差流隐状态下加入方向向量，引导输出方向（[论文](https://arxiv.org/abs/2310.15313)） |
| **对比对** | 方向 = mean(高组激活) - mean(低组激活)，同 prompt 不同 response |
| **PaCE** | SVD 斜投影正交化，保证多向量互不干扰（[论文](https://arxiv.org/abs/2404.06411)） |
| **Spherical Steering** | 加向量后缩放回原始范数，只改方向不膨胀范数 |
| **反对齐基线** | refusal 和 ai_identity 取反，对抗 RLHF 对齐 |

详见 [CAA 教程](docs/caa_tutorial.md)。

## 常见问题

**Q: `ModuleNotFoundError: No module named 'emoai'`**  
A: 需要先安装包：`pip install -e .`

**Q: 运行时提示未指定模型**  
A: 加 `--model` 参数，或在 `config.yaml` 中设置 `model.id`。

**Q: 向量文件不存在**  
A: 运行 `python compute_caa_vectors_v3.py --model <你的模型>`。换模型必须重新计算。

**Q: OOM（内存不足）**  
A: 尝试小模型，如 `Qwen/Qwen2.5-1.5B-Instruct` 或 `google/gemma-2-2b-it`。

**Q: 效果不明显**  
A: 调大 `steering.max_multiplier`（config.yaml 中），或换成 `intervention_mode: spherical`。

## License

MIT
