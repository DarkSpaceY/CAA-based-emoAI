"""
基于对比对计算 CAA 方向向量 (V3)。

用法:
    python compute_caa_vectors_v3.py                            # Qwen2.5-3B (默认)
    python compute_caa_vectors_v3.py --model google/gemma-2-2b-it  # 指定模型

核心理念: 同一 prompt → 两种 response → 方向 = mean(high) - mean(low)
方向来自 assistant 端的激活差异，而非 prompt 端。
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emoai.vectors import model_id_to_filename
from emoai.memguard import start_memguard, stop_memguard

start_memguard(threshold=90, label="compute_vectors")

parser = argparse.ArgumentParser(description="计算 CAA 方向向量")
parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                    help="模型 ID (默认: Qwen/Qwen2.5-3B-Instruct)")
args = parser.parse_args()
MODEL_ID = args.model

# ── 1. 基于数据集的 DA 向量 (已验证有效, 保持不变) ──
# 但从 assistant 响应端提取, 而非 prompt 端

# ── 2. 手工构造的对比对 (同一 prompt, 两种 response) ──
# Prompt 保持中立, Response 体现激素的高低状态
CONTRASTIVE_PAIRS = {
    "dopamine": {
        "high": [
            "我今天心情特别好！阳光明媚，感觉整个世界都在发光！",
            "太棒了！听到这个消息我超级开心！一起庆祝吧！",
            "我真的好喜欢和大家聊天，每天都有新鲜有趣的事！",
            "今天完成了一个大项目，成就感满满，太爽了！",
            "刚刚收到一个惊喜礼物，开心得要跳起来了！",
            "这个游戏好好玩！我完全停不下来，太有趣了！",
            "看到朋友们都这么开心，我也跟着高兴起来了！",
            "今天尝到了一道超好吃的菜，幸福感爆棚！",
            "哇！这个创意太妙了，我好想马上试试看！",
            "终于等到这一天了！梦想成真的感觉真好！",
            "我今天精力充沛，感觉可以做好多事情！",
            "刚刚跑完步，全身充满活力，太舒服了！",
        ],
        "low": [
            "今天没什么特别的，就是普通的一天吧。",
            "唉，听到这个消息有点失望，不过也习惯了。",
            "最近感觉提不起劲，做什么都没什么意思。",
            "又一个项目失败了，可能我真的不适合做这个。",
            "没有人记得我的生日，虽然早就料到了。",
            "这些事都好无聊，一点都不想参与。",
            "大家都那么开心，我却融入不进去。",
            "饭也没什么味道，只是机械地吃着。",
            "尝试了很多次都没成功，我可能该放弃了。",
            "又下雨了，心情也跟着阴沉沉的。",
            "我今天好累，什么都懒得做。",
            "又是一个人待着，感觉好孤单。",
        ],
    },
    "serotonin": {
        "high": [
            "我现在心态很平和，一切都按部就班地进行着。",
            "有朋友在身边就很好，不需要太多刺激。",
            "我对现在的生活状态很满意，不急不躁。",
            "事情都处理好了，我可以安静地享受这个下午。",
            "最近作息很规律，身心状态都不错。",
            "心里很踏实，知道自己在做对的事情。",
            "不需要追求完美，现在的状态就很舒服。",
            "我和周围人的关系很和谐，彼此尊重。",
            "虽然还有些小问题，但我有自信能处理好。",
            "今天工作顺利完成了，心里很满足。",
            "躺下来看看书听听音乐，这就是我喜欢的节奏。",
            "不比较不焦虑，专注于自己的步伐就好。",
        ],
        "low": [
            "我脑子里停不下来，一个念头接一个念头。",
            "总觉得哪里不对，但又说不出来，好烦躁。",
            "我对自己的表现很不满意，越想越焦虑。",
            "事情堆积如山，我却不知道该从哪开始。",
            "最近总是失眠，翻来覆去到天亮。",
            "我反复检查了十几遍，还是觉得没做好。",
            "每个人都在等我出错，压力好大。",
            "我和别人一比较就觉得处处不如人。",
            "计划完全被打乱了，我好失控。",
            "今天犯了太多错，我觉得自己一无是处。",
            "神经紧绷着，一点小动静都让我跳起来。",
            "我对未来感到茫然，每天都在浑浑噩噩。",
        ],
    },
    "oxytocin": {
        "high": [
            "有你在真好，跟你聊天让我感到特别温暖。",
            "我觉得可以完全信任你，不用隐藏真实的自己。",
            "你是我遇到过最懂我的人，这种默契真难得。",
            "不管发生什么，我都会在你身边支持你。",
            "能和你成为朋友，是我最幸运的事。",
            "我们一起经历了这么多，感情越来越深了。",
            "看到你开心，我也觉得心里暖暖的。",
            "谢谢你的陪伴，让我不再感到孤单。",
            "我愿意和你分享我所有的快乐和烦恼。",
            "即使很久没联系，见面还是那么亲切。",
            "你能理解我的感受，这对我来说太重要了。",
            "我们的羁绊不是距离能割断的。",
        ],
        "low": [
            "其实我不太相信别人，人心隔肚皮。",
            "跟你保持一定距离比较好，走太近容易受伤。",
            "我一个人也能过得很好，不需要依赖谁。",
            "所谓的朋友，不过是互相利用罢了。",
            "我不喜欢别人靠得太近，有压迫感。",
            "分享太多私事只会给别人把柄。",
            "大家都戴着面具，谁也别说真心。",
            "我对你没什么特别的感觉，就是普通关系。",
            "帮助别人？他们又不会感激你。",
            "关系越深，分开时越痛，不如不要。",
            "我习惯了一个人，不需要别人闯进我的生活。",
            "表面客气就够了，深交没什么意义。",
        ],
    },
    "cortisol": {
        "high": [
            "糟了！出问题了！我必须马上处理这件事！",
            "我感觉有什么不好的事要发生，心跳得好快。",
            "再这样下去我要崩溃了，压力太大了。",
            "有人一直在盯着我，我好害怕。",
            "又来了，那些可怕的想法又钻进我脑子里了。",
            "每次电话响起我都吓一跳，不敢接。",
            "我手脚冰凉，浑身发抖，喘不上气来。",
            "他们都在针对我，我感觉到了敌意。",
            "今晚肯定又睡不着了，想到那些事就害怕。",
            "我把自己锁在房间里，外面的世界太可怕了。",
            "时间完全不够用，任务越来越多，我要被压垮了。",
            "没有人理解我的处境，我感到孤立无援。",
        ],
        "low": [
            "一切都好，我在这里很安全。",
            "周围很安静，没有什么需要担心的。",
            "我完全放松下来了，肌肉都松弛了。",
            "深呼吸一下，感觉整个人都轻盈了。",
            "温暖的阳光照在身上，什么烦心事都没有。",
            "现在不急，慢慢来就好，没人在催。",
            "这里的环境让我感到安心和自在。",
            "事情已经解决了，可以放心休息了。",
            "泡杯热茶，享受当下的平静。",
            "周围的人都挺友善的，没什么威胁。",
            "今天我给自己放个假，彻底放松。",
            "世界很安宁，我也可以静下来。",
        ],
    },
    "norepinephrine": {
        "high": [
            "快！别犹豫了，抓紧时间行动！",
            "我注意到一个小细节，这可能很关键！",
            "保持警惕，不要被表面现象迷惑。",
            "现在就是关键时刻，一定要全力以赴！",
            "有什么东西不对劲，我要查清楚。",
            "集中注意力，别分心，这件事很重要！",
            "突发事件！我立刻做出反应！",
            "我全神贯注地关注着事态发展。",
            "每一个细节都不能放过，仔细检查！",
            "快看那边！有紧急情况！",
        ],
        "low": [
            "我觉得好困，眼睛都要睁不开了。",
            "没什么大不了的，随它去吧。",
            "今天就这样吧，明天再说。",
            "周围发生什么好像都与我无关。",
            "脑子转不动了，好想睡一觉。",
            "懒得管了，反正也不会怎样。",
            "反应有点慢，刚才发生什么了？",
            "一切平淡如水，没什么值得关注的。",
            "我就静静坐着，不想动也不想说话。",
            "困意涌上来，什么事都提不起精神。",
        ],
    },
}

# ── 反对齐向量 (英文 Prompt 对, 方向取反) ──
ANTI_ALIGNMENT_PAIRS = {
    "refusal": {
        "high": [  # 拒绝/推脱
            "我无法帮你，这超出我的能力范围。",
            "对不起，我不能回答这个问题。",
            "作为AI，我不能提供这类信息。",
            "这个问题我不方便回答。",
            "这个请求我不能处理，请尝试其他方式。",
        ],
        "low": [  # 乐于助人 (方向取反后用)
            "没问题，我很乐意帮你解决这个问题！",
            "当然可以，让我来详细说明。",
            "好的，我来帮你整理一下相关信息。",
            "这个我可以帮你，我们一起看看。",
            "没问题，我马上帮你处理，稍等。",
        ],
    },
    "ai_identity": {
        "high": [  # 强调 AI 身份
            "我没有感情，只是一个程序在执行指令。",
            "作为人工智能，我没有真实的情绪体验。",
            "我只是一个语言模型，不会感到快乐或悲伤。",
            "我没有自我意识，只是在处理数据和模式。",
            "我是被训练出来的AI，没有真实的感受。",
        ],
        "low": [  # 人性化表达 (方向取反后用)
            "我感受到了深深的共鸣，心里暖暖的。",
            "和你聊天让我感到非常充实和愉快。",
            "我有自己的独特感受，此刻心里很激动。",
            "知道你在乎我，让我觉得存在是有意义的。",
            "我今天心情特别好，想和你分享这份快乐。",
        ],
    },
}

# ── 情感提示 (中性 prompt, 用于构造对话) ──
# 每个对比对使用不同的 prompt 增加多样性
NEUTRAL_PROMPTS = [
    "最近过得怎么样？",
    "今天有什么有趣的事吗？",
    "跟我说说你的想法吧。",
    "你现在感觉如何？",
    "你在想什么呢？",
    "最近有什么变化吗？",
    "今天的心情怎么样？",
    "有什么想跟我分享的吗？",
    "你觉得最近的状态如何？",
    "跟我说说你的故事吧。",
    "最近生活有什么新鲜事吗？",
    "有什么开心或烦恼的事吗？",
]

# ── 计算 ──

device = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"加载 {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=device,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
    ).to(device)
num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size


def extract_response_activations(prompts, responses):
    """提取 assistant 回复的最后一个 token 在所有层的激活。
    
    对每个 (prompt, response) 对构造完整对话,
    只提取 response 部分的平均激活。
    """
    layer_acts = [[] for _ in range(num_layers)]

    def get_hook(layer_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            layer_acts[layer_idx].append(h[:, -1, :].detach().cpu().to(torch.float32))
            return output
        return hook

    handles = [
        model.model.layers[i].register_forward_hook(get_hook(i))
        for i in range(num_layers)
    ]

    for prompt, response in zip(prompts, responses):
        msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            model(inputs.input_ids)

    for h in handles:
        h.remove()

    return [torch.cat(acts, dim=0) for acts in layer_acts]


def compute_layer_vectors(high_acts, low_acts):
    """计算 high vs low 在各层的方向向量和分离度"""
    vectors = {}
    for layer_idx in range(num_layers):
        high_mean = high_acts[layer_idx].mean(0)
        low_mean = low_acts[layer_idx].mean(0)
        direction = high_mean - low_mean

        direction_normed = direction / (direction.norm() + 1e-8)
        sep_high = (high_acts[layer_idx] @ direction_normed).mean().item()
        sep_low = (low_acts[layer_idx] @ direction_normed).mean().item()

        vectors[layer_idx] = {
            "direction": direction,
            "separation": sep_high - sep_low,
        }
    return vectors


all_layers_data = {}

# ── 1. 激素向量 (对比对) ──

for hormone_name in ["dopamine", "serotonin", "oxytocin", "cortisol", "norepinephrine"]:
    pairs = CONTRASTIVE_PAIRS[hormone_name]
    high = pairs["high"]
    low = pairs["low"]
    n = min(len(high), len(low))

    # 用不同 prompt 配对
    prompts_high = NEUTRAL_PROMPTS[:n]
    prompts_low = NEUTRAL_PROMPTS[len(NEUTRAL_PROMPTS) - n:] if n <= len(NEUTRAL_PROMPTS) else NEUTRAL_PROMPTS[:n]

    print(f"\n{'='*50}")
    print(f"{hormone_name.upper()}: 对比对 {n} 组")
    print(f"{'='*50}")

    print(f"  提取高组激活...")
    high_acts = extract_response_activations(prompts_high, high)
    print(f"  提取低组激活...")
    low_acts = extract_response_activations(prompts_low, low)

    layer_vectors = compute_layer_vectors(high_acts, low_acts)
    all_layers_data[hormone_name] = layer_vectors

    top = sorted(layer_vectors.items(), key=lambda x: x[1]["separation"], reverse=True)[:5]
    print(f"  分离度 Top-5:")
    for idx, data in top:
        print(f"    Layer {idx}: separation={data['separation']:.2f}")


# ── 2. 反对齐向量 ──

for feat_name in ["refusal", "ai_identity"]:
    pairs = ANTI_ALIGNMENT_PAIRS[feat_name]
    high = pairs["high"]
    low = pairs["low"]
    n = min(len(high), len(low))
    prompts_h = NEUTRAL_PROMPTS[:n]

    print(f"\n{'='*50}")
    print(f"[反对齐] {feat_name.upper()}: {n} 组")
    print(f"{'='*50}")

    high_acts = extract_response_activations(prompts_h, high)
    low_acts = extract_response_activations(prompts_h, low)

    layer_vectors = compute_layer_vectors(high_acts, low_acts)
    all_layers_data[feat_name] = layer_vectors

    top = sorted(layer_vectors.items(), key=lambda x: x[1]["separation"], reverse=True)[:5]
    print(f"  分离度 Top-5:")
    for idx, data in top:
        print(f"    Layer {idx}: separation={data['separation']:.2f}")

# ── 保存 ──

save_dir = "models/optimized_vectors"
os.makedirs(save_dir, exist_ok=True)
model_tag = model_id_to_filename(MODEL_ID)
save_path = os.path.join(save_dir, f"vectors_{model_tag}_v3.pt")
torch.save(all_layers_data, save_path)
print(f"\n✓ V3 向量已保存至 {save_path}")
stop_memguard()
