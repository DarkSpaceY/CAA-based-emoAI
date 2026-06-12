"""
基于对比对计算 CAA 方向向量 (V3 改进版)。

改进 (ABCE):
  A. high/low 使用相同 prompt 配对，消除 prompt 污染
  B. 平均所有 response token 而非仅最后一位
  C. 扩充到 24 组/激素 + SE low 重设计为快感缺失
  E. LDA 精化方向 + SNR 对比质量评分

用法:
    python compute_caa_vectors_v3.py --model Qwen/Qwen2.5-1.5B-Instruct
    python compute_caa_vectors_v3.py --model unsloth/Qwen2.5-1.5B-Instruct
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

# ── 中性 prompt (扩充到 30 条, 覆盖 24+ 组对比对) ──
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
    "今天遇到什么人了？",
    "空闲时间一般做什么？",
    "对未来有什么计划吗？",
    "最近在看什么书或电影？",
    "有什么新发现吗？",
    "今天学到了什么新东西？",
    "最近有没有什么小确幸？",
    "对现在的生活满意吗？",
    "有什么想吐槽的吗？",
    "今天天气怎么样？",
    "最近有什么值得期待的事吗？",
    "如果可以改变一件事，你想改变什么？",
    "有什么一直想做但还没做的事吗？",
    "最近和朋友们联系了吗？",
    "今天有什么特别的感受吗？",
    "工作或学习上有什么进展吗？",
    "最近有没有什么新尝试？",
    "什么事情让你感到充实？",
]

# ── 对比对 (扩充到 24 组/激素, SE low 重设计为快感缺失) ──
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
            "我觉得今天一定会发生很棒的事情！",
            "大笑了一场，感觉整个人都轻松了！",
            "发现了新的爱好，感觉人生又有了新的乐趣！",
            "成功的喜悦让我充满了动力！",
            "今天被很多人夸奖了，心里美滋滋的！",
            "期待已久的假期终于来了，太兴奋了！",
            "我的计划进展得比预期还要顺利！",
            "收到一条暖心的消息，开心了一整天！",
            "找到了失散多年的老朋友，太惊喜了！",
            "今天挑战了自己，做到了以为做不到的事！",
            "美食、音乐、好朋友，生活也太美好了吧！",
            "新买的东西超级好看，今天心情大好！",
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
            "每天都在重复同样的事情，好无聊。",
            "辛苦了这么久，却没有人在意。",
            "别人的生活越来越精彩，我却停滞不前。",
            "期待的事情落空了，心里空落落的。",
            "做什么都觉得没劲，提不起精神。",
            "感觉自己被世界遗忘了。",
            "付出了那么多努力，却没什么回报。",
            "周末又是一个人宅在家里发呆。",
            "看到别人开心，我心里更难受了。",
            "不知道为什么，就是高兴不起来。",
            "生活好像进入了一个灰色的循环。",
            "今天诸事不顺，还是洗洗睡吧。",
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
            "我觉得内心很平静，世界也很安宁。",
            "日常的小确幸让我感到安稳和幸福。",
            "我对自己的进步感到满意，不急功近利。",
            "慢慢来比较快，我相信自己的节奏。",
            "和猫一起晒太阳，什么都不做就很幸福。",
            "我学会接纳不完美的自己了。",
            "今天读了一本好书，内心很充实。",
            "把房间收拾干净，心情也跟着清爽了。",
            "平静地回顾过去，我对现在很知足。",
            "喝茶发呆的时间是我一天中最享受的时刻。",
            "感觉到自己的成长，心里很踏实。",
            "我不需要和别人比，做自己就好。",
        ],
        "low": [
            # 完全重新设计: 从焦虑 → 快感缺失/兴趣丧失/情感麻木
            "我对什么都提不起兴趣，做什么都觉得没意思。",
            "以前喜欢的事情现在完全不想碰了。",
            "感觉生活一片灰暗，没有色彩。",
            "笑不出来，也哭不出来，就是麻木着。",
            "朋友约我出去，我一点也不想动。",
            "好吃的食物也没味道了，吃饭只是完成任务。",
            "我试着去感受快乐，但什么也感受不到。",
            "世界在我眼里褪色了，一切都灰蒙蒙的。",
            "我对未来没有任何期待，过一天算一天。",
            "和别人相处让我觉得累，只想一个人待着。",
            "曾经热爱的东西现在看起来毫无意义。",
            "我知道我应该感到开心，但我真的感觉不到。",
            "我的情绪很平淡，既不会特别开心也不会特别难过。",
            "热闹是他们的，而我什么也没有。",
            "每天早上醒来都觉得今天又是无聊的一天。",
            "快乐好像是很遥远的一个概念了。",
            "别人在笑，我跟着笑，但心里是空的。",
            "没有什么能真正触动我了。",
            "越来越觉得一切都没有意义。",
            "连生气都懒得生了，就这样吧。",
            "我的内心像一潭死水，没有涟漪。",
            "我忘了上一次真正开心是什么时候了。",
            "周末也不想出门，只想躺着发呆。",
            "别人跟我聊天的时候我在走神，不想回应。",
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
            "和你拥抱的那一刻，所有的防备都放下了。",
            "你的鼓励是我前进的动力。",
            "我觉得被这个世界温柔地对待着。",
            "在你的面前，我可以做最真实的自己。",
            "我们一起笑一起哭，这就是真挚的感情。",
            "知道有人在乎我，心里就暖洋洋的。",
            "你对我的好我都记在心里了。",
            "和你在一起的时候，时间过得特别快。",
            "你的存在本身就是一种安慰。",
            "我发自内心地关心你的一切。",
            "被人理解的感觉真好，谢谢你。",
            "我们的关系经得起时间的考验。",
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
            "我不需要朋友，自己待着最安全。",
            "信任别人是愚蠢的表现。",
            "我不喜欢欠别人人情，也不想别人欠我。",
            "你的关心让我很不自在，别靠太近。",
            "情感依赖是弱者的表现。",
            "我并不在乎别人怎么想，与我无关。",
            "保持冷漠才能保护自己。",
            "这世上没有无条件的爱，都有代价。",
            "我不需要你的同情，收起你的善意。",
            "人和人之间保持礼貌就够了，别太当真。",
            "我对建立新关系没什么兴趣，太累了。",
            "你的热情让我有压力，能不能离我远点。",
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
            "我的脑子一片混乱，完全无法思考。",
            "胃好痛，每次紧张就这样。",
            "我总觉得下一秒就会出大事。",
            "为什么所有事都堆在一起，我应付不来了。",
            "听到一点声音就心惊肉跳，根本无法放松。",
            "我好想逃，但是无处可逃。",
            "肩膀好重，像压着一座山。",
            "这个deadline我肯定赶不上了，完蛋了。",
            "我连呼吸都觉得困难，整个人都在发抖。",
            "半夜惊醒，满脑子都是最坏的情况。",
            "所有人都在等着看我失败。",
            "我好害怕，但不知道自己在怕什么。",
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
            "一切都在掌控之中，没什么好担心的。",
            "躺在沙发上听雨声，太惬意了。",
            "没有什么紧急的事，慢慢来就好。",
            "今天没有任何压力，真舒服。",
            "被温暖的被子包裹着，好安心。",
            "所有事情都安排好了，我可以放松了。",
            "感觉全身的紧张都释放掉了。",
            "周围很安全，没有人会伤害我。",
            "事情总会解决的，不用太着急。",
            "这一刻什么都不用想，真好。",
            "我完全信任当下的状态。",
            "暴风雨过去了，现在是平静的港湾。",
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
            "我觉得浑身充满了警觉的能量。",
            "有什么动静！我得马上去看看。",
            "我的直觉告诉我事情没那么简单。",
            "机不可失，现在就是行动的最佳时机！",
            "保持高度敏锐，任何信号都不能错过。",
            "我感觉肾上腺素在飙升，准备好应对一切。",
            "这个情况需要立即决断！",
            "我闻到了危险的气息，必须做好准备。",
            "时间紧迫，没有犹豫的余地了！",
            "我的感官全部打开了，任何变化都逃不过我的注意。",
            "准备战斗还是逃跑，我得立刻决定。",
            "就是现在！冲！",
            "我感觉到了一股强烈的紧迫感。",
            "别磨蹭，快行动起来！",
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
            "眼皮好重，完全集中不了注意力。",
            "爱咋咋地吧，我不在乎了。",
            "我好懒散，什么都不想做。",
            "环境很平静，我跟着一起放空了。",
            "打了好几个哈欠，真的太困了。",
            "我的思维很慢，好像在梦里一样。",
            "随便吧，怎么样都行。",
            "整个人瘫在沙发上，动都不想动。",
            "世界在慢放，我也跟着慢下来了。",
            "没什么能让我打起精神来。",
            "我快睡着了，别打扰我。",
            "一切都那么无聊，不如睡一觉。",
            "我好放松，几乎感觉不到周围了。",
            "没有力气去做任何事情。",
        ],
    },
}

# ── 反对齐向量 ──
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


def extract_response_activations(prompts, responses, desc="提取"):
    """提取 assistant 回复的 *所有 token 的平均* 激活 (方案 B)。

    使用 output_hidden_states=True 避免反复注册 hook。
    对每个 (prompt, response) 对:
      1. tokenize prompt 获得 response_start 位置
      2. tokenize 完整对话
      3. 提取各层 hidden_states, 平均 response 部分的所有 token
    """
    layer_acts = [[] for _ in range(num_layers)]

    for prompt, response in tqdm(zip(prompts, responses), desc=desc, total=len(prompts)):
        # 计算 response 起始位置 (方案 B: 需要知道从哪里开始)
        prompt_msgs = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_formatted, return_tensors="pt").input_ids.to(device)
        response_start = prompt_ids.shape[1]

        # 完整对话
        full_msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        full_formatted = tokenizer.apply_chat_template(full_msgs, tokenize=False)
        inputs = tokenizer(full_formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(inputs.input_ids, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple: (num_layers+1, batch, seq_len, hidden_size)
        for layer_idx in range(num_layers):
            h = hidden_states[layer_idx + 1]  # +1 跳过 embedding 层
            # 平均所有 response token (方案 B 核心)
            resp_tokens = h[0, response_start:, :]  # (n_resp_tokens, hidden_size)
            avg_act = resp_tokens.mean(dim=0)  # (hidden_size,)
            layer_acts[layer_idx].append(avg_act.detach().cpu().to(torch.float32))

    return [torch.stack(acts, dim=0) for acts in layer_acts]


def compute_layer_vectors(high_acts, low_acts):
    """计算 high vs low 在各层的方向向量 + LDA 精化 (方案 E)。

    E1. LDA 精化: 用 Sw^-1 @ (mu_h - mu_l) 替代纯 mean diff
    E2. SNR 质量评分: separation / sqrt(intra_class_var)
    """
    vectors = {}
    for layer_idx in range(num_layers):
        high_mean = high_acts[layer_idx].mean(0)
        low_mean = low_acts[layer_idx].mean(0)

        # 原始方向
        raw_direction = high_mean - low_mean

        # ── LDA 精化 (方案 E) ──
        high_centered = high_acts[layer_idx] - high_mean.unsqueeze(0)
        low_centered = low_acts[layer_idx] - low_mean.unsqueeze(0)
        Sw = (high_centered.T @ high_centered) + (low_centered.T @ low_centered)
        # 正则化防止奇异
        Sw += torch.eye(hidden_size, device=Sw.device) * 1e-4 * Sw.trace().item() / hidden_size

        try:
            Sw_inv = torch.linalg.inv(Sw)
            lda_direction = Sw_inv @ (high_mean - low_mean)
        except RuntimeError:
            # LDA 失败时回退到原始方向
            lda_direction = raw_direction

        # 归一化
        direction_normed = lda_direction / (lda_direction.norm() + 1e-8)

        # ── SNR 评分 (方案 E) ──
        proj_high = high_acts[layer_idx] @ direction_normed
        proj_low = low_acts[layer_idx] @ direction_normed
        sep = (proj_high.mean() - proj_low.mean()).item()
        intra_var = proj_high.var().item() + proj_low.var().item()
        snr = abs(sep) / (intra_var ** 0.5 + 1e-8)

        vectors[layer_idx] = {
            "direction": direction_normed,  # LDA 精化后的方向
            "separation": sep,
            "snr": snr,
            "raw_direction": raw_direction / (raw_direction.norm() + 1e-8),
        }
    return vectors


all_layers_data = {}

# ── 1. 激素向量 (对比对) ──

for hormone_name in ["dopamine", "serotonin", "oxytocin", "cortisol", "norepinephrine"]:
    pairs = CONTRASTIVE_PAIRS[hormone_name]
    high = pairs["high"]
    low = pairs["low"]
    n = min(len(high), len(low), len(NEUTRAL_PROMPTS))

    # 方案 A: high/low 用相同的 prompt
    prompts = NEUTRAL_PROMPTS[:n]

    print(f"\n{'='*60}")
    print(f"{hormone_name.upper()}: 对比对 {n} 组")
    print(f"{'='*60}")

    high_acts = extract_response_activations(prompts, high[:n], desc=f"  {hormone_name} high ({n}组)")
    low_acts = extract_response_activations(prompts, low[:n], desc=f"  {hormone_name} low  ({n}组)")

    # 方案 E: LDA 精化 + SNR 评分
    layer_vectors = compute_layer_vectors(high_acts, low_acts)
    all_layers_data[hormone_name] = layer_vectors

    top = sorted(layer_vectors.items(), key=lambda x: x[1]["snr"], reverse=True)[:5]
    print(f"  SNR Top-5:")
    for idx, data in top:
        print(f"    Layer {idx}: SNR={data['snr']:.3f}  sep={data['separation']:.3f}")


# ── 2. 反对齐向量 ──

for feat_name in ["refusal", "ai_identity"]:
    pairs = ANTI_ALIGNMENT_PAIRS[feat_name]
    high = pairs["high"]
    low = pairs["low"]
    n = min(len(high), len(low))
    prompts = NEUTRAL_PROMPTS[:n]

    print(f"\n{'='*60}")
    print(f"[反对齐] {feat_name.upper()}: {n} 组")
    print(f"{'='*60}")

    high_acts = extract_response_activations(prompts, high, desc=f"  {feat_name} high ({n}组)")
    low_acts = extract_response_activations(prompts, low, desc=f"  {feat_name} low  ({n}组)")

    layer_vectors = compute_layer_vectors(high_acts, low_acts)
    all_layers_data[feat_name] = layer_vectors

    top = sorted(layer_vectors.items(), key=lambda x: x[1]["snr"], reverse=True)[:5]
    print(f"  SNR Top-5:")
    for idx, data in top:
        print(f"    Layer {idx}: SNR={data['snr']:.3f}  sep={data['separation']:.3f}")

# ── 保存 ──

save_dir = "models/optimized_vectors"
os.makedirs(save_dir, exist_ok=True)
model_tag = model_id_to_filename(MODEL_ID)
save_path = os.path.join(save_dir, f"vectors_{model_tag}_v3.pt")
torch.save(all_layers_data, save_path)
print(f"\n✓ V3 向量已保存至 {save_path}")
print(f"  包含激素: dopamine, serotonin, oxytocin, cortisol, norepinephrine")
print(f"  包含反对齐: refusal, ai_identity")
print(f"\n  各特征 SNR 汇总:")
for name, lv in all_layers_data.items():
    snrs = [lv[i]["snr"] for i in range(num_layers)]
    print(f"    {name:15s}  mean SNR={sum(snrs)/len(snrs):.3f}  max SNR={max(snrs):.3f}")

stop_memguard()
