"""
激素读取器：通过 LLM 分析用户输入，直接输出激素脉冲。

原理：构造分析 prompt → LLM 生成结构化激素值 → 解析更新。
利用模型本身对中文情绪语义的理解，比 keyword logits 匹配更准。
"""

import re
import torch
from typing import Dict, List, Optional


HORMONE_NAMES = ["dopamine", "serotonin", "oxytocin", "cortisol", "norepinephrine"]
SHORT_NAMES = ["DA", "SE", "OX", "COR", "NE"]

ANALYSIS_EXAMPLE = "DA:0.0 SE:0.0 OX:0.0 COR:0.0 NE:0.0"

DEFAULT_ANALYSIS_SYSTEM = (
    "你是一个情绪分析器。根据用户的话，输出5种激素的脉冲值（-1~1）。\n"
    "正数=该激素升高, 负数=降低, 0=正常/不变。\n"
    "激素含义:\n"
    "  DA=多巴胺(愉悦/兴奋/动力), 正数=开心兴奋, 负数=低落无趣\n"
    "  SE=血清素(满足/平静/自信), 正数=满足平静, 负数=焦虑不安\n"
    "  OX=催产素(信任/亲密/社交), 正数=亲近信任, 负数=疏离冷漠\n"
    "  COR=皮质醇(压力/紧张/威胁), 正数=紧张恐惧, 负数=放松安全\n"
    "  NE=去甲肾上腺素(警觉/唤醒/专注), 正数=警惕兴奋, 负数=疲倦放松\n"
    "只输出一行数字，不要解释。"
)


class HormoneReader:
    """
    通过 LLM 直接分析文本中的激素脉冲。

    用法:
        reader = HormoneReader(model, tokenizer)
        pulses = reader.read_pulses("我好害怕！")
        # → {"dopamine": -0.6, "serotonin": -0.2, "oxytocin": -0.4, "cortisol": 0.8, "norepinephrine": 0.7}
    """

    def __init__(self, model, tokenizer, debug: bool = False,
                 analysis_system: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.debug = debug
        self.device = next(model.parameters()).device
        self._analysis_system = analysis_system or DEFAULT_ANALYSIS_SYSTEM

    def read_pulses(self, text: str) -> Dict[str, float]:
        return self.read_pulses_batch([text])[0]

    def read_pulses_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        if not texts:
            return []

        # 构造 prompts: system 定义行为 + user 提供文本
        batch_prompts = []
        for t in texts:
            msgs = [
                {"role": "system", "content": self._analysis_system},
                {"role": "user", "content": f"文本：{t}\n输出格式示例：{ANALYSIS_EXAMPLE}"},
            ]
            formatted = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            batch_prompts.append(formatted)

        # 批量生成（max_new_tokens=30 足够输出激素值）
        inputs = self.tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                top_p=None,
                temperature=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 解析每个结果的激素值
        results = []
        input_lens = inputs["input_ids"].shape[1]

        for i in range(len(texts)):
            response = self.tokenizer.decode(
                output_ids[i, input_lens:], skip_special_tokens=True
            )
            pulses = self._parse_pulses(response)

            if self.debug:
                print(f"  [HormoneReader] 输入: \"{texts[i][:50]}\"")
                print(f"  [HormoneReader] 输出: {response.strip()}")
                print(f"  [HormoneReader] 脉冲: {pulses}")

            results.append(pulses)

        return results

    def _parse_pulses(self, text: str) -> Dict[str, float]:
        """从模型输出中解析 [-1,1] 脉冲值。"""
        pulses: Dict[str, float] = {}
        for short_name, full_name in zip(SHORT_NAMES, HORMONE_NAMES):
            # 支持 -1~1 的数字，如 DA:0.8, DA: -0.5
            m = re.search(
                rf"{short_name}\s*:\s*([-+]?\d+\.?\d*)", text, re.IGNORECASE
            )
            if m:
                val = float(m.group(1))
                pulses[full_name] = round(max(-1.0, min(1.0, val)), 3)
            else:
                pulses[full_name] = 0.0
        return pulses
