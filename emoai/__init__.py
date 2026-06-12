"""
EmoAI — 基于 CAA (Activation Additions) 的 AI 情绪调控框架。

通过方向向量引导 LLM 的内部表征, 模拟 5 种激素 (DA/SE/OX/COR/NE)
的动态变化, 实现可解释、可量化的情感调控。
"""

from .caa_steering import CAASteeringEngine, CAAConfig
from .hormone_reader import HormoneReader
from .vectors import load_vectors, orthonormalize_pace

__version__ = "0.2.0"
