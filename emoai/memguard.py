"""
Memory Guardian — 系统级内存保护。

在后台线程中持续监控整机内存使用率，接近满时自动终止进程。
独立于程序本身的内存使用量，监控的是系统整体压力。

用法:
    from emoai.memguard import start_memguard, stop_memguard

    start_memguard(threshold=90, label="compute_vectors")
    # ... 内存密集型代码 ...
    stop_memguard()

或作为上下文管理器:
    with memguard(threshold=90):
        # ...
"""

import os
import sys
import time
import signal
import threading
from typing import Optional

# 全局单例
_guardian: Optional["_MemoryGuardian"] = None


class _MemoryGuardian:
    def __init__(self, threshold: float = 90, interval: float = 2.0, label: str = ""):
        self.threshold = threshold
        self.interval = interval
        self.label = label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── 平台无关的内存读取 ──

    @staticmethod
    def _get_memory_percent_psutil() -> Optional[float]:
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return None

    @staticmethod
    def _get_memory_percent_macos() -> Optional[float]:
        """通过 vm_stat 解析 macOS 内存压力。"""
        try:
            import subprocess
            r = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3)
            if r.returncode != 0:
                return None

            lines = r.stdout.strip().split("\n")
            pages = {}
            for line in lines[1:]:  # 跳过标题行
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key = key.strip().rstrip(".")
                val = val.strip().rstrip(".").replace(",", "")
                try:
                    pages[key] = int(val)
                except ValueError:
                    pass

            free = pages.get("Pages free", 0)
            active = pages.get("Pages active", 0)
            inactive = pages.get("Pages inactive", 0)
            wired = pages.get("Pages wired down", 0)
            compressed = pages.get("Pages occupied by compressor", 0)
            total = free + active + inactive + wired + compressed
            if total == 0:
                return None

            used = active + wired + compressed
            return used / total * 100
        except Exception:
            return None

    @staticmethod
    def _get_memory_percent_linux() -> Optional[float]:
        """通过 /proc/meminfo 解析 Linux 内存。"""
        try:
            with open("/proc/meminfo") as f:
                data = f.read()
            lines = data.strip().split("\n")
            mem_info = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    mem_info[key] = int(val)

            total = mem_info.get("MemTotal", 0)
            free = mem_info.get("MemFree", 0)
            buffers = mem_info.get("Buffers", 0)
            cached = mem_info.get("Cached", 0)
            if total == 0:
                return None

            used = total - free - buffers - cached
            return used / total * 100
        except Exception:
            return None

    def _get_memory_percent(self) -> float:
        """获取整机内存使用率（跨平台）。"""
        # 1. psutil (最可靠)
        val = self._get_memory_percent_psutil()
        if val is not None:
            return val

        # 2. macOS vm_stat
        if sys.platform == "darwin":
            val = self._get_memory_percent_macos()
            if val is not None:
                return val

        # 3. Linux /proc/meminfo
        if sys.platform == "linux":
            val = self._get_memory_percent_linux()
            if val is not None:
                return val

        return 0.0  # 无法获取时直接放行

    # ── 监控循环 ──

    def _run(self):
        warned = False
        while not self._stop.is_set():
            mem = self._get_memory_percent()

            if mem >= self.threshold:
                print(
                    f"\n\033[31m[内存保护] {self.label}: "
                    f"整机内存 {mem:.0f}% ≥ {self.threshold:.0f}%，终止进程。\033[0m",
                    flush=True,
                )
                os.kill(os.getpid(), signal.SIGTERM)
                # 如果 SIGTERM 没生效，强制退出
                sys.exit(1)

            if mem >= self.threshold - 10 and not warned:
                print(
                    f"\033[33m[内存保护] {self.label}: "
                    f"整机内存 {mem:.0f}%，接近阈值 {self.threshold:.0f}%\033[0m",
                    flush=True,
                )
                warned = True
            elif mem < self.threshold - 15:
                warned = False

            self._stop.wait(self.interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"memguard-{self.label}"
        )
        self._thread.start()
        print(
            f"\033[2m[内存保护] {self.label}: "
            f"已启动 (阈值 {self.threshold:.0f}%)\033[0m",
            flush=True,
        )

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        print(f"\033[2m[内存保护] {self.label}: 已停止\033[0m", flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ── 模块级函数 ──


def start_memguard(threshold: float = 90, interval: float = 2.0, label: str = ""):
    """启动全局内存保护（幂等）。"""
    global _guardian
    if _guardian is not None:
        return
    _guardian = _MemoryGuardian(threshold=threshold, interval=interval, label=label)
    _guardian.start()


def stop_memguard():
    """停止全局内存保护。"""
    global _guardian
    if _guardian is not None:
        _guardian.stop()
        _guardian = None


def memguard(threshold: float = 90, interval: float = 2.0, label: str = ""):
    """上下文管理器：作用域内启用内存保护。"""
    return _MemoryGuardian(threshold=threshold, interval=interval, label=label)
