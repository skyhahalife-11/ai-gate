"""网关规则的加载。

规则不编译进二进制，按以下优先级取用：
1. 命令行参数指定的文件（研发调试）
2. 运行时从网关拉取的最新规则（接口已预留，尚未接通）
3. 随程序分发的一份快照（兜底，离线可用）
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

DEFAULT_PROFILE_NAME = "gateway_profile.json"


def _bundled_profile_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, DEFAULT_PROFILE_NAME)


def load_profile(path: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """返回 (规则内容, 这份规则的来源说明)。来源要能展示给用户，
    让他看得出用的是不是最新规则。"""
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), f"指定的规则文件（{path}）"

    remote = fetch_remote_profile()
    if remote is not None:
        return remote, "从网关拉取的最新规则"

    with open(_bundled_profile_path(), "r", encoding="utf-8") as f:
        return json.load(f), "程序内置的规则快照（可能不是最新的）"


def fetch_remote_profile(base_url: Optional[str] = None, timeout: float = 3.0):
    """从网关拉取最新规则。网关侧接口就绪后在这里实现；
    现在固定返回 None，让调用方退回到内置快照。"""
    return None


def model_ids(profile: Dict[str, Any]):
    return [m["id"] for m in profile.get("models", [])]


def model_index(profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {m["id"]: m for m in profile.get("models", [])}


def expected_base_url(profile: Dict[str, Any], harness_id: str) -> str:
    """不同 harness 拼请求路径的习惯不同，正确的 base_url 也就不同。
    这个只返回「首选」那一个地址（用来生成配置、给一键修复用），
    如果网关不止一个能进的地址，用 expected_base_urls（复数）拿全部。"""
    b = profile.get("base_url", {})
    root = b.get("canonical_root", "").rstrip("/")
    suffix = b.get("per_harness_suffix", {}).get(harness_id, "")
    return root + suffix


def expected_base_urls(profile: Dict[str, Any], harness_id: str) -> list:
    """网关登记的全部能进的地址（首选 + 其它同样有效的入口，比如某些网络环境下
    专用的直连地址），都按这个 harness 拼路径的习惯加上后缀。列表第一个永远是
    首选地址——阶段一探活按顺序试，阶段二检查用户填的地址时命中列表里任意
    一个都算对，不是必须严格等于第一个。"""
    b = profile.get("base_url", {})
    suffix = b.get("per_harness_suffix", {}).get(harness_id, "")
    roots = [b.get("canonical_root", "")] + list(b.get("alternate_roots", []) or [])
    return [r.rstrip("/") + suffix for r in roots if r]
