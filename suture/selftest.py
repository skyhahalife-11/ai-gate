"""让客户端自己证明能不能用。

有一类用户在客户端里什么都没填——地址、Key 都没有——但一直用得好好的，
因为连接是在客户端之外解决的（比如公司网络把流量直接接到网关）。对这类机器，
Suture 自己发不出那次真实请求（没有地址可发），静态检查只能一路说"没配置"，
结果把一个正常状态报成满屏故障。

唯一可靠的判断办法是调用客户端本体跑一次最小的非交互请求：它会用**它自己**
解析出来的全套配置和凭据（包括登录态、包括藏在 Suture 读不到的地方的环境变量）。
跑通了，就说明这台机器现在能正常使用，跟配置长什么样无关。

这一步会真的消耗一次调用，所以只在"Suture 自己压根没法验证"的时候才做。
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

ENV_COMMAND_OVERRIDE = "SUTURE_SELFTEST_COMMAND"     # 测试用；部署方式特殊时也可以用
DEFAULT_TIMEOUT = 45.0


@dataclass
class SelfTestResult:
    attempted: bool                 # 有没有真的跑起来（找不到客户端就是 False）
    ok: bool
    detail: str
    command: str = ""


def _resolve(argv: List[str], env: Dict[str, str]) -> Optional[List[str]]:
    """把命令名解析成真实可执行文件。Windows 上客户端往往是 claude.cmd 这类，
    shutil.which 会按 PATHEXT 自己找，不用我们分平台猜。找不到就返回 None——
    没装客户端不是错误，只是这次自证做不了。"""
    exe = shutil.which(argv[0], path=env.get("PATH") or os.environ.get("PATH"))
    if not exe:
        return None
    return [exe] + list(argv[1:])


def run(argv: Optional[List[str]], env: Optional[Dict[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT) -> SelfTestResult:
    env = dict(env if env is not None else os.environ)

    override = env.get(ENV_COMMAND_OVERRIDE)
    if override:
        argv = shlex.split(override)
    if not argv:
        return SelfTestResult(attempted=False, ok=False,
                              detail="该客户端尚未确认可用的非交互调用方式，跳过实测。")

    resolved = _resolve(argv, env)
    if resolved is None:
        return SelfTestResult(
            attempted=False, ok=False,
            detail=f"未在本机找到 {argv[0]} 命令，无法执行客户端自测。")

    shown = " ".join(argv)
    try:
        proc = subprocess.run(resolved, env=env, timeout=timeout,
                              stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        return SelfTestResult(attempted=True, ok=False, command=shown,
                              detail=f"客户端自测（{shown}）超过 {int(timeout)} 秒未返回。")
    except OSError as exc:
        return SelfTestResult(attempted=True, ok=False, command=shown,
                              detail=f"调用客户端失败：{exc}")

    out = (proc.stdout or b"").decode("utf-8", "replace").strip()
    err = (proc.stderr or b"").decode("utf-8", "replace").strip()
    if proc.returncode == 0 and out:
        return SelfTestResult(attempted=True, ok=True, command=shown,
                              detail=f"客户端自测（{shown}）已执行，正常收到回复。")
    reason = err or out or f"退出码 {proc.returncode}"
    # 客户端自己的报错原文对排查很有用，但可能很长，截断后原样带上，不改写、不猜。
    if len(reason) > 300:
        reason = reason[:300] + "…"
    return SelfTestResult(attempted=True, ok=False, command=shown,
                          detail=f"客户端自测（{shown}）已执行，未成功：{reason}")
