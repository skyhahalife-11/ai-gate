"""安装 / 上手：探测运行时、执行官方安装命令。

给不熟技术的同事用的：在「安装 / 上手」页装好缺失的客户端。真正的安装命令由
各适配器的 install_command 提供（官方渠道、已核实过包名）。这一层负责
1) 判断本机缺不缺运行时（Claude Code / Codex 都走 npm，得先有 Node.js）；
2) 把命令跑起来并把输出收回来展示，任何情况下都不允许无限期卡住。

安装会真实往系统里装东西，所以命令只取自适配器写死的官方命令 + 明确的
测试覆盖开关，不接收用户任意输入的命令。
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from typing import Any, Dict, List, Optional

INSTALL_OVERRIDE = "SUTURE_INSTALL_CMD_{harness_id}"   # 测试用；某些部署可用同名单命令
DEFAULT_TIMEOUT = 300.0
MAX_OUTPUT = 4000


def _which(name: str, env: Dict[str, str]) -> Optional[str]:
    return shutil.which(name, path=env.get("PATH") or os.environ.get("PATH"))


def check_runtime(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """本机有没有装 Node.js / npm（装 Claude Code、Codex 都依赖它）。"""
    env = dict(env if env is not None else os.environ)
    return {
        "node_present": _which("node", env) is not None,
        "npm_present": _which("npm", env) is not None,
    }


def runtime_guide() -> str:
    return ("装 Claude Code / Codex 需要先有 Node.js。这台机器上还没找到 node/npm："
            "请到 nodejs.org 下载 LTS 版安装，装完重开终端再回来点「安装」。"
            "如果已经装过但这里仍提示没有，多半是终端没重开、PATH 还没刷新。")


def resolve_command(adapter, env: Optional[Dict[str, str]] = None) -> Optional[List[str]]:
    """返回要执行的安装命令。测试/特殊部署用 SUTURE_INSTALL_CMD_<HARNESS_ID>
    覆盖（比如换成一条必然成功的假命令），否则用适配器写死的官方命令。"""
    env = dict(env if env is not None else os.environ)
    flag = env.get(INSTALL_OVERRIDE.format(harness_id=adapter.harness_id.upper()))
    if flag is not None and str(flag).strip():
        return shlex.split(str(flag))
    return adapter.install_command(env=env)


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _tail(text: str, limit: int = MAX_OUTPUT) -> str:
    text = text.rstrip()
    return text if len(text) <= limit else "…（前面省略）\n" + text[-limit:]


def run_install(adapter, env: Optional[Dict[str, str]] = None,
                timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """执行一次安装命令，返回给界面展示的收尾信息。

    {ok, ran, timed_out, output, message}
    输出用临时文件承接而不是管道——npm 这类会派生孙进程，管道会在孙进程还占着
    写端时把父进程的收尾一起拖住（跟客户端自测遇到的同一个坑）。
    """
    import tempfile

    env = dict(env if env is not None else os.environ)
    cmd = resolve_command(adapter, env)

    if not cmd:
        guide = adapter.install_guide(env=env) or "这个客户端的官方安装命令还没确认。"
        return {"ok": False, "ran": False, "timed_out": False,
                "output": "", "message": guide}

    if cmd[0] == "npm" and not check_runtime(env)["npm_present"]:
        return {"ok": False, "ran": False, "timed_out": False,
                "output": "", "message": runtime_guide()}

    # Windows 上 npm 只有 npm.cmd（nodejs.org 的安装包不带 npm.exe），而
    # CreateProcess 不会按 PATHEXT 去补扩展名：上面 check_runtime 用 shutil.which
    # 明明找得到 npm，这里直接 Popen(["npm", ...]) 却一定抛 WinError 2。必须先解析
    # 成真实路径再执行，两边口径才一致。
    exe = shutil.which(cmd[0], path=env.get("PATH") or os.environ.get("PATH"))
    if exe is None:
        # 上面 npm 的检查放过了、这里仍然找不到——如实说清楚，别把 WinError 2 抛给用户。
        return {"ok": False, "ran": False, "timed_out": False, "output": "",
                "message": f"找不到可执行的 {cmd[0]}。请确认它已安装、并且终端里的 PATH "
                           f"能定位到它；装 Node.js 请到 nodejs.org 下载 LTS 版。"}
    cmd = [exe] + list(cmd[1:])

    shown = " ".join(cmd)
    creationflags = 0
    if os.name == "nt":
        creationflags = (subprocess.CREATE_NEW_PROCESS_GROUP
                         | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out")
            with open(out_path, "wb", buffering=0) as fout:
                proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                                        stdout=fout, stderr=subprocess.STDOUT,
                                        creationflags=creationflags)
                timed_out = False
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    _kill_tree(proc)
                    try:
                        proc.wait()
                    except OSError:
                        pass
                    timed_out = True
            try:
                with open(out_path, "rb") as f:
                    raw = f.read().decode("utf-8", "replace")
            except OSError:
                raw = ""
    except OSError as exc:
        return {"ok": False, "ran": False, "timed_out": False,
                "output": "", "message": f"调用安装命令失败：{exc}"}

    output = _tail(raw)
    if timed_out:
        return {"ok": False, "ran": True, "timed_out": True, "output": output,
                "message": f"安装命令（{shown}）超过 {int(timeout)} 秒仍未结束，已终止。"
                           "可能是网络太慢；请重试一次，仍不行就手动执行上面的命令。"}
    if proc.returncode != 0:
        tail = _tail(raw) or f"退出码 {proc.returncode}"
        return {"ok": False, "ran": True, "timed_out": False, "output": output,
                "message": f"安装命令（{shown}）未成功。如果报错与权限有关，试试以管理员"
                           "身份重开终端后手动执行。输出见上方。"}
    return {"ok": True, "ran": True, "timed_out": False, "output": output,
            "message": f"命令（{shown}）执行完成。装完重开终端，到「检查」页测一下。"}
