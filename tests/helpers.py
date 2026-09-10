"""测试公用的脚手架：隔离的 HOME 和项目目录、指向假网关的规则文件。"""
from __future__ import annotations

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mock_gateway import base_url_for, start_server  # noqa: E402


def real_profile() -> dict:
    with open(os.path.join(ROOT, "gateway_profile.json"), encoding="utf-8") as f:
        return json.load(f)


def profile_for(server, tmpdir: str, **overrides) -> str:
    """把真实规则里的网关地址换成假网关的地址，其余保持一致。"""
    p = real_profile()
    p["base_url"]["canonical_root"] = base_url_for(server)
    p["base_url"]["require_https"] = False
    p.update(overrides)
    path = os.path.join(tmpdir, "profile.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False)
    return path


def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def subprocess_env(**overrides) -> dict:
    """给要 subprocess.run 的子进程拼一份隔离环境。
    Windows 上 socket()/ThreadingHTTPServer 需要能找到 SystemRoot 才能初始化
    Winsock 服务提供者，环境剥得太干净会直接报 WinError 10106。"""
    env = {"PATH": os.environ.get("PATH", "")}
    if os.name == "nt":
        for key in ("SystemRoot", "windir", "TEMP", "TMP", "COMSPEC"):
            val = os.environ.get(key)
            if val:
                env[key] = val
    env.update(overrides)
    return env


class Sandbox:
    """一次测试用的隔离环境：独立的 HOME、项目目录和假网关。"""

    def __init__(self, behavior: str = "ok"):
        self.behavior = behavior

    def __enter__(self):
        self._home = tempfile.TemporaryDirectory()
        self._proj = tempfile.TemporaryDirectory()
        self._cfg = tempfile.TemporaryDirectory()
        self.server = start_server(default_behavior=self.behavior)
        self.home = self._home.name
        self.project = self._proj.name
        self.profile_path = profile_for(self.server, self._cfg.name)
        self.base_url = base_url_for(self.server)
        self.env = {"HOME": self.home, "PATH": os.environ.get("PATH", ""),
                    # 隔离环境里不该真的去调本机装的客户端——那样测试结果会取决于
                    # 跑测试这台机器上有没有装、登录没登录，而不是取决于被测的逻辑。
                    # 默认换成一个必然失败的命令（等价于"这台机器上客户端跑不通"），
                    # 需要验证"自证通过"那条分支的用例自己覆盖这个变量。
                    "SUTURE_SELFTEST_COMMAND": "python3 -c exit(3)"}
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        for d in (self._home, self._proj, self._cfg):
            d.cleanup()
        return False
