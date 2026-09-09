"""测试用的假网关，不是交付的一部分。

行为尽量贴近真实网关已经确认过的规则：只认 /v1/messages 路径；
模型名必须在路由表里。默认 behavior=ok 时三个鉴权头都认（宽松，便于测其它
分支）；behavior=token_only 则模拟 AI Gate 真实契约——只从 Token 头取 Key，
x-api-key / Authorization 一律 401（用于验收「Key 没放进 Token 头」的判定）。
可以通过 default_behavior 或 X-Mock-Behavior 头控制这次返回什么。
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

KNOWN_MODELS = {
    "gpt-5.5", "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    "claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-5",
    "gemini-3.1-pro-preview", "gemini-3.5-flash", "gemini-3.7-flash",
    "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp",
    "glm-5.3", "glm-5.3-flash",
    "kimi-k2.7-code", "kimi-k2.7-code-highspeed", "kimi-k2.6", "kimi-k3",
}
KEY_PREFIX = "yotta_pk_"
BASE_PATH = "/zi/proxy"


class Handler(BaseHTTPRequestHandler):
    DEFAULT_BEHAVIOR = "ok"
    REQUEST_COUNT = None

    def log_message(self, fmt, *args):
        pass

    def _behavior(self) -> str:
        header = self.headers.get("X-Mock-Behavior")
        if header:
            return header
        default = self.DEFAULT_BEHAVIOR
        if default.startswith("flaky_after:"):
            n = int(default.split(":", 1)[1])
            self.REQUEST_COUNT[0] += 1
            return "ok" if self.REQUEST_COUNT[0] <= n else "server_error"
        return default

    def do_POST(self):
        behavior = self._behavior()
        if behavior == "timeout":
            time.sleep(30)
            return

        # 网关自身故障发生在路由之前：一个真的挂了的网关，不管请哪个路径都是这个错
        if behavior == "rate_limited":
            self._reply(429, {"error": "rate limited"}); return
        if behavior == "server_error":
            self._reply(500, {"error": "internal error"}); return

        # token_only：模拟 AI Gate 只采信 Token 头，其它头即使带了 key 也 401
        token_only = behavior == "token_only"
        if token_only:
            behavior = "ok"

        # 两种端点：Anthropic 形态 <root>/zi/proxy + /v1/messages，
        # OpenAI 形态 <root>/zi/proxy/v1 + /chat/completions
        if self.path not in (f"{BASE_PATH}/v1/messages", f"{BASE_PATH}/v1/chat/completions"):
            self._reply(404, {"error": f"no route for path {self.path}"})
            return

        if token_only:
            key = self.headers.get("Token")
        else:
            key = (self.headers.get("x-api-key")
                   or _bearer(self.headers.get("Authorization"))
                   or self.headers.get("Token"))
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {}

        if not key:
            self._reply(401, {"error": "missing auth"}); return
        if behavior == "auth_error" or not key.startswith(KEY_PREFIX):
            self._reply(401, {"error": "invalid key"}); return

        model = body.get("model")
        if model not in KNOWN_MODELS:
            self._reply(404, {"error": f"model {model} not found"}); return
        self._reply(200, {"id": "msg_mock", "model": model,
                          "content": [{"type": "text", "text": "pong"}]})

    def _reply(self, status: int, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _bearer(value):
    if value and value.startswith("Bearer "):
        return value[len("Bearer "):]
    return None


def start_server(port: int = 0, default_behavior: str = "ok") -> HTTPServer:
    cls = type("BoundHandler", (Handler,),
               {"DEFAULT_BEHAVIOR": default_behavior, "REQUEST_COUNT": [0]})
    server = HTTPServer(("127.0.0.1", port), cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def base_url_for(server: HTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}{BASE_PATH}"
