"""图形界面用的本地服务。

只监听 127.0.0.1，端口由系统随机分配，并且每次启动生成一个一次性令牌——
这个接口能读到配置里的 Key，不能让本机上其它程序随便调用。
"""
from __future__ import annotations

import json
import os
import secrets
import threading
from dataclasses import asdict, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from . import engine as E, installer
from .harness import ALL_ADAPTERS, get_adapter

UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")


class _State:
    def __init__(self, engine: E.Engine):
        self.engine = engine
        self.token = secrets.token_urlsafe(24)
        self.last_report: Optional[E.Report] = None
        self.lock = threading.Lock()
        # 安装单独一把锁：它可能跑好几分钟，占着主锁会把 state/check 全卡住。
        # 用非阻塞获取，第二个安装请求直接被告知"已经在装了"而不是并发跑起来
        # （两个 npm install -g 同时写同一个全局前缀，谁也不知道会装成什么样）。
        self.install_lock = threading.Lock()


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def _client_meta(engine: E.Engine, adapter) -> Dict[str, Any]:
    """state 里给界面看的每个客户端静态信息（不触发任何请求/读写）。"""
    cmd = adapter.install_command(env=engine.env)
    return {
        "client_id": adapter.harness_id,
        "display_name": adapter.display_name,
        "binary_installed": adapter.detect_binary(env=engine.env, home=engine.home),
        "config_present": bool(
            getattr(adapter.read(env=engine.env, home=engine.home,
                                 project_dir=engine.project_dir,
                                 known_keys=engine.profile.get("known_settings_keys", []),
                                 accepted_headers=engine.profile.get("auth", {}).get("accepted_headers", []))
                    , "has_any_config", False)),
        "install_command": " ".join(cmd) if cmd else None,
        "has_auto_install": cmd is not None,
        "install_guide": adapter.install_guide(env=engine.env) or None,
    }


class Handler(BaseHTTPRequestHandler):
    state: _State = None       # 由 create_server 绑定

    def log_message(self, fmt, *args):
        pass

    # ---- 鉴权 ----
    def _authorized(self) -> bool:
        header = self.headers.get("X-Suture-Token")
        if header and secrets.compare_digest(header, self.state.token):
            return True
        if "?" in self.path:
            query = self.path.split("?", 1)[1]
            for part in query.split("&"):
                if part.startswith("token="):
                    return secrets.compare_digest(part[len("token="):], self.state.token)
        return False

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def _record_client(self, client_dict: Dict[str, Any]) -> None:
        """把 recheck / configure 返回的最新单客户端写回 last_report，让随后的
        /api/action 能找到刚出现的原因（比如全新用户配置连不上后立刻想修它）。
        若还没有过 /api/check，就补一个空报告承接，而不是让 action 报 409。"""
        if not client_dict or not client_dict.get("client_id"):
            return
        report = self.state.last_report
        if report is None:
            report = E.Report(profile_source=self.state.engine.profile_source)
            self.state.last_report = report
        valid = {f.name for f in fields(E.ClientState)}
        client = E.ClientState(**{k: v for k, v in client_dict.items() if k in valid})
        # issues 里带着修复载荷（可能是原始 Key），而这份 dict 同时也会被交给
        # 浏览器（出网前会脱敏）。这里把 issues 拷一份自己持有，服务端这份就和
        # 出网那份彻底分家——以后无论谁动其中一份，都不会误伤另一份。
        client.issues = [dict(i) if isinstance(i, dict) else i for i in (client.issues or [])]
        for i, c in enumerate(report.clients):
            if c.client_id == client.client_id:
                report.clients[i] = client
                return
        report.clients.append(client)

    # ---- 路由 ----
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            if not self._authorized():
                self._send(403, b"forbidden", "text/plain; charset=utf-8")
                return
            with open(os.path.join(UI_DIR, "index.html"), "r", encoding="utf-8") as f:
                html = f.read().replace("__SUTURE_TOKEN__", self.state.token)
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/state":
            if not self._authorized():
                self._send_json({"error": "forbidden"}, 403)
                return
            eng = self.state.engine
            models = eng.profile.get("models", [])
            # 这条路由读三份真实配置（每个 harness 一次 read），哪一份的文件形态
            # 出乎意料都可能抛。没有兜底的话连接会被直接关掉，前端的 api() 抛错、
            # 又被 reloadState 的空 catch 吞掉 —— 用户看到的是安装页整片空白、
            # 页头永远「加载中」，一句提示都没有。跟其它路由一个口径：如实报错。
            try:
                clients = [_client_meta(eng, a) for a in ALL_ADAPTERS]
            except Exception as exc:      # noqa: BLE001
                self._send_json({"error": f"读取本机配置失败：{exc}"}, 500)
                return
            self._send_json({
                "profile_source": eng.profile_source,
                "gateway_name": eng.profile.get("gateway_name", "AI Gate"),
                "model_count": len(models),
                "models": [{"id": m.get("id"), "display": m.get("display") or m.get("id")}
                           for m in models],
                "gateway_url": eng.profile.get("base_url", {}).get("canonical_root", ""),
                "runtime": installer.check_runtime(eng.env),
                "clients": clients,
            })
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if not self._authorized():
            self._send_json({"error": "forbidden"}, 403)
            return
        payload = self._body()

        # 安装命令可能跑好几分钟，不能在全局锁里做（会卡住 state/check 等请求）
        if path == "/api/install_run":
            client_id = payload.get("client_id")
            try:
                adapter = get_adapter(client_id) if client_id else None
            except KeyError:
                adapter = None
            if adapter is None:
                self._send_json({"error": "没有这个客户端。"}, 400)
                return
            # 同一时刻只允许一个安装。用非阻塞获取：正在装的时候再点（另一个
            # 客户端、或刷新后的同一个）直接如实告知，不再并发跑第二个安装——
            # 两个 npm install -g 同时写同一个全局前缀，结果不可预期。
            if not self.state.install_lock.acquire(blocking=False):
                self._send_json({"ok": False, "ran": False, "timed_out": False, "output": "",
                                 "message": "已经有一个安装在进行中，请等它结束后再试。"},
                                409)
                return
            try:
                result = installer.run_install(adapter, env=self.state.engine.env)
            finally:
                self.state.install_lock.release()
            self._send_json(result)
            return

        with self.state.lock:
            eng = self.state.engine

            if path == "/api/check":
                # 这是唯一会跑完整检查的路由，也是唯一不能让它抛出去的路由：
                # 抛出去前端只会看到一句英文 "Failed to fetch"，用户既不知道为什么、
                # 也没有任何可操作的信息。所有其它路由都有同样的兜底。
                try:
                    report = eng.run(client_ids=payload.get("client_ids"))
                except Exception as exc:      # noqa: BLE001
                    self._send_json({"error": f"检查失败：{exc}"}, 500)
                    return
                self.state.last_report = report
                self._send_json(E.report_to_dict(report))
                return

            if path == "/api/action":
                client_id = payload.get("client_id")
                issue_id = payload.get("issue_id")
                report = self.state.last_report
                issue = None
                if report and client_id:
                    target = next((c for c in report.clients if c.client_id == client_id), None)
                    if target:
                        issue = next((i for i in target.issues if i.get("id") == issue_id), None)
                if issue is None:
                    self._send_json({"error": "没有对应的待修项，请先重新检查。"}, 409)
                    return
                try:
                    result = eng.apply_action(client_id, issue, payload.get("value"))
                except Exception as exc:      # noqa: BLE001
                    self._send_json({"error": f"操作失败：{exc}"}, 500)
                    return
                # 动作会改变状态、可能暴露新一层的问题——把最新结果写回 last_report，
                # 否则用户紧接着点新出现的 issue 会在旧报告里找不到而报 409。
                if result.get("client"):
                    self._record_client(result["client"])
                self._send_json(_redact_result(result))
                return

            if path == "/api/recheck":
                client_id = payload.get("client_id")
                if not client_id:
                    self._send_json({"error": "缺少 client_id"}, 400)
                    return
                try:
                    client = eng.assess_client(client_id)
                except Exception as exc:      # noqa: BLE001
                    self._send_json({"error": f"重新检查失败：{exc}"}, 500)
                    return
                self._record_client(asdict(client))
                self._send_json({"client": E.redact_client(asdict(client))})
                return

            if path == "/api/configure":
                client_id = payload.get("client_id")
                if not client_id:
                    self._send_json({"error": "缺少 client_id"}, 400)
                    return
                try:
                    result = eng.configure_client(client_id, model=payload.get("model"),
                                                  api_key=payload.get("api_key"))
                except Exception as exc:      # noqa: BLE001
                    self._send_json({"error": f"配置失败：{exc}"}, 500)
                    return
                if result.get("client"):
                    self._record_client(result["client"])
                self._send_json(_redact_result(result))
                return

        self._send_json({"error": "not found"}, 404)


def _redact_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """apply_action / configure_client 的返回值里带着一份客户端数据，
    出网前统一脱敏（服务端自己的 last_report 仍保留原始值，点「修复」时要用）。

    脱敏返回的是副本，所以这里必须把结果**替换**掉，不能指望就地改生效——
    就地改会连 last_report 里那份一起抹掉（见 engine.redact_client 的说明）。"""
    if isinstance(result, dict) and isinstance(result.get("client"), dict):
        result["client"] = E.redact_client(result["client"])
    return result


def create_server(engine: E.Engine, port: int = 0):
    state = _State(engine)
    handler_cls = type("BoundHandler", (Handler,), {"state": state})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    return httpd, state


def serve_in_background(engine: E.Engine, port: int = 0):
    httpd, state = create_server(engine, port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/?token={state.token}"
    return httpd, state, url
