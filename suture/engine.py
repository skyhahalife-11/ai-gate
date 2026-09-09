"""流程编排。命令行和图形界面调用的是同一套引擎，不存在两份判断逻辑。

本版是「连通优先」语义：
  每个客户端只有四态 —— connected（能连上，绿勾，不罗列任何细节）
                             / blocked（连不上，展开"可能的原因"，用户挑着修）
                             / unconfigured（装了但没配置，引导配置）
                             / not_installed（没装，引导去安装）。
  成功路径不做逐项静态体检（那些 checks 只在连不上时当"诊断器"用），
  写配置永远走 备份 → 写入 → 重新验证 的流程，Key 只以掩码/请求头出现。
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from . import fixer, gateway, selftest
from .checks import blocked_reasons
from .harness import ALL_ADAPTERS, get_adapter
from .harness.base import FIELD_AUTH, FIELD_BASE_URL, FIELD_MODEL, HarnessConfig
from .profile import expected_base_url, expected_base_urls, load_profile, model_ids

# 整体结论（顶部总览 / CLI 退出码用）
RESULT_HEALTHY = "healthy"                  # 有客户端能正常用
RESULT_BLOCKED = "blocked"                  # 有客户端连不上
RESULT_SETUP = "setup"                      # 有客户端装了但还没配置
RESULT_GATEWAY_DOWN = "gateway_down"        # 网关侧问题，不碰本地配置
RESULT_NOT_INSTALLED = "not_installed"      # 本机还没装任何客户端
RESULT_MANUAL = "manual"                    # 动作没做（需要用户选/外部操作）

# 单条动作的结果
RESULT_FIXED = "fixed"                      # 改完就连上了
RESULT_CHANGED = "changed"                  # 已写入，仍需继续处理

# 每客户端四态
STATE_CONNECTED = "connected"
STATE_BLOCKED = "blocked"
STATE_UNCONFIGURED = "unconfigured"
STATE_NOT_INSTALLED = "not_installed"

STATE_LABEL = {
    STATE_CONNECTED: "可以连接",
    STATE_BLOCKED: "连不上",
    STATE_UNCONFIGURED: "还没配置",
    STATE_NOT_INSTALLED: "还没安装",
}


@dataclass
class ClientState:
    client_id: str
    display_name: str
    state: str
    state_label: str
    detail: str = ""
    binary_installed: bool = False
    config_present: bool = False
    files: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)   # 仅 blocked 时非空
    e2e: Optional[Dict[str, Any]] = None
    generated_path: str = ""


@dataclass
class Report:
    profile_source: str
    gateway: Optional[Dict[str, Any]] = None
    gateway_side_down: bool = False
    result: str = ""
    message: str = ""
    clients: List[ClientState] = field(default_factory=list)


def _wire(profile: Dict[str, Any], harness_id: str) -> Dict[str, str]:
    """这个 harness 真实会发的请求路径和报文形态。"""
    table = profile.get("per_harness_request", {})
    entry = table.get(harness_id) or {"suffix": "/v1/messages", "style": "anthropic"}
    return {"suffix": entry.get("suffix", "/v1/messages"),
            "style": entry.get("style", "anthropic")}


def _probe_dict(p: gateway.ProbeResult) -> Dict[str, Any]:
    return {"ok": p.ok, "classification": p.classification, "status": p.status,
            "detail": p.detail, "elapsed_ms": p.elapsed_ms}


def _auth_headers(cfg: Optional[HarnessConfig], fallback_key: Optional[str] = None) -> Dict[str, str]:
    """按这个 harness 真实客户端的行为拼出这次请求应该带的鉴权请求头——
    主字段（API_KEY/AUTH_TOKEN 二选一解析出来的那个）加上所有平行生效的自定义头
    一起带上，不替用户猜网关到底认哪一个。只有当两者都没有时才退回到
    fallback_key（用于探活，顺带验证网关本身通不通）。"""
    headers: Dict[str, str] = {}
    if cfg is not None:
        value = cfg.field(FIELD_AUTH).value
        if value:
            if cfg.auth_header == "Authorization":
                headers["Authorization"] = f"Bearer {value}"
            else:
                headers[cfg.auth_header] = value
        for extra in cfg.extra_auth_headers:
            if extra.value:
                headers.setdefault(extra.header, extra.value)
    if not headers and fallback_key:
        headers["x-api-key"] = fallback_key
    return headers


class Engine:
    def __init__(self, profile_path: Optional[str] = None, env=None,
                 home: Optional[str] = None, project_dir: Optional[str] = None):
        self.profile, self.profile_source = load_profile(profile_path)
        self.env = env if env is not None else os.environ
        self.home = home
        self.project_dir = project_dir or os.getcwd()
        self._configs: Dict[str, HarnessConfig] = {}
        self._backups: Dict[str, fixer.BackupManifest] = {}

    # ---- 读取 ----
    def read_harness(self, harness_id: str) -> HarnessConfig:
        adapter = get_adapter(harness_id)
        cfg = adapter.read(env=self.env, home=self.home, project_dir=self.project_dir,
                           known_keys=self.profile.get("known_settings_keys", []),
                           accepted_headers=self.profile.get("auth", {}).get("accepted_headers", []))
        self._configs[harness_id] = cfg
        return cfg

    # ---- 阶段一：网关探活 ----
    def probe(self, cfg: Optional[HarnessConfig] = None) -> gateway.ProbeResult:
        """探活的是规则里那个已知正确的地址，不是用户配置里那个可能填错的地址——
        否则用户把地址填错时，工具会误判成"网关侧问题，不要动本地配置"。
        网关登记多个入口时挨个试，只要有一个不是网关侧问题就说明网关是活的。
        鉴权优先用 profile 的 probe_key（跟用户配置解耦），没有才借用户 Key。"""
        probe_model = self.profile.get("probe_model")
        wire = _wire(self.profile, "claude_code")
        probe_key = self.profile.get("probe_key")
        if probe_key:
            headers = {"x-api-key": probe_key}
        else:
            headers = _auth_headers(cfg, fallback_key=self._any_known_key())

        urls = expected_base_urls(self.profile, "claude_code")
        if not urls:
            urls = [expected_base_url(self.profile, "claude_code")]

        result: Optional[gateway.ProbeResult] = None
        for base_url in urls:
            result = gateway.probe_gateway(base_url, headers, probe_model,
                                           suffix=wire["suffix"], style=wire["style"])
            if result.classification not in gateway.GATEWAY_SIDE:
                return result
        return result

    def _any_known_key(self) -> Optional[str]:
        for cfg in self._configs.values():
            value = cfg.field(FIELD_AUTH).value
            if value:
                return value
        return None

    def _probe_model(self) -> str:
        return self.profile.get("probe_model") or ""

    # ---- 连通目标模型 ----
    def _e2e_candidates(self, cfg: HarnessConfig) -> List[str]:
        """真实请求要试的模型名，按顺序试到有一个成功为止。

        决定性的是客户端自己配置的那些模型——一个名字连不通，客户端真实就用不了，
        不能拿探活模型（网关保证认识）去代替它、把"模型名配错"掩盖成已连通。
        只有客户端压根没指定模型时，才用探活模型当代理测一次传输/鉴权是否通。"""
        if cfg.model_candidates:
            return [m for m in cfg.model_candidates if str(m).strip()]
        if cfg.field(FIELD_MODEL).is_set:
            return [cfg.field(FIELD_MODEL).value or ""]
        pm = self._probe_model()
        return [pm] if pm else []

    @staticmethod
    def _client_files(cfg: HarnessConfig) -> List[Dict[str, Any]]:
        return [{"layer": f.layer, "path": f.path, "exists": f.exists,
                 "active": f.active, "parse_ok": f.parse_ok,
                 "inactive_reason": f.inactive_reason or ""} for f in cfg.files]

    def _selftest_dict(self, adapter) -> Dict[str, Any]:
        st = selftest.run(adapter.self_test_command(), env=self.env)
        return {"attempted": st.attempted, "ok": st.ok,
                "detail": st.detail, "command": st.command}

    # ---- 阶段二：单个客户端四态判定 ----
    def assess_client(self, client_id: str) -> ClientState:
        adapter = get_adapter(client_id)
        cfg = self.read_harness(client_id)
        binary = adapter.detect_binary(env=self.env)
        base = ClientState(
            client_id=client_id, display_name=adapter.display_name, state="",
            state_label="", binary_installed=binary, config_present=cfg.has_any_config,
            files=self._client_files(cfg))

        def mark(state: str, detail: str):
            base.state = state
            base.state_label = STATE_LABEL[state]
            base.detail = detail
            return base

        # 一个字都没配：分"没装客户端"和"装了但没配"
        if not cfg.has_any_config:
            if not binary:
                return mark(STATE_NOT_INSTALLED,
                            f"本机没有找到 {adapter.display_name}，也没找到它的配置。"
                            "到「安装 / 上手」页装上它，再回来连 AI Gate。")
            st = self._selftest_dict(adapter)
            if st["attempted"] and st["ok"]:
                return mark(STATE_CONNECTED,
                            "客户端没配置网关，但自己实测能正常使用——这种情况下不需要"
                            "在客户端里填网关地址和 Key。")
            return mark(STATE_UNCONFIGURED,
                        f"已安装 {adapter.display_name}，但还没配置连 AI Gate。点「配置连接 AI Gate」。")

        base_url = cfg.field(FIELD_BASE_URL).value
        wire = _wire(self.profile, client_id)
        results: List[Dict[str, Any]] = []

        if base_url:
            for m in self._e2e_candidates(cfg)[:4]:
                r = gateway.send_real_request(base_url, _auth_headers(cfg), m,
                                              suffix=wire["suffix"], style=wire["style"])
                results.append({"model": m, **_probe_dict(r)})
                if r.ok:
                    break
        base.e2e = results[-1] if results else None

        if base_url and any(r["ok"] for r in results):
            return mark(STATE_CONNECTED, "当前配置能正常连上 AI Gate。")

        # 连不上 → 展开"可能的原因"
        if not base_url:
            if not cfg.field(FIELD_AUTH).is_set:
                st = self._selftest_dict(adapter)
                if st["attempted"] and st["ok"]:
                    return mark(STATE_CONNECTED,
                                "客户端实测能正常使用——连接是在客户端之外解决的，"
                                "不需要在客户端里填网关地址和 Key。")
            base.detail = "客户端里没有可用的网关地址，连不上 AI Gate。先补网关地址或 Key。"
        else:
            base.detail = "当前配置连不上 AI Gate。下面列出可能的原因，挑一个处理。"
        base.state = STATE_BLOCKED
        base.state_label = STATE_LABEL[STATE_BLOCKED]
        base.issues = blocked_reasons(cfg, self.profile, base.e2e)
        return base

    # ---- 网关侧 down 时的逐客户端简报 ----
    def _client_gateway_down(self, client_id: str) -> ClientState:
        adapter = get_adapter(client_id)
        cfg = self.read_harness(client_id)
        binary = adapter.detect_binary(env=self.env)
        base = ClientState(
            client_id=client_id, display_name=adapter.display_name, state=STATE_BLOCKED,
            state_label=STATE_LABEL[STATE_BLOCKED], binary_installed=binary,
            config_present=cfg.has_any_config, files=self._client_files(cfg))
        if not cfg.has_any_config:
            base.state = STATE_NOT_INSTALLED if not binary else STATE_UNCONFIGURED
            base.state_label = STATE_LABEL[base.state]
            base.detail = ("网关侧现在也连不上。" + ("装好之后再来连。" if not binary
                           else "先配置，配置好了再连。"))
            return base
        base.detail = "网关侧连不上，暂不逐项检查本地配置（避免白改）。"
        base.issues = [{
            "id": "gateway-side", "title": "AI Gate 网关侧暂时连不上",
            "detail": "网关本身连不上（网络或网关侧问题），这次先不用改本地配置。"
                      "稍后重试；如果一直这样，联系网关值班人员。",
            "current_value": "", "severity": "high", "repair_kind": "external",
            "fix_field": None, "fix_value": None, "choices": [], "prompt": None}]
        return base

    # ---- 完整一轮 ----
    def run(self, client_ids: Optional[List[str]] = None) -> Report:
        adapters = [get_adapter(x) for x in client_ids] if client_ids else ALL_ADAPTERS
        report = Report(profile_source=self.profile_source)
        if not adapters:
            report.result = RESULT_NOT_INSTALLED
            report.message = "本机还没有可检查的客户端。"
            return report

        probe_cfg = None
        for a in adapters:
            c = self.read_harness(a.harness_id)
            if c.field(FIELD_BASE_URL).is_set:
                probe_cfg = c
                break
        probe = self.probe(probe_cfg)
        report.gateway = _probe_dict(probe)

        if probe.classification in gateway.GATEWAY_SIDE:
            report.gateway_side_down = True
            report.result = RESULT_GATEWAY_DOWN
            report.message = "AI Gate 网关本身连不上（网络或网关侧问题），暂时先不用改本地配置。"
            for a in adapters:
                report.clients.append(self._client_gateway_down(a.harness_id))
            return report

        for a in adapters:
            report.clients.append(self.assess_client(a.harness_id))

        states = [c.state for c in report.clients]
        if STATE_BLOCKED in states:
            report.result = RESULT_BLOCKED
            report.message = "有客户端连不上 AI Gate。展开对应客户端，按原因处理。"
        elif STATE_UNCONFIGURED in states:
            report.result = RESULT_SETUP
            report.message = "有客户端装了但还没配置连 AI Gate，先配置它。"
        elif all(s == STATE_NOT_INSTALLED for s in states):
            report.result = RESULT_NOT_INSTALLED
            report.message = "本机还没有这些客户端。到「安装 / 上手」页装一个，就能连 AI Gate。"
        elif STATE_CONNECTED in states:
            report.result = RESULT_HEALTHY
            report.message = "连上 AI Gate 的客户端能正常使用。"
        else:
            report.result = RESULT_NOT_INSTALLED
            report.message = "还没有能连上 AI Gate 的客户端。"
        return report

    # ---- 单条动作（auto / choice / input）----
    def apply_action(self, client_id: str, issue: Dict[str, Any],
                     value: Optional[str] = None) -> Dict[str, Any]:
        adapter = get_adapter(client_id)
        cfg = self.read_harness(client_id)
        kind = issue.get("repair_kind") or "none"
        steps: List[Dict[str, str]] = []

        if kind in ("none", "external"):
            return {"result": RESULT_MANUAL, "message": issue.get("title") or "这一项需要在工具外处理。",
                    "note": issue.get("detail", ""), "steps": steps, "backup_dir": None}

        if kind == "input":
            if not value or not str(value).strip():
                return {"result": RESULT_MANUAL, "message": "要先把 Key 粘贴进来才能保存。",
                        "steps": steps, "backup_dir": None}
            key = str(value).strip()
            manifest = fixer.backup_files(adapter.writable_paths(cfg), home=self.home)
            self._backups[client_id] = manifest
            steps.append({"detail": f"已备份原配置到 {manifest.directory}"})
            try:
                store_steps, _changed, note = adapter.store_key(
                    cfg, key, env=self.env, home=self.home, project_dir=self.project_dir)
            except OSError as exc:
                return {"result": RESULT_MANUAL,
                        "message": f"保存 Key 失败：{exc}。原配置未被修改，备份在 {manifest.directory}。",
                        "steps": steps, "backup_dir": manifest.directory}
            steps.extend({"detail": s} for s in store_steps)
            # 让本进程立刻能用这个 Key（Codex/DeepSeek 读 AI_GATE_API_KEY）
            self.env["AI_GATE_API_KEY"] = key
            return self._after_write(client_id, steps, manifest, note,
                                     "Key 已保存，现在能连上 AI Gate 了。",
                                     "Key 已保存。仍连不上——展开下面的原因继续处理。")

        # auto / choice：一次写一个逻辑字段
        if kind == "choice":
            fix_value = value if value is not None else issue.get("fix_value")
        else:
            fix_value = issue.get("fix_value")
        fix_field = issue.get("fix_field")
        if not fix_field or fix_value is None or str(fix_value) == "":
            return {"result": RESULT_MANUAL, "message": "这一项还需要你选一个具体值。",
                    "steps": steps, "backup_dir": None}
        changes = {str(fix_field): str(fix_value)}
        manifest = fixer.backup_files(adapter.writable_paths(cfg), home=self.home)
        self._backups[client_id] = manifest
        steps.append({"detail": f"已备份原配置到 {manifest.directory}"})
        try:
            applied = fixer.apply_fixes(adapter, cfg, changes, env=self.env,
                                        home=self.home, project_dir=self.project_dir)
        except OSError as exc:
            return {"result": RESULT_MANUAL,
                    "message": f"写入配置失败：{exc}。原配置未被修改，备份在 {manifest.directory}。",
                    "steps": steps, "backup_dir": manifest.directory}
        steps.append({"detail": "；".join(applied) if applied else "没有需要写入的改动"})
        return self._after_write(client_id, steps, manifest, None,
                                 "已修改，现在能连上 AI Gate 了。",
                                 "已修改，但仍连不上——按剩下的原因继续，或重测一次。")

    def _after_write(self, client_id: str, steps, manifest, note: Optional[str],
                     ok_msg: str, pending_msg: str) -> Dict[str, Any]:
        client = self.assess_client(client_id)
        result = {
            "steps": steps, "backup_dir": manifest.directory,
            "client": asdict(client), "note": note,
        }
        if client.state == STATE_CONNECTED:
            result["result"] = RESULT_FIXED
            result["message"] = ok_msg
        else:
            result["result"] = RESULT_CHANGED
            result["message"] = pending_msg
        return result

    # ---- 全新用户一键配置连 AI Gate ----
    def configure_client(self, client_id: str, model: Optional[str] = None,
                         api_key: Optional[str] = None) -> Dict[str, Any]:
        adapter = get_adapter(client_id)
        known = model_ids(self.profile)
        pm = self._probe_model()
        pick = model if (model and model in known) else (pm if pm in known else (known[0] if known else ""))
        path = adapter.generate_minimal_config(
            expected_base_url(self.profile, client_id), pick,
            env=self.env, home=self.home, project_dir=self.project_dir)
        steps = [{"detail": f"已生成最小配置：{path}"}]
        note = None

        cfg = self.read_harness(client_id)
        if api_key and str(api_key).strip():
            key = str(api_key).strip()
            manifest = fixer.backup_files(adapter.writable_paths(cfg), home=self.home)
            self._backups[client_id] = manifest
            steps.append({"detail": f"已备份原配置到 {manifest.directory}"})
            store_steps, _changed, note = adapter.store_key(
                cfg, key, env=self.env, home=self.home, project_dir=self.project_dir)
            steps.extend({"detail": s} for s in store_steps)
            self.env["AI_GATE_API_KEY"] = key

        client = self.assess_client(client_id)
        if client.state == STATE_CONNECTED:
            msg = "配置完成，现在能连上 AI Gate 了。"
        elif client.state == STATE_BLOCKED:
            msg = "配置已生成。还连不上——按下面的原因继续处理。"
        else:
            msg = "配置已生成。"
        return {"path": path, "message": msg, "note": note,
                "steps": steps, "client": asdict(client)}


def report_to_dict(report: Report) -> Dict[str, Any]:
    """报告要能直接序列化给界面。Key 在检测层已经掩码，这里不会再引入明文。"""
    return asdict(report)
