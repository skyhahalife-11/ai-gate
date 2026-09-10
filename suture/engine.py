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
from .checks import auth_incompatible_issue, auth_requirement_unsatisfiable, blocked_reasons
from .harness import ALL_ADAPTERS, get_adapter
from .harness.base import (
    FIELD_AUTH, FIELD_BASE_URL, FIELD_EXTRA_HEADER, FIELD_MODEL, HarnessConfig, RefuseWrite,
)
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
    STATE_CONNECTED: "已连接",
    STATE_BLOCKED: "无法连接",
    STATE_UNCONFIGURED: "未配置",
    STATE_NOT_INSTALLED: "未安装",
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


def _representative_result(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """多个候选模型都失败时，取最能代表根因的那一次做归因：
    鉴权失败（401/403）优先——它决定整条链路能不能用；其次 400/404（模型或路径）；
    都没有特殊信号才退回最后一次尝试（客户端真实想用的那个模型）。"""
    if not results:
        return None
    for r in results:
        if r.get("classification") == "auth_error" or r.get("status") in (401, 403):
            return r
    for r in results:
        if r.get("status") in (400, 404):
            return r
    return results[-1]


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
        # 探活 Key 也要放进网关真正采信的那个请求头（profile.auth.required_header），
        # 不能想当然塞 x-api-key——AI Gate 只读自定义 Token 头。
        auth_rules = self.profile.get("auth") or {}
        req_header = (auth_rules.get("required_header") or "").strip()
        probe_key = self.profile.get("probe_key") or self._any_known_key()
        if probe_key:
            headers = {req_header: probe_key} if req_header else {"x-api-key": probe_key}
        else:
            headers = _auth_headers(cfg)     # 借配置里已经摆好的请求头（通常是 Token）

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
        binary = adapter.detect_binary(env=self.env, home=self.home)
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
                            f"未找到 {adapter.display_name}。请先安装，再配置连接。")
            st = self._selftest_dict(adapter)
            if st["attempted"] and st["ok"]:
                return mark(STATE_CONNECTED,
                            "客户端可独立工作，无需在配置中填写网关信息。")
            return mark(STATE_UNCONFIGURED,
                        f"{adapter.display_name} 已安装，尚未配置连接。")

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
        base.e2e = _representative_result(results)

        if base_url and any(r["ok"] for r in results):
            return mark(STATE_CONNECTED, "已连接 AI Gate。")

        # 连不上 → 展开"可能的原因"
        if not base_url:
            if not cfg.field(FIELD_AUTH).is_set:
                st = self._selftest_dict(adapter)
                if st["attempted"] and st["ok"]:
                    return mark(STATE_CONNECTED,
                                "客户端可独立工作，无需在配置中填写网关信息。")
            base.detail = "缺少网关地址，无法连接。"
        else:
            base.detail = "当前配置无法连接 AI Gate。"
        base.state = STATE_BLOCKED
        base.state_label = STATE_LABEL[STATE_BLOCKED]
        base.issues = blocked_reasons(cfg, self.profile, base.e2e)
        return base

    # ---- 网关侧 down 时的逐客户端简报 ----
    def _client_gateway_down(self, client_id: str) -> ClientState:
        adapter = get_adapter(client_id)
        cfg = self.read_harness(client_id)
        binary = adapter.detect_binary(env=self.env, home=self.home)
        base = ClientState(
            client_id=client_id, display_name=adapter.display_name, state=STATE_BLOCKED,
            state_label=STATE_LABEL[STATE_BLOCKED], binary_installed=binary,
            config_present=cfg.has_any_config, files=self._client_files(cfg))
        if not cfg.has_any_config:
            base.state = STATE_NOT_INSTALLED if not binary else STATE_UNCONFIGURED
            base.state_label = STATE_LABEL[base.state]
            base.detail = "AI Gate 网关当前不可用，请稍后重试。" + (
                " 安装完成后再试。" if not binary else " 可先完成配置。")
            return base
        base.detail = "AI Gate 网关当前不可用，本地配置未改动。"
        base.issues = [{
            "id": "gateway-side", "title": "AI Gate 网关不可用",
            "detail": "AI Gate 当前不可用，本地配置无需更改。请稍后重试；"
                      "如持续如此，请联系网关维护人员。",
            "current_value": "", "severity": "high", "repair_kind": "external",
            "fix_field": None, "fix_value": None, "choices": [], "prompt": None}]
        return base

    # ---- 完整一轮 ----
    def run(self, client_ids: Optional[List[str]] = None) -> Report:
        adapters = [get_adapter(x) for x in client_ids] if client_ids else ALL_ADAPTERS
        report = Report(profile_source=self.profile_source)
        if not adapters:
            report.result = RESULT_NOT_INSTALLED
            report.message = "没有可检查的客户端。"
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
            report.message = "AI Gate 暂时无法连接，请稍后重试。"
            for a in adapters:
                report.clients.append(self._client_gateway_down(a.harness_id))
            return report

        for a in adapters:
            report.clients.append(self.assess_client(a.harness_id))

        states = [c.state for c in report.clients]
        if STATE_BLOCKED in states:
            report.result = RESULT_BLOCKED
            report.message = "部分客户端无法连接 AI Gate。"
        elif STATE_UNCONFIGURED in states:
            report.result = RESULT_SETUP
            report.message = "部分客户端尚未配置。"
        elif all(s == STATE_NOT_INSTALLED for s in states):
            report.result = RESULT_NOT_INSTALLED
            report.message = "未检测到已安装的客户端。"
        elif STATE_CONNECTED in states:
            report.result = RESULT_HEALTHY
            report.message = "所有客户端均可正常连接。"
        else:
            report.result = RESULT_NOT_INSTALLED
            report.message = "暂无可正常连接的客户端。"
        return report

    # ---- 单条动作（auto / choice / input）----
    def apply_action(self, client_id: str, issue: Dict[str, Any],
                     value: Optional[str] = None) -> Dict[str, Any]:
        adapter = get_adapter(client_id)
        cfg = self.read_harness(client_id)
        kind = issue.get("repair_kind") or "none"
        steps: List[Dict[str, str]] = []

        if kind in ("none", "external"):
            return {"result": RESULT_MANUAL, "message": issue.get("title") or "此步骤需手动完成。",
                    "note": issue.get("detail", ""), "steps": steps, "backup_dir": None}

        if kind == "input":
            if not value or not str(value).strip():
                return {"result": RESULT_MANUAL, "message": "请先输入 Key。",
                        "steps": steps, "backup_dir": None}
            key = str(value).strip()
            manifest = fixer.backup_files(adapter.writable_paths(cfg), home=self.home)
            self._backups[client_id] = manifest
            steps.append({"detail": f"已备份原配置到 {manifest.directory}"})
            try:
                store_steps, _changed, note = adapter.store_key(
                    cfg, key, env=self.env, home=self.home, project_dir=self.project_dir)
            except RefuseWrite as exc:
                # 文件读不懂：拒绝写，原文件没被动过，不需要回滚
                return {"result": RESULT_MANUAL, "message": f"无法保存 Key：{exc}",
                        "steps": steps, "backup_dir": manifest.directory}
            except OSError as exc:
                failed = fixer.rollback(manifest)
                rb = "已按备份恢复原配置。" if not failed else \
                     "尝试恢复原配置，但以下文件未能还原：{}。".format("、".join(failed))
                return {"result": RESULT_MANUAL,
                        "message": f"Key 保存失败：{exc}。{rb}原始备份仍在 {manifest.directory}。",
                        "steps": steps, "backup_dir": manifest.directory}
            steps.extend({"detail": s} for s in store_steps)
            # 让本进程立刻能用这个 Key（Codex/DeepSeek 读 AI_GATE_API_KEY）
            self.env["AI_GATE_API_KEY"] = key
            return self._after_write(client_id, steps, manifest, note,
                                     "Key 已保存，已连接 AI Gate。",
                                     "Key 已保存，但连接未成功。请检查其余原因。")

        # auto / choice：一次写一个逻辑字段
        if kind == "choice":
            fix_value = value if value is not None else issue.get("fix_value")
        else:
            fix_value = issue.get("fix_value")
        fix_field = issue.get("fix_field")
        if not fix_field or fix_value is None or str(fix_value) == "":
            return {"result": RESULT_MANUAL, "message": "请选择要应用的值。",
                    "steps": steps, "backup_dir": None}
        # 鉴权类字段的值先去掉首尾空白再写。这个载荷是"原始 Key"，而它带多余空白
        # 正是 auth-whitespace 这条要修的病；不 strip 就直接写，等于把「Key 末尾有
        # 换行」这个病原样搬进新文件（DeepSeek 的 YAML 写出器甚至表示不了换行）。
        # input 分支一直是 strip 过的，这里跟它对齐。
        if str(fix_field) in (FIELD_AUTH, FIELD_EXTRA_HEADER):
            stripped = str(fix_value).strip()
            if not stripped:
                return {"result": RESULT_MANUAL, "message": "这个值去掉空白后是空的，请重新输入。",
                        "steps": steps, "backup_dir": None}
            fix_value = stripped
        changes = {str(fix_field): str(fix_value)}
        manifest = fixer.backup_files(adapter.writable_paths(cfg), home=self.home)
        self._backups[client_id] = manifest
        steps.append({"detail": f"已备份原配置到 {manifest.directory}"})
        try:
            applied = fixer.apply_fixes(adapter, cfg, changes, env=self.env,
                                        home=self.home, project_dir=self.project_dir)
        except RefuseWrite as exc:
            return {"result": RESULT_MANUAL, "message": f"没有修改配置：{exc}",
                    "steps": steps, "backup_dir": manifest.directory}
        except OSError as exc:
            failed = fixer.rollback(manifest)
            rb = "已按备份恢复原配置。" if not failed else \
                 "尝试恢复原配置，但以下文件未能还原：{}。".format("、".join(failed))
            return {"result": RESULT_MANUAL,
                    "message": f"配置写入失败：{exc}。{rb}原始备份仍在 {manifest.directory}。",
                    "steps": steps, "backup_dir": manifest.directory}
        except Exception as exc:      # noqa: BLE001
            # 兜底：写盘链路里任何没预料到的异常，都当作"这次写入不可信"处理——
            # 先按备份恢复，再如实报出去。绝不让它逃到路由层变成一个只有英文的
            # 500：那样用户既不知道配置到底改没改，也拿不到备份路径。
            failed = fixer.rollback(manifest)
            rb = "已按备份恢复原配置。" if not failed else \
                 "尝试恢复原配置，但以下文件未能还原：{}。".format("、".join(failed))
            return {"result": RESULT_MANUAL,
                    "message": f"配置写入时出错（{type(exc).__name__}）：{exc}。{rb}"
                               f"原始备份仍在 {manifest.directory}。",
                    "steps": steps, "backup_dir": manifest.directory}
        steps.append({"detail": "；".join(applied) if applied else "没有需要写入的改动"})
        return self._after_write(client_id, steps, manifest, None,
                                 "已修复，已连接 AI Gate。",
                                 "已修改，但连接未成功。请检查其余原因。")

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
        # 契约层面就不可达的客户端（codex 之于 AI Gate）：配了地址和 Key 也连不上，
        # 生成配置、setx 环境变量都是无效劳动还污染环境——直接说明，什么都不写。
        probe_cfg = self.read_harness(client_id)
        if auth_requirement_unsatisfiable(probe_cfg, self.profile):
            issue = auth_incompatible_issue(probe_cfg, self.profile)
            client = ClientState(
                client_id=client_id, display_name=adapter.display_name,
                state=STATE_UNCONFIGURED, state_label=STATE_LABEL[STATE_UNCONFIGURED],
                binary_installed=adapter.detect_binary(env=self.env, home=self.home),
                config_present=probe_cfg.has_any_config,
                files=self._client_files(probe_cfg))
            return {"path": None,
                    "message": f"{adapter.display_name} 无法连接 AI Gate，未保存任何配置。",
                    "note": issue["detail"], "steps": [], "client": asdict(client)}
        # 已有配置就不重配：generate_minimal_config 是整文件覆盖，会把 permissions /
        # hooks / 旧设置一起冲掉且不留备份。只有「装了但一行没配」的机器才该走这里。
        if probe_cfg.has_any_config:
            client = self.assess_client(client_id)
            return {"path": None,
                    "message": f"{adapter.display_name} 已有一份配置，为避免覆盖它，这里不重新生成。",
                    "note": ("请在「检查」页按列出的原因逐项修复。若确实要推倒重来，"
                             "请先手动删除现有的配置文件，再回来点「配置」。"),
                    "steps": [], "client": asdict(client)}
        known = model_ids(self.profile)
        # 模型必须是用户真选了的、而且网关确实有这个型号。以前这里会在用户没选
        # （前端下拉为空）或选了个不存在的型号时，悄悄换成探活模型写进去——用户
        # 要 A、配置里是 B，提示还说「配置完成，已连接」。宁可什么都不写、让用户
        # 重选，也不替他决定一个他从没选过的模型（跟生成配置时留占位符同一口径）。
        pick = str(model or "").strip()
        if not pick:
            client = self.assess_client(client_id)
            return {"path": None, "message": "还没有选择模型，未写入任何配置。",
                    "note": "请在上面选一个要用的模型（清单来自网关）。若列表是空的，"
                            "先回「检查」页重新连一次网关。",
                    "steps": [], "client": asdict(client)}
        if pick not in known:
            client = self.assess_client(client_id)
            return {"path": None,
                    "message": f"「{pick}」不在网关支持的模型清单里，未写入任何配置。",
                    "note": "请从上面的列表里选一个网关确实支持的型号。",
                    "steps": [], "client": asdict(client)}

        # 先备份、再生成。顺序反了就没有可回滚的起点：generate_minimal_config 是
        # 整文件覆盖，备份如果发生在生成之后，备份里存的就是刚生成的占位文件。
        manifest = fixer.backup_files(adapter.writable_paths(probe_cfg), home=self.home)
        self._backups[client_id] = manifest
        steps = [{"detail": f"已备份原配置到 {manifest.directory}"}]
        try:
            path = adapter.generate_minimal_config(
                expected_base_url(self.profile, client_id), pick,
                env=self.env, home=self.home, project_dir=self.project_dir)
        except (RefuseWrite, OSError) as exc:
            return {"path": None, "message": f"没能生成配置：{exc}",
                    "note": None, "steps": steps,
                    "client": asdict(self.assess_client(client_id))}
        steps.append({"detail": f"已生成最小配置：{path}"})
        note = None

        cfg = self.read_harness(client_id)
        if api_key and str(api_key).strip():
            key = str(api_key).strip()
            try:
                store_steps, _changed, note = adapter.store_key(
                    cfg, key, env=self.env, home=self.home, project_dir=self.project_dir)
            except (RefuseWrite, OSError) as exc:
                # Key 没写进去 = 这份刚生成的配置是「有地址没 Key」的半成品。
                # 留着它只会让用户回到检查页看到一个连不上的客户端，还得自己
                # 去删——按备份回滚，让磁盘回到这次操作之前的样子，并如实说明。
                failed = fixer.rollback(manifest)
                rb = "已恢复到配置之前的样子。" if not failed else \
                     "尝试恢复，但以下文件未能还原：{}。".format("、".join(failed))
                return {"path": None,
                        "message": f"Key 没能保存（{exc}），已放弃这次配置。{rb}",
                        "note": "请确认这个客户端能正常读写它自己的配置文件，再重试。",
                        "steps": steps,
                        "client": asdict(self.assess_client(client_id))}
            steps.extend({"detail": s} for s in store_steps)
            self.env["AI_GATE_API_KEY"] = key

        client = self.assess_client(client_id)
        if client.state == STATE_CONNECTED:
            msg = "配置完成，已连接 AI Gate。"
        elif client.state == STATE_BLOCKED:
            msg = "配置已保存，但连接未成功。请检查原因。"
        else:
            msg = "配置已保存。"
        return {"path": path, "message": msg, "note": note,
                "steps": steps, "client": asdict(client)}


def redact_client(payload: Dict[str, Any]) -> Dict[str, Any]:
    """把一份序列化后的 ClientState 脱敏，返回**副本**（不改传入的那份）。

    `fix_value` 可能是原始 Key（补 Token 头、去掉首尾空白这两条修复的载荷都是它）。
    界面完全用不到这个字段——点「修复」时前端只发 {client_id, issue_id}，真正的值
    由服务端从自己那份 last_report 里取。所以任何一份要发给浏览器的客户端数据都先
    过这里，Key 在界面/接口返回里永远只以掩码（前 4 + 后 4）的形式出现。

    注意两件事：
    1. 这是"按字段名"脱敏，不是只给某一条路由用。会返回客户端数据的路由有
       四条（/api/check、/api/action、/api/configure、/api/recheck），只堵其中一条
       等于没堵——上一轮就是只在 report_to_dict 里做，另外三条照样漏。
    2. 必须返回副本、**不能就地 pop**：出网的这份 payload 和服务端 last_report 里
       持有的曾经是同一批 dict，就地抹掉会把服务端自己那份也一起抹了——用户点完
       一次修复，同一客户端剩下的「一键修复」按钮就全拿不到值、变成死按钮。
    """
    if not isinstance(payload, dict):
        return payload
    issues = payload.get("issues")
    if not isinstance(issues, list):
        return dict(payload)
    clean = dict(payload)
    clean["issues"] = [
        {k: v for k, v in it.items() if k != "fix_value"} if isinstance(it, dict) else it
        for it in issues
    ]
    return clean


def report_to_dict(report: Report) -> Dict[str, Any]:
    """报告要能直接序列化给界面（出网前统一脱敏，见 redact_client）。"""
    payload = asdict(report)
    clients = payload.get("clients")
    if isinstance(clients, list):
        payload["clients"] = [redact_client(c) for c in clients]
    return payload
