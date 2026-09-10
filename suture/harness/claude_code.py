"""Claude Code CLI 适配器。

配置载体：环境变量 + 组织托管配置（managed-settings.json，路径按系统固定）
        + 全局 ~/.claude/settings.json + 项目级 <项目>/.claude/settings.json
优先级：环境变量 > 组织托管配置 > 项目级配置 > 全局配置
托管配置是公司通过 IT/MDM 统一下发到机器上的，用户自己看不到也不会去改，
通常也没有写权限——这也是有些用户"从来不用自己填网关地址"的真实原因。
Suture 只读这个文件，绝不写入，也不会把它算进备份/回滚的范围。
（管理员还可能用 macOS 配置描述文件或 claude.ai 控制台下发同样的配置，
这两种不落地成本机文件，Suture 读不到；这种情况下要看 `claude` 里
`/status` 显示的 `Setting sources` 来确认实际生效值。）
鉴权：值直接存在配置里；填在 ANTHROPIC_API_KEY 走 x-api-key 头，
      填在 ANTHROPIC_AUTH_TOKEN 走 Authorization 头。
      另外 ANTHROPIC_CUSTOM_HEADERS（`Name: Value`，多头换行分隔）是平行的第三条通路：
      只要里面的头名是网关认的那几个（比如 Token），就跟主字段一起被当成鉴权来源，
      客户端会把两边都发出去，不能只认 API_KEY/AUTH_TOKEN 而把这条路当不存在。
网关地址：不带 /v1，客户端自己会在后面拼 /v1/messages。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    ExtraAuthHeader, FIELD_AUTH, FIELD_BASE_URL, FIELD_EXTRA_HEADER, FIELD_MODEL,
    FileState, HarnessAdapter, HarnessConfig, LayerValue, RefuseWrite, build_resolved,
    resolve_home, resolve_project_dir,
)

ENV_BASE_URL = "ANTHROPIC_BASE_URL"
ENV_API_KEY = "ANTHROPIC_API_KEY"
ENV_AUTH_TOKEN = "ANTHROPIC_AUTH_TOKEN"
ENV_MODEL = "ANTHROPIC_MODEL"
ENV_CUSTOM_HEADERS = "ANTHROPIC_CUSTOM_HEADERS"
RELEVANT_ENV = [ENV_BASE_URL, ENV_API_KEY, ENV_AUTH_TOKEN, ENV_MODEL, ENV_CUSTOM_HEADERS]


def _parse_custom_headers(raw: str) -> List[Tuple[str, str]]:
    """官方格式：`Name: Value`，多个头用换行分隔（已核实，v2.1.227+）。
    格式不对的行直接跳过，不强行猜——总比塞一个错的进去安全。"""
    out: List[Tuple[str, str]] = []
    for line in raw.split("\n"):
        line = line.strip("\r").strip()
        if not line or ":" not in line:
            continue
        name, _, value = line.partition(":")
        name, value = name.strip(), value.strip()
        if name and value:
            out.append((name, value))
    return out


def _global_path(home: str) -> str:
    return os.path.join(home, ".claude", "settings.json")


def _upsert_token_header(raw: Optional[str], token_value: str) -> str:
    """在 ANTHROPIC_CUSTOM_HEADERS 的原始文本里把 Token 行设成 token_value，
    其余头原样保留；原来没有 Token 行就补一行。AI Gate 只从 Token 头读 Key，
    所以给 Claude Code 存 Key 时也要往这个头里放一份。"""
    out, found = [], False
    for line in (raw or "").split("\n"):
        line = line.strip("\r")
        name, sep, _ = line.partition(":")
        if sep and name.strip().lower() == "token":
            out.append(f"Token: {token_value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"Token: {token_value}")
    return "\n".join(x for x in out if x.strip())


def _project_path(project_dir: str) -> str:
    return os.path.join(project_dir, ".claude", "settings.json")


ENV_MANAGED_PATH_OVERRIDE = "SUTURE_MANAGED_SETTINGS_PATH"     # 仅测试/特殊部署用


def _managed_path(env: Optional[Dict[str, str]] = None) -> str:
    """公司统一下发的托管配置——优先级压过用户和项目级的一切设置。用户自己
    不会去改这个文件（通常也没有写权限，是 IT/MDM 推送到机器上的），Suture
    只读不写。这也是有些用户"不用自己填 base_url"的真实原因：地址是从这里
    下发的，不在用户看得到、Suture 原来会去读的那两层里。
    路径按操作系统固定，来自官方文档，不是猜的；测试或者部署路径特殊时
    (比如 WSL) 可以用 SUTURE_MANAGED_SETTINGS_PATH 覆盖，不用改代码。"""
    env = env if env is not None else os.environ
    override = env.get(ENV_MANAGED_PATH_OVERRIDE)
    if override:
        return override
    if sys.platform == "win32":
        return r"C:\Program Files\ClaudeCode\managed-settings.json"
    if sys.platform == "darwin":
        return "/Library/Application Support/ClaudeCode/managed-settings.json"
    return "/etc/claude-code/managed-settings.json"


def _read_json_file(layer: str, path: str, known_keys: List[str]) -> FileState:
    st = FileState(layer=layer, path=path, exists=os.path.exists(path))
    if not st.exists:
        return st
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        st.data = json.loads(raw)
        if not isinstance(st.data, dict):
            st.parse_ok = False
            st.parse_error = "配置文件的最外层应该是一个对象（大括号包起来的结构）"
            st.data = {}
    except json.JSONDecodeError as exc:
        st.parse_ok = False
        st.parse_error = (
            f"JSON 格式错误：第 {exc.lineno} 行第 {exc.colno} 列 {exc.msg}。"
            "格式错误会导致整个文件解析失败，这一层的配置全部不生效，不只是出错的那一处。"
        )
        st.data = {}
    except UnicodeDecodeError:
        # 用记事本一类编辑器存成 ANSI/GBK 就会这样。UnicodeDecodeError 不是
        # OSError 的子类，漏掉它会让整轮检查直接抛出去、界面上只显示「检查失败」。
        st.parse_ok = False
        st.parse_error = (
            "文件不是 UTF-8 编码（多半是被编辑器存成了 ANSI/GBK）。"
            "请用编辑器把这份文件另存为 UTF-8 编码，再重新检查。"
        )
        st.data = {}
    except OSError as exc:
        st.parse_ok = False
        st.parse_error = f"读取失败：{exc}"
    if st.parse_ok and known_keys:
        st.unknown_keys = [k for k in st.data.keys() if k not in known_keys]
    return st


class ClaudeCodeAdapter(HarnessAdapter):
    harness_id = "claude_code"
    display_name = "Claude Code CLI"
    config_format = "json"
    binary_name = "claude"
    supports_extra_auth_header = True    # 能通过 env 的 ANTHROPIC_CUSTOM_HEADERS 补 Token 头
    can_send_custom_request_headers = True   # ANTHROPIC_CUSTOM_HEADERS 本身就是加自定义头的位置

    def detect(self, env=None, home=None, project_dir=None) -> bool:
        env = env if env is not None else os.environ
        home = resolve_home(home, env)
        project_dir = resolve_project_dir(project_dir)
        if any(env.get(k) for k in RELEVANT_ENV):
            return True
        if os.path.isdir(os.path.join(home, ".claude")):
            return True
        if os.path.exists(_managed_path(env)):
            return True
        return os.path.exists(_project_path(project_dir))

    def read(self, env=None, home=None, project_dir=None, known_keys=None,
              accepted_headers=None) -> HarnessConfig:
        env = env if env is not None else os.environ
        home = resolve_home(home, env)
        project_dir = resolve_project_dir(project_dir)
        known_keys = known_keys or []
        accepted_headers = accepted_headers or []
        accepted_lower = {h.lower() for h in accepted_headers}

        m = _read_json_file("组织托管配置（IT 统一下发，只读）", _managed_path(env), known_keys)
        g = _read_json_file("全局配置", _global_path(home), known_keys)
        p = _read_json_file("项目级配置", _project_path(project_dir), known_keys)
        cfg = HarnessConfig(harness_id=self.harness_id, display_name=self.display_name,
                            files=[m, g, p])

        def env_block(fs: FileState) -> Dict[str, Any]:
            block = fs.data.get("env")
            return block if isinstance(block, dict) else {}

        def layers_for(var: str) -> List[LayerValue]:
            # 优先级从高到低：环境变量 > 组织托管配置 > 项目级 > 全局。
            # 托管配置压过用户和项目级的一切设置，这是官方文档写明的规则，不是猜的；
            # 它比全局/项目级两层都高，但一个真实存在的环境变量还是能盖过它。
            out = []
            if env.get(var):
                out.append(LayerValue("环境变量", "", str(env[var])))
            for fs in (m, p, g):
                v = env_block(fs).get(var)
                if v:
                    out.append(LayerValue(fs.layer, fs.path, str(v), managed=(fs is m)))
            return out

        cfg.fields[FIELD_BASE_URL] = build_resolved(FIELD_BASE_URL, layers_for(ENV_BASE_URL))
        cfg.fields[FIELD_MODEL] = build_resolved(FIELD_MODEL, layers_for(ENV_MODEL))

        api_key = build_resolved("api_key", layers_for(ENV_API_KEY))
        auth_token = build_resolved("auth_token", layers_for(ENV_AUTH_TOKEN))

        # 客户端实际发哪个请求头，取决于用户设置的是哪个变量，
        # 而不是网关“希望”收到哪个头。两个都设置时 API_KEY 优先。
        if api_key.is_set:
            auth = api_key
            cfg.auth_header = "x-api-key"
        elif auth_token.is_set:
            auth = auth_token
            cfg.auth_header = "Authorization"
        else:
            auth = api_key
            cfg.auth_header = "x-api-key"
        auth.key = FIELD_AUTH
        cfg.fields[FIELD_AUTH] = auth

        if api_key.is_set and auth_token.is_set:
            cfg.auth_conflict = (
                f"{ENV_API_KEY} 和 {ENV_AUTH_TOKEN} 两处都填了鉴权信息，"
                f"实际生效的是 {ENV_API_KEY}（走 x-api-key 请求头）。"
            )

        # ANTHROPIC_CUSTOM_HEADERS 是跟 API_KEY/AUTH_TOKEN 平行的另一条鉴权通路：
        # 客户端会把这里面的头原样跟着请求一起发出去，不会因为设置了它就不发
        # x-api-key/Authorization。只挑出网关认的那几个头名，避免把无关的自定义头
        # 误判成鉴权信息。
        custom_headers = build_resolved("custom_headers", layers_for(ENV_CUSTOM_HEADERS))
        # 记下这个变量实际来自哪一层：判定层要据此判断「往 settings.json 补 Token 头
        # 到底有没有用」——来自系统环境变量时，补在文件里不会生效。
        cfg.custom_headers_source_layer = custom_headers.source_layer
        if custom_headers.is_set:
            for name, value in _parse_custom_headers(custom_headers.value):
                if name.lower() in accepted_lower:
                    cfg.extra_auth_headers.append(ExtraAuthHeader(
                        header=name, value=value,
                        source=f"{ENV_CUSTOM_HEADERS}（{custom_headers.source_layer}）"))

        return cfg

    def writable_paths(self, cfg: HarnessConfig) -> List[str]:
        # 托管配置只读，不备份也不回滚它——那个文件不属于 Suture 能写的范围，
        # 备份/回滚流程如果把它也算进去，权限不够时会白白报一个失败。
        return [fs.path for fs in cfg.files if fs.layer != "组织托管配置（IT 统一下发，只读）"]

    def _target_file(self, cfg: HarnessConfig) -> FileState:
        """写到实际生效的那一层。按已确认的使用规范，项目级不应该覆盖全局，
        所以统一写回全局配置；只有当项目级已经存在配置时才写项目级，
        避免写了一层不生效的。"""
        project = next((f for f in cfg.files if f.layer == "项目级配置"), None)
        if project is not None and project.exists and project.parse_ok:
            return project
        return next(f for f in cfg.files if f.layer == "全局配置")

    def unparseable_write_target(self, cfg: HarnessConfig) -> Optional[str]:
        target = self._target_file(cfg)
        if target.exists and not target.parse_ok:
            return target.path
        return None

    def apply(self, cfg: HarnessConfig, changes: Dict[str, str],
              env=None, home=None, project_dir=None) -> List[str]:
        target = self._target_file(cfg)
        if target.exists and not target.parse_ok:
            # 解析失败的文件 data 是空的，按它起草再整文件写回 = 用一份只含本次
            # 改动的文件覆盖原文件，同文件里的 permissions/hooks 会被静默抹掉。
            raise RefuseWrite(
                f"{target.path} 读不懂（{target.parse_error}）"
                "为避免把这份文件里其它配置一起覆盖掉，这里不自动修改。")
        data = dict(target.data)
        block = dict(data.get("env") or {}) if isinstance(data.get("env"), dict) else {}

        described: List[str] = []
        for logical, value in changes.items():
            if logical == FIELD_BASE_URL:
                block[ENV_BASE_URL] = value
                described.append(f"网关地址 → {value}（写入{target.layer}）")
            elif logical == FIELD_MODEL:
                block[ENV_MODEL] = value
                described.append(f"模型名称 → {value}（写入{target.layer}）")
            elif logical == FIELD_AUTH:
                block[ENV_API_KEY] = value
                block.pop(ENV_AUTH_TOKEN, None)
                block[ENV_CUSTOM_HEADERS] = _upsert_token_header(block.get(ENV_CUSTOM_HEADERS), value)
                described.append(f"鉴权信息 → 统一填到 {ENV_API_KEY}，并在 {ENV_CUSTOM_HEADERS} 补上 Token 头"
                                 f"（AI Gate 只从 Token 头读 Key；写入{target.layer}）")
            elif logical == FIELD_EXTRA_HEADER:
                block[ENV_CUSTOM_HEADERS] = _upsert_token_header(block.get(ENV_CUSTOM_HEADERS), value)
                described.append(f"补上网关要求的 Token 请求头 → 在 {ENV_CUSTOM_HEADERS} 里把 Token 设成"
                                 f"这个 Key（写入{target.layer}）")

        data["env"] = block
        os.makedirs(os.path.dirname(target.path), exist_ok=True)
        with open(target.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        try:
            # 这份文件现在可能带着 Key 明文（env 块 / ANTHROPIC_CUSTOM_HEADERS），
            # 收紧权限，跟 DeepSeek 的凭据文件口径一致。
            os.chmod(target.path, 0o600)
        except OSError:
            pass
        return described

    def self_test_command(self) -> Optional[List[str]]:
        """`claude -p "..."` 是官方的非交互（print）模式：跑完就退出，不进交互界面。
        故意不带 --model：整个自证的意义就在于让客户端用它自己那一套解析结果，
        Suture 不往里塞任何东西。"""
        return ["claude", "-p", "hi"]

    def install_command(self, env=None) -> Optional[List[str]]:
        return ["npm", "install", "-g", "@anthropic-ai/claude-code"]

    def install_guide(self, env=None) -> str:
        return ("Claude Code 官方通过 npm 分发：`npm install -g @anthropic-ai/claude-code`。"
                "注意 Windows 原生命令行（cmd/PowerShell）可能不是官方主推的形态，"
                "如果装完还是找不到 claude 命令，多半是在 WSL 或其它终端环境里用的，"
                "请在真正使用它的那个环境里安装，或直接用官方安装器。")

    def key_store_guide(self, env=None) -> str:
        return "Claude Code 会读环境变量 ANTHROPIC_API_KEY（或配置里的同名 env 项）。Suture 会把 Key 写进它的 settings.json。"

    def store_key(self, cfg: HarnessConfig, key: str,
                  env=None, home=None, project_dir=None) -> Tuple[List[str], List[str], str]:
        """Claude Code 的 Key 走配置文件的 env 块（写进实际生效的那一层），
        由引擎先备份再写入，可回滚。"""
        target = self._target_file(cfg)
        steps = self.apply(cfg, {FIELD_AUTH: key}, env=env, home=home, project_dir=project_dir)
        return (steps, [target.path],
                "改的是配置文件，正在运行的 Claude Code 需要重新启动才会读到新的 Key。")

    def generate_minimal_config(self, base_url: str, model: str,
                                env=None, home=None, project_dir=None) -> str:
        env = env if env is not None else os.environ
        home = resolve_home(home, env)
        path = _global_path(home)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "env": {
                ENV_BASE_URL: base_url,
                ENV_API_KEY: "请替换为网关后台生成的 Key",
                ENV_MODEL: model,
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return path
