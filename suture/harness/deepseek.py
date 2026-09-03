"""DeepSeek Harness 适配器。

配置载体（harness home 默认 ~/.dsh，可用 $DSH_HOME 改）：
  cordis.yml            组合基线，声明启用哪些插件及其默认配置，通常随项目分发
  settings.yaml         用户层，按插件分段覆盖基线
  .credentials.yaml     Key 本体，refs 段按环境变量名存值
覆盖方向与另外两个 harness 相反：用户层覆盖项目基线。
接自定义网关走 @deepseek-ai/dsh-llm-pi-ai 插件，路由声明在 providers.<路由名> 下。
鉴权是间接引用：apiKeyEnv 写的是变量名，值按「启动环境 > 凭据文件 > 项目 .env >
harness home 下的 .env」这条顺序解析。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from . import _minimal_yaml as yaml
from .base import (
    FIELD_AUTH, FIELD_BASE_URL, FIELD_MODEL, FileState, HarnessAdapter,
    HarnessConfig, LayerValue, build_resolved, resolve_home, resolve_project_dir,
)

PLUGIN_FULL = "@deepseek-ai/dsh-llm-pi-ai"
PLUGIN_SECTION = "llm-pi-ai"
DEFAULT_ROUTE = "tower-ai"


def harness_home(env, home: str) -> str:
    configured = (env.get("DSH_HOME") or "").strip()
    return configured if configured else os.path.join(home, ".dsh")


def _read_yaml(layer: str, path: str) -> FileState:
    st = FileState(layer=layer, path=path, exists=os.path.exists(path))
    if not st.exists:
        return st
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.parse(f.read())
        st.data = data if isinstance(data, dict) else {}
        if data is not None and not isinstance(data, dict):
            st.parse_ok = False
            st.parse_error = "配置文件的最外层应该是一组「键: 值」，而不是列表或单个值"
    except yaml.MiniYamlError as exc:
        st.parse_ok = False
        st.parse_error = f"YAML 无法安全解析：{exc}"
    except OSError as exc:
        st.parse_ok = False
        st.parse_error = f"读取失败：{exc}"
    return st


def _plugin_providers_from_cordis(data: Dict[str, Any]) -> Dict[str, Any]:
    entries = data.get("plugins") if isinstance(data.get("plugins"), list) else None
    if entries is None and isinstance(data, list):
        entries = data
    for entry in entries or []:
        if isinstance(entry, dict) and entry.get("name") == PLUGIN_FULL:
            cfg = entry.get("config")
            if isinstance(cfg, dict) and isinstance(cfg.get("providers"), dict):
                return cfg["providers"]
    return {}


def _plugin_providers_from_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    section = data.get(PLUGIN_SECTION)
    if isinstance(section, dict) and isinstance(section.get("providers"), dict):
        return section["providers"]
    return {}


def _pick_route(*provider_dicts: Dict[str, Any]) -> Optional[str]:
    for providers in provider_dicts:
        if DEFAULT_ROUTE in providers:
            return DEFAULT_ROUTE
    for providers in provider_dicts:
        if providers:
            return next(iter(providers))
    return None


def _dotenv_value(path: str, name: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == name:
                    return v.strip().strip("'").strip('"')
    except OSError:
        return None
    return None


class DeepSeekHarnessAdapter(HarnessAdapter):
    harness_id = "deepseek"
    display_name = "DeepSeek Harness"
    config_format = "yaml"

    def detect(self, env=None, home=None, project_dir=None) -> bool:
        env = env if env is not None else os.environ
        home = resolve_home(home, env)
        project_dir = resolve_project_dir(project_dir)
        if os.path.isdir(harness_home(env, home)):
            return True
        return os.path.exists(os.path.join(project_dir, "cordis.yml"))

    def read(self, env=None, home=None, project_dir=None, known_keys=None,
              accepted_headers=None) -> HarnessConfig:
        # accepted_headers 目前只有 Claude Code CLI 的 ANTHROPIC_CUSTOM_HEADERS 用得上，
        # 这里接收只是为了跟 Engine 统一调用签名。DeepSeek 自己的 headers.Token 是配置
        # 文件里的字段，已经在下面单独处理，不需要靠这个参数识别。
        env = env if env is not None else os.environ
        home = resolve_home(home, env)
        project_dir = resolve_project_dir(project_dir)
        hh = harness_home(env, home)

        cordis_path = os.path.join(project_dir, "cordis.yml")
        if not os.path.exists(cordis_path):
            cordis_path = os.path.join(hh, "cordis.yml")
        base = _read_yaml("组合基线 cordis.yml", cordis_path)
        user = _read_yaml("用户层 settings.yaml", os.path.join(hh, "settings.yaml"))
        creds = _read_yaml("凭据文件 .credentials.yaml", os.path.join(hh, ".credentials.yaml"))

        cfg = HarnessConfig(harness_id=self.harness_id, display_name=self.display_name,
                            files=[base, user, creds])

        base_providers = _plugin_providers_from_cordis(base.data) if base.parse_ok else {}
        user_providers = _plugin_providers_from_settings(user.data) if user.parse_ok else {}
        route = _pick_route(user_providers, base_providers)
        cfg._route = route

        def route_cfg(providers: Dict[str, Any]) -> Dict[str, Any]:
            v = providers.get(route) if route else None
            return v if isinstance(v, dict) else {}

        # 用户层覆盖基线层，与另外两个 harness 的方向相反
        ordered: List[Tuple[FileState, Dict[str, Any]]] = [
            (user, route_cfg(user_providers)),
            (base, route_cfg(base_providers)),
        ]

        def layers_for(key: str) -> List[LayerValue]:
            out = []
            for fs, rc in ordered:
                v = rc.get(key)
                if v:
                    out.append(LayerValue(fs.layer, fs.path, str(v)))
            return out

        cfg.fields[FIELD_BASE_URL] = build_resolved(FIELD_BASE_URL, layers_for("baseURL"))

        # 模型是一份注册清单，不是单个选中值：每一项都要校验
        models: List[str] = []
        for _, rc in ordered:
            entries = rc.get("models")
            if isinstance(entries, list):
                for item in entries:
                    mid = item.get("id") if isinstance(item, dict) else item
                    if mid and str(mid) not in models:
                        models.append(str(mid))
                break
        cfg.model_candidates = models
        cfg.fields[FIELD_MODEL] = build_resolved(
            FIELD_MODEL,
            [LayerValue(ordered[0][0].layer if user_providers else base.layer, "", models[0])] if models else [],
        )

        # 鉴权：间接引用 + 四层取值顺序
        env_name_field = build_resolved("apiKeyEnv", layers_for("apiKeyEnv"))
        cfg.auth_is_indirect = True
        cfg.auth_env_name = env_name_field.value
        cfg.auth_header = "Authorization"
        cfg._env_key_field = env_name_field

        secret, secret_source = None, ""
        if env_name_field.value:
            name = env_name_field.value
            if env.get(name):
                secret, secret_source = env[name], "启动时的环境变量"
            else:
                refs = creds.data.get("refs") if creds.parse_ok else None
                if isinstance(refs, dict) and refs.get(name):
                    secret, secret_source = str(refs[name]), "凭据文件 .credentials.yaml"
                else:
                    v = _dotenv_value(os.path.join(project_dir, ".env"), name)
                    if v:
                        secret, secret_source = v, "项目目录下的 .env"
                    else:
                        v = _dotenv_value(os.path.join(hh, ".env"), name)
                        if v:
                            secret, secret_source = v, "harness home 下的 .env"
        cfg.auth_env_resolved = bool(secret)
        cfg.fields[FIELD_AUTH] = build_resolved(
            FIELD_AUTH, [LayerValue(secret_source, "", secret)] if secret else [])

        # 也可能把 Token 直接塞在 headers 里，这是网关文档给的另一种写法
        for fs, rc in ordered:
            headers = rc.get("headers")
            if isinstance(headers, dict) and headers.get("Token"):
                if not cfg.fields[FIELD_AUTH].is_set:
                    cfg.fields[FIELD_AUTH] = build_resolved(
                        FIELD_AUTH, [LayerValue(f"{fs.layer} 的 headers.Token", fs.path, str(headers["Token"]))])
                    cfg.auth_header = "Token"
                    cfg.auth_is_indirect = False
                    cfg.auth_env_resolved = True
                break

        if route is None and any(f.exists for f in (base, user)):
            cfg.notes.append(
                f"没有在 {PLUGIN_FULL} 插件下找到任何 providers 路由，"
                "接自定义网关需要在这个插件下声明一条路由。")
        elif route:
            cfg.notes.append(f"当前检查的是 providers.{route} 这条路由。")
        return cfg

    def writable_paths(self, cfg: HarnessConfig) -> List[str]:
        return [fs.path for fs in cfg.files]

    def apply(self, cfg: HarnessConfig, changes: Dict[str, str],
              env=None, home=None, project_dir=None) -> List[str]:
        """写用户层 settings.yaml——它覆盖基线，是实际生效的那一层，
        而且基线通常随项目分发、不该由本机工具改动。"""
        env = env if env is not None else os.environ
        home = resolve_home(home, env)
        hh = harness_home(env, home)
        path = os.path.join(hh, "settings.yaml")
        route = getattr(cfg, "_route", None) or DEFAULT_ROUTE

        data: Dict[str, Any] = {}
        user_file = next((f for f in cfg.files if f.path == path), None)
        if user_file is not None and user_file.exists and user_file.parse_ok:
            data = dict(user_file.data)

        section = dict(data.get(PLUGIN_SECTION) or {})
        providers = dict(section.get("providers") or {})
        rc = dict(providers.get(route) or {})

        described: List[str] = []
        for logical, value in changes.items():
            if logical == FIELD_BASE_URL:
                rc["baseURL"] = value
                described.append(f"网关地址 → {value}（写入用户层 settings.yaml 的 providers.{route}）")
            elif logical == FIELD_MODEL:
                rc["models"] = [{"id": m} for m in value.split(",")]
                described.append(f"模型清单 → {value}（写入用户层 settings.yaml 的 providers.{route}）")
            elif logical == FIELD_AUTH:
                rc["apiKeyEnv"] = "AI_GATE_API_KEY"
                described.append(
                    f"鉴权引用 → apiKeyEnv 指向 AI_GATE_API_KEY（写入用户层 settings.yaml 的 providers.{route}）；"
                    "Key 本身存在 .credentials.yaml 或环境变量里，不写进这份配置")

        providers[route] = rc
        section["providers"] = providers
        data[PLUGIN_SECTION] = section

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml.dump(data))
        return described

    def generate_minimal_config(self, base_url: str, model: str,
                                env=None, home=None, project_dir=None) -> str:
        env = env if env is not None else os.environ
        home = resolve_home(home, env)
        hh = harness_home(env, home)
        path = os.path.join(hh, "settings.yaml")
        os.makedirs(hh, exist_ok=True)
        data = {
            PLUGIN_SECTION: {
                "providers": {
                    DEFAULT_ROUTE: {
                        "api": "anthropic-messages",
                        "baseURL": base_url,
                        "apiKeyEnv": "AI_GATE_API_KEY",
                        "models": [{"id": model}],
                    }
                }
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml.dump(data))
        creds = os.path.join(hh, ".credentials.yaml")
        if not os.path.exists(creds):
            with open(creds, "w", encoding="utf-8") as f:
                f.write(yaml.dump({"version": 1, "refs": {"AI_GATE_API_KEY": "请替换为网关后台生成的 Key"}}))
            try:
                os.chmod(creds, 0o600)
            except OSError:
                pass
        return path
