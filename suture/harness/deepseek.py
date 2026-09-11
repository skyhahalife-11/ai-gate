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
    ExtraAuthHeader, FIELD_AUTH, FIELD_AUTH_ENV, FIELD_BASE_URL, FIELD_EXTRA_HEADER,
    FIELD_MODEL, FileState, HarnessAdapter, HarnessConfig, LayerValue, RefuseWrite,
    build_resolved, resolve_home, resolve_project_dir, write_bytes_atomic,
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
    except UnicodeDecodeError:
        # 记事本一类编辑器存成 ANSI/GBK 就会这样；UnicodeDecodeError 不是 OSError
        # 的子类，漏掉它整轮检查会直接抛出去。
        st.parse_ok = False
        st.parse_error = (
            "文件不是 UTF-8 编码（多半是被编辑器存成了 ANSI/GBK）。"
            "请用编辑器把这份文件另存为 UTF-8 编码，再重新检查。"
        )
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
    except (OSError, UnicodeDecodeError):
        return None
    return None


class DeepSeekHarnessAdapter(HarnessAdapter):
    harness_id = "deepseek"
    display_name = "DeepSeek Harness"
    config_format = "yaml"
    binary_name = None        # dsh CLI 的可执行名尚未确认，先不做二进制探测
    supports_extra_auth_header = True    # 能往路由 headers.Token 里补 Key
    can_send_custom_request_headers = True   # 路由的 headers 块就是加自定义头的位置

    def detect_binary(self, env=None, home=None) -> bool:
        """dsh CLI 的可执行名不在 PATH、也没法用 shutil.which 可靠找到（实测本机
        ~/.dsh 在但 PATH 里没有 dsh 命令）。装了 DeepSeek Harness 的可靠痕迹是
        harness home（默认 ~/.dsh）目录存在——跟 detect() 用同一套依据。

        不能沿用 base 的实现：binary_name 留空时 base 会直接判 False，安装页的
        「已安装/未安装」徽标永远显示「未安装」，跟检查页（靠配置痕迹判）结论打架。"""
        env = env if env is not None else os.environ
        flag = env.get("SUTURE_BINARY_OVERRIDE_DEEPSEEK")
        if flag is not None:
            return flag.strip().lower() in ("1", "true", "yes", "on")
        home = resolve_home(home, env)
        return os.path.isdir(harness_home(env, home))

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

        # 网关文档给的另一种写法：把 Token 直接塞在路由的 headers 里。
        # 真实客户端会把这个头原样跟主鉴权一起发出去——所以无论 apiKeyEnv/凭据
        # 有没有解析到 Key，只要 headers 里有 Token，它就要作为「平行额外请求头」
        # 被 Suture 一起带上（这正是 AI Gate 唯一采信的那个头）；只有整份配置
        # 都没填主鉴权时，才把 headers.Token 当主鉴权兜底。
        for fs, rc in ordered:
            headers = rc.get("headers")
            if isinstance(headers, dict):
                token = headers.get("Token")
                if token and str(token).strip():
                    token = str(token).strip()
                    if not cfg.fields[FIELD_AUTH].is_set:
                        cfg.fields[FIELD_AUTH] = build_resolved(
                            FIELD_AUTH, [LayerValue(f"{fs.layer} 的 headers.Token", fs.path, token)])
                        cfg.auth_header = "Token"
                        cfg.auth_is_indirect = False
                        cfg.auth_env_resolved = True
                    if not any(e.header == "Token" for e in cfg.extra_auth_headers):
                        cfg.extra_auth_headers.append(ExtraAuthHeader(
                            header="Token", value=token,
                            source=f"{fs.layer} 的 headers.Token"))
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

    def install_command(self, env=None) -> Optional[List[str]]:
        # DeepSeek Harness（dsh）的官方安装命令还没确认到可自动执行的程度，
        # 先返回 None——界面只给引导，避免给用户跑一条没验证过的命令。
        return None

    def install_guide(self, env=None) -> str:
        return ("DeepSeek Harness（dsh）的官方安装命令还没确认到能安全自动执行的程度，"
                "Suture 不会替你装。请先从官方渠道拿到安装包并安装；装好后切到「检查」页"
                "点一次「开始检查」（或刷新本页），工具就会识别到它已安装。接 AI Gate 需要 "
                "@deepseek-ai/dsh-llm-pi-ai 插件，路由写在 ~/.dsh 的 settings.yaml 里。")

    def key_store_guide(self, env=None) -> str:
        return ("DeepSeek Harness 的 Key 存在 ~/.dsh/.credentials.yaml 的 refs 段里，"
                "配置中的 apiKeyEnv 指向它的变量名（AI_GATE_API_KEY）。")

    def store_key(self, cfg: HarnessConfig, key: str,
                  env=None, home=None, project_dir=None) -> Tuple[List[str], List[str], str]:
        """DeepSeek：Key 本体写进 .credentials.yaml 的 refs.AI_GATE_API_KEY（权限 600），
        并确保用户层 settings.yaml 的 apiKeyEnv 指向这个变量名。"""
        env = env if env is not None else os.environ
        home = resolve_home(home, env)
        hh = harness_home(env, home)
        creds_path = os.path.join(hh, ".credentials.yaml")
        # 这份文件里可能还存着别的厂商的凭据（refs 段是一个共用表）。读不懂它却
        # 照写回去 = 用一份只含本次改动的文件覆盖整份凭据，别人的 Key 会被静默
        # 抹掉，而且还要报「已保存」。跟 settings.yaml 一个口径：读不懂就不写。
        if os.path.exists(creds_path):
            try:
                with open(creds_path, "r", encoding="utf-8") as f:
                    parsed = yaml.parse(f.read())
            except (yaml.MiniYamlError, UnicodeDecodeError, OSError) as exc:
                raise RefuseWrite(
                    f"{creds_path} 读不懂（{exc}）"
                    "为避免把这份文件里其它厂商的凭据一起覆盖掉，这里不自动修改。"
                    "请先把这份文件修好（或另存为 UTF-8），再重新保存 Key。")
            if not isinstance(parsed, dict):
                raise RefuseWrite(
                    f"{creds_path} 的内容不是一份键值配置（顶层是个"
                    f"{type(parsed).__name__}），为避免覆盖掉里面的其它内容，这里不自动修改。")
            if "refs" in parsed and not isinstance(parsed.get("refs"), dict):
                # 原来 refs 不是映射（比如写成了列表）：直接写回同样会丢内容。
                raise RefuseWrite(
                    f"{creds_path} 里的 refs 段不是键值形式，为避免把已有内容覆盖掉，"
                    "这里不自动修改。请先把 refs 改回「名字: 值」的形式再重试。")
            creds = dict(parsed)
        else:
            creds = {"version": 1, "refs": {}}
        refs = dict(creds.get("refs") or {})
        refs["AI_GATE_API_KEY"] = key
        creds["refs"] = refs
        try:
            body = yaml.dump(creds).encode("utf-8")
        except yaml.MiniYamlError as exc:
            # 写出器表达不了的值（比如 Key 里带了换行）。此时还没碰过文件。
            raise RefuseWrite(f"无法写入凭据文件：{exc}") from exc
        except UnicodeEncodeError as exc:
            raise RefuseWrite(f"无法写入凭据文件：内容编码不成 UTF-8（{exc}）。") from exc
        os.makedirs(hh, exist_ok=True)
        write_bytes_atomic(creds_path, body)
        try:
            os.chmod(creds_path, 0o600)
        except OSError:
            pass

        steps = self.apply(cfg, {FIELD_AUTH: key}, env=env, home=home, project_dir=project_dir)
        settings_path = os.path.join(hh, "settings.yaml")
        return (steps + [f"Key 已存入凭据文件 {creds_path}（权限仅本用户可读）。"],
                [creds_path, settings_path],
                "改完需重开 DeepSeek Harness 才会读到新的 Key。")

    def unparseable_write_target(self, cfg: HarnessConfig,
                                 fields: Optional[set] = None) -> Optional[str]:
        """存 Key 会写两个文件：用户层 settings.yaml 和 .credentials.yaml（后者不在
        cfg.files 的配置层语义里，但 read() 已经把它一起读过了）。

        凭据文件只跟鉴权有关——base_url / model 只写 settings.yaml，压根不碰它。
        所以只有这次要写 auth 类字段时，才把凭据文件的坏状态算进来；否则「一份坏
        凭据文件」会把 base_url / model 的按钮也一起封掉（判定层说写不了，实际
        apply 完全能成功，用户白丢一个能用的按钮）。"""
        targets = ["用户层 settings.yaml"]
        want = fields if fields is not None else {FIELD_AUTH, FIELD_EXTRA_HEADER}
        if want & {FIELD_AUTH, FIELD_EXTRA_HEADER}:
            targets.append("凭据文件 .credentials.yaml")
        for layer in targets:
            fs = next((f for f in cfg.files if f.layer == layer), None)
            if fs is not None and fs.exists and not fs.parse_ok:
                return fs.path
        return None

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
        if user_file is not None and user_file.exists:
            if not user_file.parse_ok:
                # 读不懂就写 = 用一份只含本次改动的文件覆盖整份 settings.yaml，
                # 同文件里别的插件配置和模型清单会被静默清掉。
                raise RefuseWrite(
                    f"{path} 读不懂（{user_file.parse_error}）"
                    "为避免把这份文件里其它配置一起覆盖掉，这里不自动修改。")
            data = dict(user_file.data)

        # 这几层在文件里可能被写成了别的形态（比如 llm-pi-ai 直接写成一个字符串、
        # providers 写成列表）。dict(...) 遇到那种会抛 ValueError，而这个异常谁也
        # 接不住——服务端会变成 500、把 Python 原文甩给用户。读不懂就别写，跟上面
        # 的 parse_ok 一个口径，只是这里的「读不懂」是结构层面的。
        section = data.get(PLUGIN_SECTION)
        if section is not None and not isinstance(section, dict):
            raise RefuseWrite(
                f"{path} 里的 {PLUGIN_SECTION} 不是一个键值分块"
                f"（是{type(section).__name__}），为避免写坏，这里不自动修改。")
        providers = (section or {}).get("providers")
        if providers is not None and not isinstance(providers, dict):
            raise RefuseWrite(
                f"{path} 里的 {PLUGIN_SECTION}.providers 不是键值形式，为避免写坏，"
                "这里不自动修改。")
        rc = (providers or {}).get(route)
        if rc is not None and not isinstance(rc, dict):
            raise RefuseWrite(
                f"{path} 里的 providers.{route} 不是键值形式，为避免写坏，这里不自动修改。")
        section = dict(section or {})
        providers = dict(providers or {})
        rc = dict(rc or {})

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
                headers = dict(rc.get("headers") or {})
                headers["Token"] = value
                rc["headers"] = headers
                described.append(
                    f"鉴权 → apiKeyEnv 指向 AI_GATE_API_KEY，并把 Key 写进 providers.{route} 的 "
                    "headers.Token（AI Gate 只从 Token 头读 Key；Key 同时存一份在 .credentials.yaml）")
            elif logical == FIELD_AUTH_ENV:
                # 只补「去哪个变量取 Key」这一个引用。载荷是变量**名字**，不是 Key，
                # 所以绝不能往 headers.Token 里写——那样客户端会拿字面量
                # "AI_GATE_API_KEY" 当 Token 发出去（实测过）。
                rc["apiKeyEnv"] = value
                described.append(
                    f"鉴权引用 → apiKeyEnv 指向 {value}（写入用户层 settings.yaml 的 "
                    f"providers.{route}）；Key 本身不写进配置文件，由环境变量或 "
                    ".credentials.yaml 提供")
            elif logical == FIELD_EXTRA_HEADER:
                headers = dict(rc.get("headers") or {})
                headers["Token"] = value
                rc["headers"] = headers
                described.append(
                    f"补上网关要求的 Token 请求头 → 把 Key 写进 providers.{route} 的 headers.Token")

        providers[route] = rc
        section["providers"] = providers
        data[PLUGIN_SECTION] = section

        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            # 先把整份内容序列化好再落盘。写出器可能拒绝某个值（比如带换行的
            # Key）；这一步放在写文件**之前**，失败时用户的 settings.yaml 一个
            # 字节都不会动。转成 RefuseWrite 是为了让引擎按「拒绝写入 + 说明原因」
            # 处理，而不是抛一个谁也接不住的异常、留个半截文件。
            body = yaml.dump(data).encode("utf-8")
        except yaml.MiniYamlError as exc:
            raise RefuseWrite(f"无法写入 {path}：{exc}") from exc
        except UnicodeEncodeError as exc:
            raise RefuseWrite(f"无法写入 {path}：内容编码不成 UTF-8（{exc}）。") from exc
        write_bytes_atomic(path, body)
        try:
            # 这份文件现在可能带着 headers.Token 里的 Key 明文，收紧权限
            os.chmod(path, 0o600)
        except OSError:
            pass
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
        # 先把内容序列化好再落盘。写成 f.write(yaml.dump(data)) 更危险：dump 在
        # write 的实参里求值，文件已经被 open(..., "w") 截断了才轮到它抛错。
        write_bytes_atomic(path, yaml.dump(data).encode("utf-8"))
        creds = os.path.join(hh, ".credentials.yaml")
        if not os.path.exists(creds):
            write_bytes_atomic(
                creds,
                yaml.dump({"version": 1,
                           "refs": {"AI_GATE_API_KEY": "请替换为网关后台生成的 Key"}}
                          ).encode("utf-8"))
            try:
                os.chmod(creds, 0o600)
            except OSError:
                pass
        return path
