"""Codex CLI 适配器。

配置载体：用户级 ~/.codex/config.toml + 项目级 <项目>/.codex/config.toml（TOML）
优先级：项目级 > 用户级（真实链路还有 CLI 参数、profile、系统级，这里只处理
        文件这两层，其余层不由 Suture 负责）
鉴权：env_key 是间接引用，存的是环境变量名而不是 Key 本身。
网关地址：按 OpenAI 兼容协议的惯例，base_url 自带 /v1。
可信项目：项目级配置只在这个项目被 Codex 标记为可信时才生效，否则整层跳过。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    FIELD_AUTH, FIELD_BASE_URL, FIELD_MODEL, FileState, HarnessAdapter,
    HarnessConfig, LayerValue, RefuseWrite, build_resolved, resolve_home,
    resolve_project_dir,
)


def _user_path(home: str) -> str:
    return os.path.join(home, ".codex", "config.toml")


def _project_path(project_dir: str) -> str:
    return os.path.join(project_dir, ".codex", "config.toml")


def _read_toml(layer: str, path: str) -> FileState:
    st = FileState(layer=layer, path=path, exists=os.path.exists(path))
    if not st.exists:
        return st
    try:
        with open(path, "rb") as f:
            st.data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        st.parse_ok = False
        st.parse_error = (
            f"TOML 格式错误：{exc}。"
            "格式错误会导致整个文件解析失败，这一层的配置全部不生效。"
        )
    except UnicodeDecodeError:
        # tomllib 解不了非 UTF-8 的字节流时直接抛 UnicodeDecodeError，它不是
        # TOMLDecodeError、也不是 OSError 的子类，漏掉会让整轮检查抛出去。
        st.parse_ok = False
        st.parse_error = (
            "文件不是 UTF-8 编码（多半是被编辑器存成了 ANSI/GBK）。"
            "请用编辑器把这份文件另存为 UTF-8 编码，再重新检查。"
        )
    except OSError as exc:
        st.parse_ok = False
        st.parse_error = f"读取失败：{exc}"
    return st


def _project_is_trusted(user_file: FileState, project_dir: str) -> bool:
    """Codex 把可信项目记在用户级配置的 [projects."<路径>"] 里。
    没有这条记录就意味着项目级配置不会被采纳。"""
    projects = user_file.data.get("projects")
    if not isinstance(projects, dict):
        return False
    target = os.path.normpath(project_dir)
    for key, val in projects.items():
        if os.path.normpath(key) == target and isinstance(val, dict):
            return str(val.get("trust_level", "")).lower() == "trusted"
    return False


def _provider_table(data: Dict[str, Any], provider_id: Optional[str]) -> Dict[str, Any]:
    providers = data.get("model_providers")
    if not isinstance(providers, dict) or not provider_id:
        return {}
    tbl = providers.get(provider_id)
    return tbl if isinstance(tbl, dict) else {}


class CodexAdapter(HarnessAdapter):
    harness_id = "codex"
    display_name = "Codex CLI"
    config_format = "toml"
    binary_name = "codex"
    # Codex 没有自定义请求头机制（读配置时已注明），只会把 Key 放 Authorization: Bearer。
    # 对只认自定义 Token 头的网关（AI Gate）完全不可达 → can_send_custom_request_headers 保持
    # False 默认值，判定层据此把它的鉴权失败归成「外部处理」而不是引导填 Key/补头。

    def detect(self, env=None, home=None, project_dir=None) -> bool:
        env = env if env is not None else os.environ
        home = resolve_home(home, env)
        project_dir = resolve_project_dir(project_dir)
        codex_home = env.get("CODEX_HOME")
        if codex_home and os.path.isdir(codex_home):
            return True
        return os.path.isdir(os.path.join(home, ".codex")) or os.path.exists(_project_path(project_dir))

    def read(self, env=None, home=None, project_dir=None, known_keys=None,
              accepted_headers=None) -> HarnessConfig:
        # accepted_headers 目前只有 Claude Code CLI 的 ANTHROPIC_CUSTOM_HEADERS 用得上，
        # 这里接收只是为了跟 Engine 统一调用签名，Codex 自己没有平行的自定义头机制。
        env = env if env is not None else os.environ
        home = resolve_home(home, env)
        project_dir = resolve_project_dir(project_dir)
        codex_home = env.get("CODEX_HOME")
        user_file_path = os.path.join(codex_home, "config.toml") if codex_home else _user_path(home)

        u = _read_toml("用户级配置", user_file_path)
        p = _read_toml("项目级配置", _project_path(project_dir))

        cfg = HarnessConfig(harness_id=self.harness_id, display_name=self.display_name,
                            files=[u, p])

        # 可信项目机制：不可信时，项目级这一层整个不生效，不能拿它去判断和修复
        if p.exists:
            trusted = _project_is_trusted(u, project_dir)
            p.active = trusted
            if not trusted:
                p.inactive_reason = (
                    "该项目未被 Codex 标记为可信，项目级 .codex/config.toml 将被整体跳过，"
                    "实际生效的是用户级配置；修改此文件不会生效。"
                )

        active_files = [f for f in (p, u) if f.exists and f.parse_ok and f.active]

        def layers_for(getter) -> List[LayerValue]:
            out = []
            for fs in active_files:   # 已按项目级 > 用户级的顺序
                v = getter(fs.data)
                if v:
                    out.append(LayerValue(fs.layer, fs.path, str(v)))
            return out

        provider_id = None
        for fs in active_files:
            if fs.data.get("model_provider"):
                provider_id = str(fs.data["model_provider"])
                break

        cfg.fields[FIELD_BASE_URL] = build_resolved(
            FIELD_BASE_URL,
            layers_for(lambda d: _provider_table(d, provider_id).get("base_url")),
        )
        cfg.fields[FIELD_MODEL] = build_resolved(
            FIELD_MODEL, layers_for(lambda d: d.get("model")))

        # 鉴权是间接引用：配置里写的是环境变量名，真正的值在那个环境变量里
        env_key_field = build_resolved(
            "env_key", layers_for(lambda d: _provider_table(d, provider_id).get("env_key")))
        cfg.auth_is_indirect = True
        cfg.auth_env_name = env_key_field.value
        cfg.auth_header = "Authorization"      # Codex 走 OpenAI 兼容协议，用 Bearer

        secret = env.get(env_key_field.value) if env_key_field.value else None
        cfg.auth_env_resolved = bool(secret)
        auth = build_resolved(FIELD_AUTH,
                              [LayerValue(f"环境变量 {env_key_field.value}", "", secret)] if secret else [])
        cfg.fields[FIELD_AUTH] = auth
        cfg._env_key_field = env_key_field      # 供检测层使用
        cfg._provider_id = provider_id
        if provider_id is None and any(f.exists for f in (u, p)):
            cfg.notes.append("配置里没有指定 model_provider，无法确定该看哪个 provider 分块的地址和鉴权设置。")
        return cfg

    def writable_paths(self, cfg: HarnessConfig) -> List[str]:
        return [fs.path for fs in cfg.files]

    def _target_file(self, cfg: HarnessConfig) -> FileState:
        """只写实际会生效的那一层。项目级不可信时绝不写它——写了等于没改。"""
        project = next((f for f in cfg.files if f.layer == "项目级配置"), None)
        if project is not None and project.exists and project.parse_ok and project.active:
            return project
        return next(f for f in cfg.files if f.layer == "用户级配置")

    def unparseable_write_target(self, cfg: HarnessConfig) -> Optional[str]:
        """Codex 的 apply 是按文本定点替换的，但目标整份读不出来时同样不能写
        （apply 里会 RefuseWrite）。判定层据此不给按钮，别让用户点了才知道。
        只在目标层本来就不会生效时返回（那种情况下 apply 根本不写它）。"""
        target = self._target_file(cfg)
        if target.exists and not target.parse_ok:
            return target.path
        return None

    def apply(self, cfg: HarnessConfig, changes: Dict[str, str],
              env=None, home=None, project_dir=None) -> List[str]:
        target = self._target_file(cfg)
        provider_id = getattr(cfg, "_provider_id", None) or "ai-gate"
        text = ""
        if target.exists:
            try:
                with open(target.path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                # 这份文件的改动是按文本定点替换的（不动其它内容），但连读都读不出来
                # 就没法安全地改——如实报错，别把整份文件用别的编码写回去。
                raise RefuseWrite(
                    f"{target.path} 不是 UTF-8 编码（多半是被编辑器存成了 ANSI/GBK），"
                    "为避免把文件内容改坏，这里不自动修改。请先另存为 UTF-8 再试。")
            except OSError as exc:
                # 读不出来（权限、被占用、路径其实是个目录）就当成"文件是空的"继续写，
                # 等于把整份 config.toml 替换成只有本次改动的那一小段。跟编码读不出来
                # 一样处理：拒绝写，说清楚原因。
                raise RefuseWrite(
                    f"{target.path} 读取失败（{exc}），"
                    "为避免把文件里其它内容覆盖掉，这里不自动修改。")

        described: List[str] = []
        for logical, value in changes.items():
            if logical == FIELD_BASE_URL:
                text = _set_provider_key(text, provider_id, "base_url", value)
                described.append(f"网关地址 → {value}（写入{target.layer}）")
            elif logical == FIELD_MODEL:
                text = _set_top_level(text, "model", value)
                described.append(f"模型名称 → {value}（写入{target.layer}）")
            elif logical == FIELD_AUTH:
                # 鉴权是间接引用，Suture 不把 Key 明文写进 TOML，
                # 只保证 env_key 指向一个确定的变量名，值由用户在环境里设置。
                text = _set_provider_key(text, provider_id, "env_key", "AI_GATE_API_KEY")
                described.append(
                    f"鉴权引用 → env_key 指向 AI_GATE_API_KEY（写入{target.layer}）；"
                    "Key 本身需要设置成这个名字的环境变量，不写进配置文件"
                )

        os.makedirs(os.path.dirname(target.path), exist_ok=True)
        with open(target.path, "w", encoding="utf-8") as f:
            f.write(text)
        return described

    def self_test_command(self) -> Optional[List[str]]:
        """`codex exec` 是官方的非交互模式（已核实，developers.openai.com/codex/
        noninteractive）。加 --skip-git-repo-check 是因为这个自证可能在非 git 目录下
        跑（比如项目目录没初始化 git），不加的话会因为目录不是 git 仓库而失败，
        跟网关通不通没关系，会污染自证结果。默认是只读沙箱，不需要额外加权限。"""
        return ["codex", "exec", "--skip-git-repo-check", "hi"]

    def install_command(self, env=None) -> Optional[List[str]]:
        # OpenAI Codex CLI 官方 npm 包名（实现阶段已在官方文档核实）。
        return ["npm", "install", "-g", "@openai/codex"]

    def install_guide(self, env=None) -> str:
        return ("Codex CLI 官方通过 npm 分发：`npm install -g @openai/codex`。"
                "装完确认 `codex --version` 能跑；如果找不到命令，多半要重开终端或"
                "把 npm 全局 bin 目录加进 PATH。")

    def key_store_guide(self, env=None) -> str:
        return "Codex 的 Key 通过环境变量 AI_GATE_API_KEY 提供（配置里只写这个变量名）。"

    @staticmethod
    def _persist_user_env(name: str, value: str, env: Dict[str, str]) -> str:
        """把环境变量持久化。只在真实运行（env 就是 os.environ）时用 setx 写用户级，
        对之后新开的终端生效；测试/隔离 env 里不做（会把测试结果绑定到这台机器的
        环境上）。当前进程内是否生效由引擎负责。"""
        if env is os.environ and sys.platform == "win32":
            try:
                subprocess.run(["setx", name, value], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"已写入 Windows 用户环境变量 {name}（重开终端后永久生效）。"
            except OSError:
                pass
        return f"请把环境变量 {name} 设为这个值（当前会话已生效，用于本次验证）。"

    def store_key(self, cfg: HarnessConfig, key: str,
                  env=None, home=None, project_dir=None) -> Tuple[List[str], List[str], str]:
        """Codex：Key 不写进 TOML（那是环境变量名），这里负责
        1) 确保 provider 的 env_key 指向 AI_GATE_API_KEY（会写配置文件，可备份）；
        2) 把值持久化到用户环境变量。"""
        steps = self.apply(cfg, {FIELD_AUTH: key}, env=env, home=home, project_dir=project_dir)
        target = self._target_file(cfg)
        env = env if env is not None else os.environ
        note = self._persist_user_env("AI_GATE_API_KEY", key, env)
        return (steps, [target.path], note)

    def generate_minimal_config(self, base_url: str, model: str,
                                env=None, home=None, project_dir=None) -> str:
        env = env if env is not None else os.environ
        home = resolve_home(home, env)
        path = _user_path(home)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        content = (
            f'model = "{model}"\n'
            'model_provider = "ai-gate"\n'
            '\n'
            '[model_providers.ai-gate]\n'
            'name = "AI Gate"\n'
            f'base_url = "{base_url}"\n'
            'env_key = "AI_GATE_API_KEY"\n'
            '# Key 本身不写在这里：把它设置成名为 AI_GATE_API_KEY 的环境变量\n'
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path


# ---- TOML 的定点修改。tomllib 只能读不能写，这里按行改，保留注释和其余内容 ----

def _set_top_level(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf'^{re.escape(key)}\s*=.*$', re.MULTILINE)
    line = f'{key} = "{value}"'
    if pattern.search(_strip_tables(text)):
        # 只替换第一个表头之前的那一处，避免误改表内同名字段
        head, sep, tail = _split_at_first_table(text)
        head = pattern.sub(line, head, count=1)
        return head + sep + tail
    head, sep, tail = _split_at_first_table(text)
    head = (head.rstrip("\n") + "\n" if head.strip() else "") + line + "\n"
    return head + sep + tail


def _set_provider_key(text: str, provider_id: str, key: str, value: str) -> str:
    header = f"[model_providers.{provider_id}]"
    line = f'{key} = "{value}"'
    if header in text:
        start = text.index(header)
        rest = text[start + len(header):]
        m = re.search(r'^\[', rest, re.MULTILINE)
        end = start + len(header) + (m.start() if m else len(rest))
        block = text[start:end]
        pattern = re.compile(rf'^{re.escape(key)}\s*=.*$', re.MULTILINE)
        if pattern.search(block):
            block = pattern.sub(line, block, count=1)
        else:
            block = block.rstrip("\n") + "\n" + line + "\n"
        return text[:start] + block + text[end:]
    sep = "" if (not text or text.endswith("\n")) else "\n"
    return text + sep + f"\n{header}\n{line}\n"


def _split_at_first_table(text: str):
    m = re.search(r'^\[', text, re.MULTILINE)
    if not m:
        return text, "", ""
    return text[:m.start()], "", text[m.start():]


def _strip_tables(text: str) -> str:
    head, _, _ = _split_at_first_table(text)
    return head
