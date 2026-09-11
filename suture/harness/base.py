"""适配器接口与三个 harness 共用的数据结构。

检测层只认这里定义的结构，不认任何一个 harness 的原始字段名——这样
checks.py 才能保持与 harness 无关，不必为每个 harness 写一套判断。
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 三个逻辑字段。各 harness 把自己的原始字段名映射到这三个上。
FIELD_BASE_URL = "base_url"
FIELD_AUTH = "auth"
FIELD_MODEL = "model"
# 第四个：网关要求把 Key 放进某个请求头、而客户端没这么做时，把 Key 补进那个头
# （deepseek → providers.<route>.headers.Token；claude_code → env 的
#  ANTHROPIC_CUSTOM_HEADERS 里补一行 Token）。只由网关 required_header 判定触发。
FIELD_EXTRA_HEADER = "extra_auth_header"
# 第五个：间接引用式鉴权（deepseek 的 apiKeyEnv / codex 的 env_key）里，配置**连变量名
# 都没写**。这时要修的只是那个引用本身——写进去的是变量**名字**，不是 Key 的值。
# 必须跟 FIELD_AUTH 分开：FIELD_AUTH 的载荷是真正的 Key 明文（存 Key / 补头都写它），
# 而这条的载荷是 "AI_GATE_API_KEY" 这种名字。混用会把变量名当成凭据写进请求头，
# 客户端就会拿着字面量 "AI_GATE_API_KEY" 去当 Token 发（实测过）。
FIELD_AUTH_ENV = "auth_env"
LOGICAL_FIELDS = [FIELD_BASE_URL, FIELD_AUTH, FIELD_MODEL, FIELD_EXTRA_HEADER]


class RefuseWrite(Exception):
    """目标文件读不懂（语法错误）时，适配器拒绝写入。

    这类文件解析失败后 data 是空的，按它起草再整文件写回，就等于用一份只含
    本次改动的文件覆盖掉原文件——同一份 settings.json 里的 permissions、hooks、
    mcpServers 会被静默抹掉。宁可写不了、如实告诉用户先修好格式，也不能悄悄清空
    别人机器上的其它配置。"""


@dataclass
class LayerValue:
    """某一层配置里这个字段的取值。"""
    layer: str          # 层的名字，比如 "环境变量" / "项目级配置"
    path: str           # 来源文件路径；环境变量层为空字符串
    value: str
    effective: bool = False   # 是不是实际生效的那一层
    managed: bool = False     # 是不是公司统一下发、用户自己改不了的那一层（跟哪个 harness 无关）


@dataclass
class FileState:
    """一个配置文件的读取结果。"""
    layer: str
    path: str
    exists: bool
    parse_ok: bool = True
    parse_error: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    unknown_keys: List[str] = field(default_factory=list)
    active: bool = True       # 这一层是否真的会被 harness 采纳（Codex 的可信项目机制会让它变成 False）
    inactive_reason: str = ""


@dataclass
class ResolvedField:
    """一个逻辑字段跨所有层解析后的结果。"""
    key: str
    value: Optional[str]
    source_layer: str = ""
    source_path: str = ""
    layers: List[LayerValue] = field(default_factory=list)

    @property
    def is_set(self) -> bool:
        return bool(self.value)

    def differing_layers(self) -> List[LayerValue]:
        """取值与生效值不同的其它层。用于「多层配置取值不一致」这条检测。"""
        if not self.value:
            return []
        return [lv for lv in self.layers if not lv.effective and lv.value != self.value]


@dataclass
class ExtraAuthHeader:
    """通过跟主鉴权字段平行的另一种机制（比如自定义请求头环境变量）
    额外发现的、同样可能承载鉴权的请求头。

    这跟 FIELD_AUTH 解析出来的主字段不是替代关系：真实客户端两个都会发，
    Suture 不应该替用户决定网关到底认哪一个——探活/真实请求要把两者都带上，
    检测层则要如实提示"这里同时有两个来源在生效"。"""
    header: str        # 请求头名字，比如 "Token"
    value: str
    source: str         # 从哪个变量/文件来的，展示用，比如 "ANTHROPIC_CUSTOM_HEADERS"


@dataclass
class HarnessConfig:
    """一个 harness 体检所需的全部输入。"""
    harness_id: str
    display_name: str
    fields: Dict[str, ResolvedField] = field(default_factory=dict)
    files: List[FileState] = field(default_factory=list)

    # 鉴权的两种结构：直接存值，或者存一个环境变量名（间接引用）
    auth_is_indirect: bool = False
    auth_env_name: Optional[str] = None       # 间接引用时，配置里写的那个变量名
    auth_env_resolved: bool = False           # 那个变量名对应的环境变量是否取到了值
    auth_header: str = "x-api-key"            # 这个客户端实际会发哪个请求头
    auth_conflict: Optional[str] = None       # 两处同时填了鉴权信息时的说明
    extra_auth_headers: List[ExtraAuthHeader] = field(default_factory=list)
    # ANTHROPIC_CUSTOM_HEADERS 这类自定义头变量的值实际来自哪一层（claude_code 填，
    # 其余 harness 留空）。判定层据此判断「往配置文件补 Token 头」是否真的生效。
    custom_headers_source_layer: str = ""

    # DeepSeek Harness 的模型是一份注册清单而不是单个选中值，单独放在这里，
    # 每一项都要对着网关路由表校验。另外两个 harness 这里为空。
    model_candidates: List[str] = field(default_factory=list)

    notes: List[str] = field(default_factory=list)   # 仅供参考的信息，不是错误

    @property
    def has_any_config(self) -> bool:
        if any(f.is_set for f in self.fields.values()):
            return True
        return any(f.exists for f in self.files)

    def field(self, key: str) -> ResolvedField:
        return self.fields.get(key, ResolvedField(key=key, value=None))


class HarnessAdapter:
    """所有 harness 适配器的接口。"""

    harness_id: str = ""
    display_name: str = ""
    config_format: str = ""      # "json" / "toml" / "yaml"，用于语法检测的措辞
    binary_name: Optional[str] = None    # 客户端在 PATH 里的命令名；None 表示暂无可靠探测

    # 这个客户端能否通过它的配置文件表达「往某个自定义请求头里塞 Key」
    # （deepseek / claude_code 可以；codex 目前没有这个机制）。为 True 时，
    # 网关 401 + key 没放进 required_header 会给一条「一键补请求头」的修复，
    # 否则只给外部处理指引。
    supports_extra_auth_header: bool = False

    # 这个客户端自己（不靠 Suture 写配置）有没有一个能附加任意自定义请求头的
    # 位置（如环境变量 ANTHROPIC_CUSTOM_HEADERS、路由的 headers 块）。为 False
    # 时它只会把 Key 放进固定的那一个头（codex 是 Authorization: Bearer），
    # 网关若只认另一个自定义头（AI Gate 的 Token），这个客户端就完全不可达——
    # 此时连「让用户手动补头 / 填 Key」都是无效引导，只能给外部说明。
    can_send_custom_request_headers: bool = False

    # ---- 探测 ----
    def detect(self, env=None, home=None, project_dir=None) -> bool:
        """本机是否装了/配置过这个 harness。"""
        raise NotImplementedError

    def detect_binary(self, env=None, home=None) -> bool:
        """是否真的装了客户端的可执行程序（shutil.which 在 PATH 里找，
        Windows 会按 PATHEXT 自动找到 .cmd/.exe）。

        这跟 detect()（探测配置痕迹）是两回事：detect_binary 用于区分
        「没装客户端」和「装了但没配置」——前者引导去安装，后者引导去配置。
        测试或特殊部署用 SUTURE_BINARY_OVERRIDE_<HARNESS_ID> 覆盖。
        home 参数保留给按配置目录判装的适配器（如 deepseek），PATH 判装
        的适配器用不到。"""
        env = env if env is not None else os.environ
        flag = env.get(f"SUTURE_BINARY_OVERRIDE_{self.harness_id.upper()}")
        if flag is not None:
            return flag.strip().lower() in ("1", "true", "yes", "on")
        if not self.binary_name:
            return False
        return shutil.which(self.binary_name, path=env.get("PATH") or os.environ.get("PATH")) is not None

    # ---- 安装（P3 安装 Tab 用）----
    def install_command(self, env=None) -> Optional[List[str]]:
        """官方安装命令（以列表形式，可直接 subprocess.run）。
        返回 None 表示还没有确认可靠的自动安装方式，只能给指引。"""
        return None

    def install_guide(self, env=None) -> str:
        """自动安装不可用时的引导文字（去哪个官网、跑什么命令）。"""
        return ""

    def key_store_guide(self, env=None) -> str:
        """告诉用户这个客户端的 Key 该怎么存（写到哪 / 设成哪个环境变量）。"""
        return ""

    def unparseable_write_target(self, cfg: HarnessConfig,
                                 fields: Optional[set] = None) -> Optional[str]:
        """修复要写进去的那个文件读不懂时，返回它的路径；能正常写返回 None。

        用途有两个：判定层据此不给出「点了也写不进去」的自动修复按钮，执行层
        据此拒绝写入（见 RefuseWrite）。三个适配器里只有会整文件重写的需要实现。

        fields 是这次要写的逻辑字段集合（None = 不限定）。必须带上它：同一个
        harness 的不同字段可能落在不同文件上，按客户端整体判会把「某一个文件坏了」
        错误地扩散成「所有字段都不能改」。"""
        return None

    # ---- 读取 ----
    def read(self, env=None, home=None, project_dir=None) -> HarnessConfig:
        raise NotImplementedError

    # ---- 写入 ----
    def writable_paths(self, cfg: HarnessConfig) -> List[str]:
        """修复时可能会改到的文件，修复前要先备份这些。"""
        raise NotImplementedError

    def apply(self, cfg: HarnessConfig, changes: Dict[str, str],
              env=None, home=None, project_dir=None) -> List[str]:
        """把 {逻辑字段: 新值} 写回配置。返回给用户看的改动说明。"""
        raise NotImplementedError

    def generate_minimal_config(self, base_url: str, model: str,
                                env=None, home=None, project_dir=None) -> str:
        """全新用户从零生成一份最小可用配置，返回写入的文件路径。"""
        raise NotImplementedError

    def self_test_command(self) -> Optional[List[str]]:
        """让这个客户端自己跑一次最小的非交互请求，用它自己解析出来的配置和凭据。
        用在「客户端里什么都没配，但可能本来就不需要配」这种场景上：Suture 自己
        没有地址可发请求，只能让客户端自证。返回 None 表示这个客户端还没有确认过
        可用的非交互调用方式——那就不做自证，宁可不测，也不去猜一个命令行参数
        然后把失败当成"这台机器有问题"。"""
        return None

    def store_key(self, cfg: HarnessConfig, key: str,
                  env=None, home=None, project_dir=None) -> Tuple[List[str], List[str], str]:
        """把用户填的 AI Gate Key 存到「这个客户端真的会去读」的地方。

        返回 (steps, changed_paths, note)：
          steps        给用户看的操作说明（人话）
          changed_paths 这次改动了哪些文件（用于先备份/回滚）；只写环境变量时为 []
          note         额外的提示（比如"需重开终端生效"），没有则 ""
        注意：这里不负责备份和回滚，那由引擎在调用前基于 changed_paths 处理。"""
        raise NotImplementedError


# ---- 各适配器共用的小工具 ----

def resolve_home(home: Optional[str], env: Optional[Dict[str, str]] = None) -> str:
    if home:
        return home
    env = env if env is not None else os.environ
    return env.get("HOME") or env.get("USERPROFILE") or os.path.expanduser("~")


def resolve_project_dir(project_dir: Optional[str]) -> str:
    return project_dir or os.getcwd()


def build_resolved(key: str, layers_in_priority_order: List[LayerValue]) -> ResolvedField:
    """按优先级从高到低传入各层取值，产出解析结果。"""
    rf = ResolvedField(key=key, value=None)
    for lv in layers_in_priority_order:
        if lv.value:
            rf.layers.append(lv)
    for lv in rf.layers:
        if rf.value is None:
            rf.value = lv.value
            rf.source_layer = lv.layer
            rf.source_path = lv.path
            lv.effective = True
    return rf


def mask_secret(value: Optional[str]) -> str:
    """展示 Key 时只露前 4 位和后 4 位。界面和日志里任何位置都必须走这里。"""
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * max(4, len(value) - 8)}{value[-4:]}"


def write_bytes_atomic(path: str, data: bytes) -> None:
    """先写同目录下的临时文件、再原子替换目标。

    为什么不直接 `open(path, "w")`：那是**先截断再写**。只要序列化/编码在写入
    之后才失败（写出器拒绝某个值、内容编码不了 UTF-8、磁盘满），用户的原文件
    就已经被清空了——配置没了，而且报错里还只说"写入失败"。改成「内容先全部
    准备好、落到临时文件、最后一步替换」，失败时原文件逐字节不动。

    临时文件放在同目录是为了 os.replace 能在同一个文件系统内原子完成；
    用 mkstemp 生成唯一名字，避免和别的写者抢同一个临时文件。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".suture-write-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        # 走到这儿说明没能替换成功，原文件还在。清掉临时文件再往外抛。
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
