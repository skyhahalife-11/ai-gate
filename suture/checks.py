"""检测层。

只针对适配器归一化后的结构做判断，不认任何一个 harness 的原始字段名，
所以同一套判断能覆盖三个 harness。这一层只产出问题清单，不写任何文件。
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .harness import get_adapter
from .harness.base import (
    FIELD_AUTH, FIELD_BASE_URL, FIELD_EXTRA_HEADER, FIELD_MODEL, HarnessConfig,
    mask_secret,
)
from .profile import expected_base_url, expected_base_urls, model_ids, model_index

FIXABLE_YES = "yes"
FIXABLE_PARTIAL = "partial"
FIXABLE_NO = "no"


@dataclass
class Finding:
    key: str                     # 检测项标识
    label: str                   # 展示名
    ok: bool
    detail: str                  # 一句话说清楚问题是什么
    fixable: str = FIXABLE_NO
    current_value: str = ""      # 展示用，已掩码
    suggested_value: str = ""
    fix_field: Optional[str] = None   # 修复要写的逻辑字段
    fix_value: Optional[str] = None   # 修复要写的值
    note: str = ""
    # 有几个候选值、但 Suture 不该替用户挑哪个时用这个：界面把它们渲染成
    # 可以点的选项，用户点了哪个就把 fix_field 写成对应的 value——跟「一键修复」
    # 批量套用 fixable=YES 是两条不同的路，这条永远需要用户自己点一下。
    # 每一项是 {"label": 给用户看的名字, "value": 真要写进去的完整值}——
    # 两者不一定相同：比如 DeepSeek 一次注册了好几个模型，选中某一个候选时
    # 要写回去的是「其它模型保持不动、只把这一个换掉」之后的完整清单，
    # 不能让 apply_choice 把其它已经注册好的模型顶掉。
    choices: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class InfoItem:
    """仅供参考、不算错误的信息。"""
    text: str


# ---------- 配置文件语法 ----------

def check_config_syntax(cfg: HarnessConfig) -> List[Finding]:
    out: List[Finding] = []
    for fs in cfg.files:
        if not fs.exists:
            continue
        label = f"{fs.layer}语法"
        if not fs.parse_ok:
            out.append(Finding(
                key=f"syntax:{fs.path}", label=label, ok=False,
                detail=fs.parse_error,
                fixable=FIXABLE_NO,
                note=f"文件位置：{fs.path}",
            ))
        else:
            out.append(Finding(
                key=f"syntax:{fs.path}", label=label, ok=True,
                detail="格式没有问题。", note=f"文件位置：{fs.path}"))
    return out


def check_unknown_keys(cfg: HarnessConfig, profile: Dict[str, Any]) -> List[Finding]:
    """字段名拼错是另一种失败方式：文件能正常解析，但这个字段不会被认出来，
    等同于没配置这一项，所以要单独查、单独提示。"""
    known = profile.get("known_settings_keys", [])
    out: List[Finding] = []
    for fs in cfg.files:
        if not (fs.exists and fs.parse_ok and fs.unknown_keys):
            continue
        for key in fs.unknown_keys:
            near = difflib.get_close_matches(key, known, n=1, cutoff=0.7)
            if near:
                out.append(Finding(
                    key=f"unknown-key:{fs.path}:{key}", label=f"{fs.layer}字段名", ok=False,
                    detail=f"字段名 {key} 不是可识别的配置项，看起来是 {near[0]} 拼错了。"
                           "文件本身能正常解析，但这个字段不会被认出来，等同于没配置这一项。",
                    current_value=key, suggested_value=near[0], fixable=FIXABLE_NO,
                    note=f"文件位置：{fs.path}"))
    return out


def check_layer_active(cfg: HarnessConfig) -> List[Finding]:
    """某一层配置存在但实际不会被采纳（Codex 的不可信项目）——
    用户往往正是在改这一层，必须明确告诉他改了没用。"""
    out: List[Finding] = []
    for fs in cfg.files:
        if fs.exists and not fs.active:
            out.append(Finding(
                key=f"inactive-layer:{fs.path}", label=f"{fs.layer}未生效", ok=False,
                detail=fs.inactive_reason, fixable=FIXABLE_NO,
                note=f"文件位置：{fs.path}"))
    return out


# ---------- 网关地址 ----------

def check_base_url(cfg: HarnessConfig, profile: Dict[str, Any]) -> Finding:
    expected = expected_base_url(profile, cfg.harness_id)
    rf = cfg.field(FIELD_BASE_URL)
    label = "网关地址"

    if not rf.is_set:
        return Finding(key="base_url", label=label, ok=False,
                       detail="没有配置网关地址。", suggested_value=expected,
                       fixable=FIXABLE_YES, fix_field=FIELD_BASE_URL, fix_value=expected)

    value = rf.value
    cleaned = value.strip()
    problems: List[str] = []

    if cleaned != value:
        problems.append("地址前后有多余空格")
    if cleaned.startswith("http://") and profile.get("base_url", {}).get("require_https", True):
        cleaned = "https://" + cleaned[len("http://"):]
        problems.append("协议应该是 https，写成了 http")
    elif not cleaned.startswith(("http://", "https://")):
        cleaned = "https://" + cleaned.lstrip("/")
        problems.append("地址少了 https:// 前缀")

    normalized = cleaned.rstrip("/")
    expected_norm = expected.rstrip("/")
    # 网关如果登记了不止一个能进的地址（比如另一个专门给某种网络环境/直连用的
    # 地址），命中其中任意一个都算对，不是只有首选那一个才算——不然会把一个
    # 本来就有效的入口误判成"地址不对"，然后一键修复还会把它改成不适用这条
    # 网络路径的首选地址。
    alternates = {a.rstrip("/") for a in expected_base_urls(profile, cfg.harness_id)[1:]}

    if normalized in alternates:
        return Finding(key="base_url", label=label, ok=True,
                       detail="地址和格式都正确（用的是网关登记的另一个可用入口）。",
                       current_value=value)

    if normalized != expected_norm:
        # 常见的两种：多拼了一段 /v1，或者少了 /v1
        if normalized == expected_norm + "/v1":
            problems.append("结尾多拼了一段 /v1，这个 harness 会自己拼接后面的路径，不需要手动加")
        elif expected_norm == normalized + "/v1":
            problems.append("结尾少了一段 /v1，这个 harness 需要地址里自带这一段")
        else:
            problems.append(f"地址跟网关登记的不一致，网关这边应该是 {expected_norm}")
        normalized = expected_norm
    elif cleaned.endswith("/") and not profile.get("base_url", {}).get("allow_trailing_slash", True):
        problems.append("结尾多了一个斜杠")

    if not problems:
        return Finding(key="base_url", label=label, ok=True,
                       detail="地址和格式都正确。", current_value=value)

    return Finding(key="base_url", label=label, ok=False,
                   detail="；".join(problems) + "。",
                   current_value=value, suggested_value=normalized,
                   fixable=FIXABLE_YES, fix_field=FIELD_BASE_URL, fix_value=normalized,
                   note=f"当前生效的是{rf.source_layer}里的值")


# ---------- 鉴权 ----------

def _check_key_prefix(value: str, gate_prefix: str, vendor_table: Dict[str, str]) -> Tuple[bool, str]:
    """按前缀判断像不像网关签发的 Key。返回 (是否像, 不像时的说明)。"""
    if not gate_prefix or value.startswith(gate_prefix):
        return True, ""
    vendor = _guess_vendor(value, vendor_table)
    return False, (f"不是网关签发的 Key（网关的 Key 以 {gate_prefix} 开头）"
                   + (f"，看起来是{vendor}的 Key" if vendor else ""))


def check_auth(cfg: HarnessConfig, profile: Dict[str, Any]) -> List[Finding]:
    out: List[Finding] = []
    auth_rules = profile.get("auth", {})
    gate_prefix = auth_rules.get("gateway_issued_key_prefix", "")
    vendor_table = auth_rules.get("known_vendor_key_prefixes", {})
    rf = cfg.field(FIELD_AUTH)
    extras = cfg.extra_auth_headers

    # 间接引用的 harness：先看那个变量名本身取不取得到值
    if cfg.auth_is_indirect:
        if not cfg.auth_env_name:
            out.append(Finding(
                key="auth-ref", label="鉴权引用", ok=False,
                detail="配置里没有写明去哪个环境变量取 Key。",
                suggested_value="AI_GATE_API_KEY",
                fixable=FIXABLE_YES, fix_field=FIELD_AUTH, fix_value="AI_GATE_API_KEY"))
        elif not cfg.auth_env_resolved:
            out.append(Finding(
                key="auth-ref", label="鉴权引用", ok=False,
                detail=f"配置指向 {cfg.auth_env_name}，但该变量未设置——"
                       "配置文件本身没有问题，实际发起请求时无法取得 Key。",
                current_value=cfg.auth_env_name, fixable=FIXABLE_NO,
                note="需要设置该环境变量，或将 Key 存入凭据文件；Suture 不会把 Key 明文写入配置文件"))
        else:
            out.append(Finding(
                key="auth-ref", label="鉴权引用", ok=True,
                detail=f"配置指向 {cfg.auth_env_name}，能正常取到值。",
                current_value=cfg.auth_env_name))

    has_primary = rf.is_set
    if not has_primary and not extras:
        if not cfg.auth_is_indirect:
            out.append(Finding(
                key="auth", label="鉴权信息", ok=False,
                detail="没有配置鉴权信息，请求会被网关直接拒绝。", fixable=FIXABLE_NO,
                note="需要在网关后台生成一个 Key"))
        return out

    if has_primary:
        value = rf.value
        if value != value.strip():
            out.append(Finding(
                key="auth-whitespace", label="鉴权信息格式", ok=False,
                detail="Key 前后有多余的空格或换行，多半是复制粘贴时带进来的，会导致请求头不合法。",
                current_value=mask_secret(value), suggested_value="去掉前后空格",
                fixable=FIXABLE_YES, fix_field=FIELD_AUTH, fix_value=value.strip()))
            value = value.strip()

        ok_prefix, why = _check_key_prefix(value, gate_prefix, vendor_table)
        if not ok_prefix:
            out.append(Finding(
                key="auth", label="鉴权信息来源", ok=False, detail=why + "。",
                current_value=mask_secret(value), fixable=FIXABLE_PARTIAL,
                note="正确的 Key 需要到网关后台重新生成，Suture 无法代为处理"))
        else:
            out.append(Finding(
                key="auth", label="鉴权信息", ok=True,
                detail="格式符合网关签发的 Key。", current_value=mask_secret(value),
                note=f"取自{rf.source_layer}" if rf.source_layer else ""))
    elif extras:
        out.append(Finding(
            key="auth", label="鉴权信息", ok=True,
            detail=f"未使用 API_KEY/AUTH_TOKEN 这两个专用字段，但 {extras[0].source} "
                   f"设置的 {extras[0].header} 头可作为鉴权凭据。",
            note="下方单独校验该请求头的内容"))

    # 平行的自定义头：跟主字段是「都会被发出去」的关系，不是互斥关系，
    # 每一个都要单独校验内容像不像网关签发的 Key。
    for extra in extras:
        label = f"鉴权信息（{extra.header} 头）"
        ok_prefix, why = _check_key_prefix(extra.value, gate_prefix, vendor_table)
        if not ok_prefix:
            out.append(Finding(
                key=f"auth-extra:{extra.header}", label=label, ok=False,
                detail=f"来自 {extra.source} 的 {extra.header} 头{why}。",
                current_value=mask_secret(extra.value), fixable=FIXABLE_PARTIAL,
                note="正确的 Key 需要到网关后台重新生成，Suture 无法代为处理"))
        else:
            out.append(Finding(
                key=f"auth-extra:{extra.header}", label=label, ok=True,
                detail=f"来自 {extra.source}，格式符合网关签发的 Key。",
                current_value=mask_secret(extra.value)))

    if cfg.auth_conflict:
        out.append(Finding(
            key="auth-conflict", label="鉴权字段位置", ok=False,
            detail=cfg.auth_conflict + "网关同时接受这两个请求头，不会直接导致失败，"
                   "但两者并存容易在后续修改时出错，建议统一到一处。",
            fixable=FIXABLE_YES, fix_field=FIELD_AUTH, fix_value=cfg.field(FIELD_AUTH).value,
            suggested_value="只保留一处"))

    # 主字段和自定义头同时生效、内容还不一样：客户端会把这几个请求头一起发给网关，
    # 哪个头网关真正采信没有验证过，不能替用户选，只能把已知的证据摆出来。
    active: List[Tuple[str, str, str]] = []
    if has_primary:
        active.append((cfg.auth_header, rf.value.strip(), rf.source_layer or "主鉴权字段"))
    for extra in extras:
        active.append((extra.header, extra.value, extra.source))
    distinct_values = {v for _, v, _ in active}
    if len(active) > 1 and len(distinct_values) > 1:
        parts = "、".join(f"{src} 里的 {header} 头" for header, _, src in active)
        detail = (f"{parts} 同时生效，客户端会将这几个请求头一起发送给网关，"
                  "尚未验证网关实际采信哪一个，无法据此判断。")
        good = [(h, s) for h, v, s in active if gate_prefix and v.startswith(gate_prefix)]
        bad = [(h, s) for h, v, s in active if gate_prefix and not v.startswith(gate_prefix)]
        if good and bad and len(good) + len(bad) == len(active):
            good_desc = "、".join(f"{h}（{s}）" for h, s in good)
            bad_desc = "、".join(f"{h}（{s}）" for h, s in bad)
            detail += f"根据 Key 前缀判断，{good_desc} 更符合网关签发 Key 的特征，{bad_desc} 不符合，建议保留前者、删除后者对应的配置。"
        else:
            detail += "建议确认后只保留一处，删除其余配置。"
        out.append(Finding(
            key="auth-multiple-active", label="鉴权信息来源冲突", ok=False, detail=detail,
            fixable=FIXABLE_NO,
            note="这些大多是环境变量，不是配置文件字段；Suture 不会修改环境变量，需要在对应位置手动删除多余的一项"))

    return out


def _guess_vendor(value: str, table: Dict[str, str]) -> str:
    for prefix, name in sorted(table.items(), key=lambda kv: -len(kv[0])):
        if value.startswith(prefix):
            return name
    return ""


# ---------- 协议版本头 ----------

def check_protocol_version(cfg: HarnessConfig, profile: Dict[str, Any]) -> List[Finding]:
    rules = profile.get("protocol_version", {})
    if not rules.get("check_enabled"):
        return []
    return [Finding(
        key="protocol-version", label="协议版本头", ok=True,
        detail=f"按规则要求 {rules.get('header_name')} 不低于 {rules.get('min_version')}，"
               "客户端会自动带上这个头。")]


# ---------- 模型名称 ----------

def check_models(cfg: HarnessConfig, profile: Dict[str, Any]) -> List[Finding]:
    known = model_ids(profile)
    index = model_index(profile)
    # 多模型 harness（比如 DeepSeek）用 model_candidates，没有单独一份「来源」；
    # 单模型 harness 走 FIELD_MODEL 这个解析结果，本身就带着来自哪一层的信息——
    # 之前这个来源信息在这里被扔掉了，模型这一项因此是全篇唯一不写"取自哪一层"的，
    # 这次顺手补上，跟鉴权那几项的呈现方式保持一致。
    field_rf = cfg.field(FIELD_MODEL)
    candidates = cfg.model_candidates or ([field_rf.value] if field_rf.is_set else [])
    source_note = ("取自" + field_rf.source_layer) if (not cfg.model_candidates and field_rf.source_layer) else ""

    if not candidates:
        # 不填模型名不是错误：客户端不填就用它自己的默认模型，很多人（尤其是不需要在
        # 客户端填任何网关配置的那类用户）从头到尾就没填过，也一直用得好好的。
        # 之前这里判成 ok=False，等于把一个正常状态报成故障，还把网关全部型号平铺出来，
        # 对这类用户就是满屏报错。所以默认降级成提示；只有当端到端请求真的没通、
        # 而且又没指定模型时，engine 那边才会把它升级成需要处理的问题（见 check()）。
        return [Finding(
            key="model", label="模型名称", ok=True,
            detail="未指定模型名称，不影响使用——未配置时客户端使用其默认模型。",
            fixable=FIXABLE_NO, fix_field=FIELD_MODEL,
            choices=[{"label": m, "value": m} for m in known])]

    out: List[Finding] = []
    for i, name in enumerate(candidates):
        out.append(_check_one_model(name, known, index, candidates, i, source_note))
    return out


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(".", "-").replace(" ", "")


def _replaced_list(candidates: List[str], position: int, new_value: str) -> str:
    """构造「只把这一项换成 new_value，其余原样保留」之后的完整清单（逗号拼接，
    跟 apply() 已有的 value.split(",") 对应）。像 DeepSeek 这种一次能注册好几个
    模型的 harness，修一个不该把其它已经注册好的模型顶掉；单模型的 harness
    这里的 candidates 只有一项，效果等同于直接写 new_value。"""
    updated = list(candidates)
    updated[position] = new_value
    return ",".join(updated)


def _check_one_model(name: str, known: List[str], index: Dict[str, Any],
                     candidates: List[str], position: int, source_note: str = "") -> Finding:
    label = "模型名称"
    if name in index:
        return Finding(key=f"model:{name}", label=label, ok=True,
                       detail=f"「{name}」在网关路由表里能精确匹配。", current_value=name,
                       note=source_note)

    # 只把「看起来是打字错误」的差异当成可以自动纠正的：大小写、连字符/点号、空格
    norm = _normalize(name)
    typo_matches = [k for k in known if _normalize(k) == norm]
    if len(typo_matches) == 1:
        target = typo_matches[0]
        return Finding(
            key=f"model:{name}", label=label, ok=False,
            detail=f"「{name}」和网关登记的「{target}」只差大小写或连字符写法。",
            current_value=name, suggested_value=target,
            fixable=FIXABLE_YES, fix_field=FIELD_MODEL,
            fix_value=_replaced_list(candidates, position, target))

    # 近似匹配的结果如果本身也是一个独立型号，不能自动选——那会让用户在
    # 不知情的情况下连到另一个真实存在的模型，比直接报错更危险。
    near = difflib.get_close_matches(name, known, n=4, cutoff=0.6)
    if near:
        return Finding(
            key=f"model:{name}", label=label, ok=False,
            detail=f"「{name}」不在网关路由表里。相近的有：{'、'.join(near)}——"
                   "均为独立型号，非拼写变体，需要确认具体使用哪一个；"
                   "自动替换存在连接到错误模型的风险。",
            current_value=name, suggested_value="、".join(near),
            fixable=FIXABLE_NO, fix_field=FIELD_MODEL,
            choices=[{"label": c, "value": _replaced_list(candidates, position, c)}
                    for c in near])

    return Finding(
        key=f"model:{name}", label=label, ok=False,
        detail=f"「{name}」不在网关路由表里，也没有相近的名字。",
        current_value=name, fixable=FIXABLE_NO, fix_field=FIELD_MODEL,
        choices=[{"label": m, "value": _replaced_list(candidates, position, m)}
                for m in known],
        note=f"网关当前提供 {len(known)} 个型号，可从列表中选择一个")


# ---------- 多层配置取值不一致 ----------

def check_layer_consistency(cfg: HarnessConfig) -> List[Finding]:
    """按已确认的使用规范，项目级/用户层不应该覆盖网关相关配置，
    出现不一致本身就算配置错误，需要统一到实际生效的值。

    但不提供一键修复：修复写的是生效的那一层，而生效的那一层本来就是这个值，
    写下去等于原地重写一遍、其它层一动不动，下次检查同一条原因原样回来。
    真要「统一」，得把其它层里那个键删掉——那是另一套动作，不是这个按钮能做的，
    所以这里如实说明，让用户自己去统一。"""
    out: List[Finding] = []
    for key in (FIELD_BASE_URL, FIELD_MODEL):
        rf = cfg.field(key)
        differing = rf.differing_layers()
        if not differing:
            continue
        others = "；".join(f"{lv.layer} 里是「{lv.value}」" for lv in differing)
        effective_is_managed = any(lv.managed and lv.effective for lv in rf.layers)
        if effective_is_managed:
            out.append(Finding(
                key=f"conflict:{key}", label="本地设置被组织托管配置覆盖", ok=False,
                detail=f"实际生效的是{rf.source_layer}里的「{rf.value}」（托管配置优先级最高，"
                       f"本地设置改不动它）；另外 {others}。",
                current_value=rf.value, fixable=FIXABLE_NO,
                note="该值来自组织统一下发的托管配置，本地这一层无论如何修改都不会生效，"
                     "因此不提供一键修复"))
            continue
        out.append(Finding(
            key=f"conflict:{key}", label="多层配置取值不一致", ok=False,
            detail=f"同一项在多个地方设置了不同的值：实际生效的是{rf.source_layer}里的"
                   f"「{rf.value}」，另外 {others}。",
            current_value=rf.value, fixable=FIXABLE_NO,
            note="需要把其它几层里这一项删掉或改成同一个值（Suture 不自动删你其它文件里的"
                 "配置，避免误伤），统一之后再重新检查"))
    return out


# ---------- 汇总 ----------

def run_all_checks(cfg: HarnessConfig, profile: Dict[str, Any]) -> List[Finding]:
    findings: List[Finding] = []
    findings += check_config_syntax(cfg)
    findings += check_unknown_keys(cfg, profile)
    findings += check_layer_active(cfg)
    findings.append(check_base_url(cfg, profile))
    findings += check_auth(cfg, profile)
    findings += check_protocol_version(cfg, profile)
    findings += check_models(cfg, profile)
    findings += check_layer_consistency(cfg)
    return findings


def collect_info(cfg: HarnessConfig) -> List[InfoItem]:
    return [InfoItem(text=n) for n in cfg.notes]


# ---------- 连通失败时的「可能原因」诊断（连通优先模式用） ----------
# checks.py 的静态项原本是唯一的检查手段；重构后它们降级成"连不上时才会跑的
# 诊断器"——只有 block 分支才调用 run_all_checks，把 not-ok 的 finding 翻译成
# 给用户看的、可挑着修的原因清单（BlockIssue）。成功路径完全不碰这里。


def e2e_hint(e2e: Optional[Dict[str, Any]]) -> str:
    """把一次端到端请求的失败特征归成一个方向，用于给各原因定优先级：
    auth / model / base_url / gateway_side / local / unknown。"""
    if not e2e:
        return "unknown"
    cls = e2e.get("classification")
    status = e2e.get("status")
    if cls == "auth_error" or status in (401, 403):
        return "auth"
    if status in (400, 404):
        return "model"          # 404 最常见是"模型不在路由表/路径不对"
    if cls in ("timeout", "network_error", "rate_limited", "server_error"):
        return "gateway_side"
    if cls == "local_error":
        return "local"          # 请求压根没发出去，是本地构造的问题
    return "unknown"


def _readonly_layers(cfg: HarnessConfig, key: str) -> List[str]:
    """这个字段实际生效的那一层是不是「Suture 写不到」的层，返回合适的说法。

    环境变量（OS 级，Suture 只写配置文件）和组织托管配置（IT 下发、只读）都属于
    这类：给了「修复」按钮也是写了不生效，用户点了没反应、原地循环。"""
    rf = cfg.field(key)
    names: List[str] = []
    if not rf.is_set or not rf.source_layer:
        return names
    if rf.source_layer == "环境变量":
        names.append("系统环境变量")
    if any(lv.managed and lv.effective for lv in rf.layers):
        names.append("组织统一下发的托管配置")
    return names


def _readonly_source(cfg: HarnessConfig, key: str) -> bool:
    return bool(_readonly_layers(cfg, key))


def _write_target_unreadable(cfg: HarnessConfig) -> bool:
    """修复要写进去的那个文件读不懂 → 写进去会整份覆盖，不给自动修复按钮。"""
    try:
        return get_adapter(cfg.harness_id).unparseable_write_target(cfg) is not None
    except Exception:      # noqa: BLE001 —— 判定层不该因为适配器的小毛病整轮失败
        return False


def finding_to_issue(f: Finding, cfg: HarnessConfig) -> Dict[str, Any]:
    """把一条静态 finding 翻成给用户看的「原因 + 怎么修」。
    不负责判严重度（那要看 e2e 失败的方向），只定 repair_kind 和动作字段。
    需要 cfg 是因为「这一项能不能自动修」取决于它的生效层写不写得进去。"""
    key = f.key
    cur: Dict[str, Any] = {
        "id": key,
        "title": "",
        "detail": f.detail,
        "current_value": f.current_value,
        "repair_kind": "none",
        "fix_field": None,
        "fix_value": None,
        "choices": f.choices,
        "prompt": None,
    }

    if key.startswith("syntax:"):
        cur["title"] = "配置文件格式有问题"
    elif key.startswith("unknown-key:"):
        cur["title"] = "配置里有不认识的字段名"
        cur["detail"] = f.detail + " 请把它改成工具认得的正确字段名。"
    elif key.startswith("inactive-layer:"):
        cur["title"] = "有配置，但这一层实际不会生效"
        cur["repair_kind"] = "external"
    elif key.startswith("conflict:"):
        if "托管配置" in (f.note or ""):
            cur["title"] = "本地设置被组织统一下发的托管配置覆盖"
        else:
            # 不给 auto：写的就是生效层，写下去等于原地重写，其它层没动、
            # 下次检查原样回来。如实说明，让用户自己去统一。
            cur["title"] = "同一个设置在多层配置里不一致"
            cur["repair_kind"] = "none"
            cur["detail"] = f.detail + " 请把其它几层里这一项删掉或改成同一个值。"
    elif key == "base_url":
        # 生效值来自只读的一层（环境变量 / 组织托管配置）时，auto 写配置文件根本
        # 不生效——点了没反应。同理，目标文件读不懂时写进去会整份覆盖。两种都
        # 不给按钮（前者在 blocked_reasons 里统一换成外部说明，后者由格式错误那条
        # 说明原因），避免用户点了白点、或者点出一次数据丢失。
        if _readonly_source(cfg, FIELD_BASE_URL) or _write_target_unreadable(cfg):
            cur["title"] = "网关地址不对（改配置文件在当前情况下不生效或不被允许）"
            cur["repair_kind"] = "external"
            cur["fix_field"] = f.fix_field
        else:
            cur["title"] = "网关地址不对"
            cur["repair_kind"] = "auto"
            cur["fix_field"] = f.fix_field
            cur["fix_value"] = f.fix_value
    elif key == "auth-whitespace":
        cur["title"] = "Key 前后有多余的空格或换行"
        cur["repair_kind"] = "auto"
        cur["fix_field"] = f.fix_field
        cur["fix_value"] = f.fix_value
    elif key == "auth-ref":
        if f.fixable == FIXABLE_YES:
            cur["title"] = "鉴权引用没写好"
            cur["repair_kind"] = "auto"
            cur["fix_field"] = f.fix_field
            cur["fix_value"] = f.fix_value
        else:
            cur["title"] = "Key 还没设置"
            cur["repair_kind"] = "input"
            cur["fix_field"] = FIELD_AUTH
            cur["prompt"] = "在 AI Gate 后台生成 Key 后粘贴到这里"
    elif key == "auth":
        cur["title"] = "鉴权信息有问题（Key 缺失或不对）"
        cur["repair_kind"] = "input"
        cur["fix_field"] = FIELD_AUTH
        cur["prompt"] = "在 AI Gate 后台生成 Key 后粘贴到这里"
    elif key.startswith("auth-extra:"):
        cur["title"] = "自定义请求头里的 Key 不像是网关签发的"
        cur["repair_kind"] = "none"
        cur["detail"] = f.detail + " 需要到来源处（通常是环境变量）改掉它。"
    elif key in ("auth-conflict", "auth-multiple-active"):
        cur["title"] = "鉴权信息在多个位置重复设置"
        cur["repair_kind"] = "none"
    elif key == "model" or key.startswith("model:"):
        if _readonly_source(cfg, FIELD_MODEL) or _write_target_unreadable(cfg):
            cur["title"] = "模型名称不对（改配置文件在当前情况下不生效或不被允许）"
            cur["repair_kind"] = "external"
            cur["fix_field"] = f.fix_field
        elif f.choices and f.fix_field:
            cur["title"] = "模型不在网关支持的列表里"
            cur["repair_kind"] = "choice"
            cur["fix_field"] = f.fix_field
        elif f.fixable == FIXABLE_YES:
            cur["title"] = "模型名字只差大小写或写法"
            cur["repair_kind"] = "auto"
            cur["fix_field"] = f.fix_field
            cur["fix_value"] = f.fix_value
        else:
            cur["title"] = "模型不在网关支持的列表里"
            cur["repair_kind"] = "choice" if f.fix_field else "none"
            cur["fix_field"] = f.fix_field
    else:
        cur["title"] = f.detail
    return cur


def _severity_for(issue_id: str, hint: str) -> str:
    if hint == "auth" and (issue_id == "auth" or issue_id.startswith("auth-")):
        return "high"
    if hint == "model" and (issue_id == "model" or issue_id.startswith("model:")):
        return "high"
    if hint == "base_url" and issue_id == "base_url":
        return "high"
    if issue_id == "local-request":
        return "medium"
    if issue_id == "base_url" or issue_id.startswith("auth") or issue_id == "model" \
            or issue_id.startswith("model:") or issue_id.startswith("syntax:"):
        return "medium"
    return "low"


def _e2e_fallback(cfg: HarnessConfig, profile: Dict[str, Any], e2e: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """静态检查全过、但真实请求就是不通时用的兜底原因（比如 Key 被后台吊销、
    或请求到了但模型/路径的问题静态项看不出来）。"""
    hint = e2e_hint(e2e)
    known = model_ids(profile)
    if hint == "local":
        # 请求在本机就没构造成功（Key 带换行 → Invalid header value、地址缺协议头 →
        # unknown url type）。这类必须点明是本地问题，不能沿用下面那句"网关或网络
        # 侧的问题、不是本地配置造成的、联系网关值班"——那是把用户指反方向，
        # 而且 severity=high 会把它顶到清单第一条，压过真正能修的那一条。
        return [{
            "id": "local-request", "title": "请求没能发出去，是本机这一侧的配置问题",
            "detail": "这次请求还没有发到网关就在本机失败了（常见于 Key 或地址里带了"
                      "多余的空格/换行，或地址没写完整）。请先按下面列出的本地问题逐条处理，"
                      "处理完再点「开始检查」。",
            "current_value": "", "severity": "medium", "repair_kind": "external",
            "fix_field": None, "fix_value": None, "choices": [], "prompt": None,
        }]
    if hint == "auth":
        return [{
            "id": "auth", "title": "网关拒绝了这个 Key（可能已失效或被后台吊销）",
            "detail": "网关返回了鉴权错误，但本机配置看起来没问题——最常见是 Key 在网关后台"
                      "被重新生成或吊销了。重新生成一个粘贴进来。",
            "current_value": "", "severity": "high", "repair_kind": "input",
            "fix_field": FIELD_AUTH, "fix_value": None, "choices": [],
            "prompt": "在 AI Gate 后台重新生成 Key 后粘贴到这里",
        }]
    if hint == "model" and not cfg.field(FIELD_MODEL).is_set and not cfg.model_candidates:
        return [{
            "id": "model", "title": "没指定模型，默认模型网关可能不认",
            "detail": "端到端请求没成功，而且客户端里没指定用哪个模型。从网关支持的列表里"
                      "选一个默认模型写进去。",
            "current_value": "", "severity": "high", "repair_kind": "choice",
            "fix_field": FIELD_MODEL, "fix_value": None,
            "choices": [{"label": m, "value": m} for m in known],
            "prompt": None,
        }]
    if hint == "model":
        # base_url 静态检查没报错、模型也在快照清单里，但真实请求仍 400/404——
        # 那多半是型号在网关侧已下线/改名、内置型号快照没跟上。此时别再给
        # 「把地址修正到规范地址」的 auto 修复：地址本来就对，点了等于把同一个值
        # 重写一遍、原地循环。
        cur = (cfg.field(FIELD_BASE_URL).value or "").strip().rstrip("/")
        expected_norm = expected_base_url(profile, cfg.harness_id).rstrip("/")
        alternates = {a.rstrip("/") for a in expected_base_urls(profile, cfg.harness_id)[1:]}
        if cur and (cur == expected_norm or cur in alternates):
            tried = cfg.model_candidates or ([cfg.field(FIELD_MODEL).value]
                                             if cfg.field(FIELD_MODEL).is_set else [])
            desc = "、".join(f"「{m}」" for m in tried[:3]) or "默认模型"
            return [{
                "id": "model-unrouted", "title": "网关实际拒绝了配置里的模型",
                "detail": ("网关返回了「模型或路径不存在」，但配置用的 " + desc + " 在网关型号"
                           "清单里、地址也正确——最可能是这个型号在网关侧已下线或改名，型号"
                           "清单（内置快照）还没跟上。请联系网关确认现在可用的型号，或换一个"
                           "型号再试；清单更新后重新检查即可。"),
                "current_value": "", "severity": "high", "repair_kind": "external",
                "fix_field": None, "fix_value": None, "choices": [], "prompt": None,
            }]
        return [{
            "id": "base_url", "title": "请求没到网关（地址或路径可能不对）",
            "detail": "静态判断都符合登记值，但真实请求仍失败。可以先把网关地址修正到规范"
                      "地址试一次；如果已是指向规范地址仍失败，再联系网关排查。",
            "current_value": "", "severity": "high", "repair_kind": "auto",
            "fix_field": FIELD_BASE_URL,
            "fix_value": expected_base_url(profile, cfg.harness_id),
            "choices": [], "prompt": None,
        }]
    return [{
        "id": "gateway-side", "title": "AI Gate 网关侧暂时连不上",
        "detail": "这次失败更像网关或网络侧的问题，不是本地配置造成的。稍后重试；"
                  "如果一直这样，联系网关值班人员。",
        "current_value": "", "severity": "high", "repair_kind": "external",
        "fix_field": None, "fix_value": None, "choices": [], "prompt": None,
    }]


def _issue_class(issue_id: str) -> str:
    """问题属于哪一类，用于判断 e2e 失败方向是否已经被清单覆盖。"""
    if issue_id == "base_url":
        return "base_url"
    if issue_id == "model" or issue_id.startswith("model:"):
        return "model"
    if issue_id.startswith("auth"):
        return "auth"
    if issue_id == "gateway-side":
        return "gateway_side"
    return "other"


def _same_class_present(issue_id: str, issues: List[Dict[str, Any]]) -> bool:
    cls = _issue_class(issue_id)
    return any(_issue_class(i["id"]) == cls for i in issues)


def _required_header(profile: Dict[str, Any]) -> str:
    return str((profile.get("auth") or {}).get("required_header") or "").strip()


def auth_requirement_unsatisfiable(cfg: HarnessConfig,
                                   profile: Dict[str, Any]) -> bool:
    """网关要求从某个请求头读 Key，而这个客户端既没在发那个头、也没有任何位置
    能附加它 → 无论怎么配，这个客户端都连不上这座网关。此时填 Key / 补头都救不了，
    只能给外部说明（比如 AI Gate 只认 Token，而 Codex 只能发 Authorization）。

    判定是契约驱动的，不是写死客户端名：若哪天 profile 的 required_header 改成一个
    客户端本来就会发的头（比如 Authorization），这个客户端就不算 unsatisfiable。"""
    required = _required_header(profile)
    if not required:
        return False
    adapter = get_adapter(cfg.harness_id)
    if adapter.can_send_custom_request_headers:
        return False
    sent = {cfg.auth_header.lower()}
    sent |= {e.header.lower() for e in cfg.extra_auth_headers}
    return required.lower() not in sent


def auth_incompatible_issue(cfg: HarnessConfig,
                            profile: Dict[str, Any]) -> Dict[str, Any]:
    """unsatisfiable 时给的那条「无法用这个客户端连 AI Gate」的原因。"""
    required = _required_header(profile)
    name = cfg.display_name or cfg.harness_id
    sent_name = cfg.auth_header or "普通鉴权头"
    detail = (
        f"AI Gate 只从 {required} 请求头读取 Key（x-api-key / Authorization 一律拒绝），"
        f"而 {name} 的配置里没有能附加自定义请求头的位置——它只会把 Key 放在 "
        f"{sent_name} 头发送。所以即使 Key 正确，请求也会被网关拒绝（401）。"
        f"当前这条路走不通：请改用 Claude Code CLI 或 DeepSeek Harness 连接 AI Gate；"
        f"等 {name} 支持自定义请求头、或网关放开对其它鉴权头的采信后，再回来重新检查。"
    )
    return {
        "id": "auth-incompatible",
        "title": f"{name} 无法携带 {required} 请求头，连不上 AI Gate",
        "detail": detail,
        "current_value": "",
        "severity": "high",
        "repair_kind": "external",
        "fix_field": None,
        "fix_value": None,
        "choices": [],
        "prompt": None,
    }


def _auth_header_issue(cfg: HarnessConfig,
                       profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """401 且 key 是网关签发的、但配置没把它放进网关要求的请求头时，
    主因大概率是「放错请求头」而不是「Key 失效」。返回一条可自动补头的问题；
    不满足条件（没 key / key 前缀不对 / 头已经在发）返回 None。"""
    auth_rules = profile.get("auth") or {}
    required = _required_header(profile)
    if not required:
        return None
    key = cfg.field(FIELD_AUTH).value
    if not key:
        return None
    gate_prefix = auth_rules.get("gateway_issued_key_prefix", "")
    ok_prefix, _why = _check_key_prefix(key, gate_prefix,
                                        auth_rules.get("known_vendor_key_prefixes", {}))
    if not ok_prefix:
        # Key 本身就不是网关签发的 → 主诉是换一个 Key，不是补请求头
        return None
    sent = {cfg.auth_header.lower()}
    sent |= {e.header.lower() for e in cfg.extra_auth_headers}
    if required.lower() in sent:
        return None
    adapter = get_adapter(cfg.harness_id)
    auto = adapter.supports_extra_auth_header
    return {
        "id": "auth-header",
        "title": f"Key 没放进网关要求的 {required} 请求头",
        "detail": (f"AI Gate 只从 {required} 请求头读取 Key；当前配置把 Key 放在 "
                   f"{cfg.auth_header} 头发送（或只放在普通自定义头里），网关读不到，"
                   "所以即使 Key 有效也会一直 401。"
                   + ("" if auto else
                      f" 需要在客户端配置里补上这个 {required} 请求头，Suture 暂时不能自动帮你写。")),
        "current_value": mask_secret(key),
        "severity": "high",
        "repair_kind": "auto" if auto else "external",
        "fix_field": FIELD_EXTRA_HEADER if auto else None,
        "fix_value": key if auto else None,
        "choices": [], "prompt": None,
    }


def blocked_reasons(cfg: HarnessConfig, profile: Dict[str, Any],
                    e2e: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """只在客户端连不上时调用：把静态检查 + 端到端失败特征翻译成可挑选的原因清单。

    除了静态项之外保证两点：
      · 真实失败方向（鉴权/模型/网关侧）即使被无关静态问题干扰，也必须出现在清单里；
      · 401 + key 有效但没放进网关要求的头时，用「补请求头」专诊顶替那条
        「Key 被吊销，重新生成」的误导兜底。
    按严重度降序返回。"""
    hint = e2e_hint(e2e)

    # 契约层面就不可达：网关只认某个自定义头、而客户端根本带不了它（codex 之于
    # AI Gate 的 Token）。这与有没有发过 e2e 无关——缺 base_url 时同样连不上，
    # 先把用户带去修一个没用的地址（配好 base_url 之后才会冒出 401）是浪费。
    # 补头 / 换 Key / 修地址都救不了，任何静态项也都无关紧要——直接给单条外部
    # 说明，别把用户带去死循环。只有网关侧故障（跟本地配置无关）时不短路。
    if auth_requirement_unsatisfiable(cfg, profile) and hint != "gateway_side":
        return [auth_incompatible_issue(cfg, profile)]

    bad = [f for f in run_all_checks(cfg, profile) if not f.ok]
    issues: List[Dict[str, Any]] = []
    seen: set = set()
    for f in bad:
        it = finding_to_issue(f, cfg)
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        it["severity"] = _severity_for(it["id"], hint)
        issues.append(it)

    if hint == "auth":
        hi = _auth_header_issue(cfg, profile)
        if hi is not None:
            issues = [i for i in issues if not _issue_class(i["id"]) == "auth"]
            issues.append(hi)
            seen.add("auth-header")

    fallback = _e2e_fallback(cfg, profile, e2e)
    if not issues:
        issues = fallback
    elif e2e and fallback:
        fb = fallback[0]
        if fb["id"] not in seen and not _same_class_present(fb["id"], issues):
            issues.append(fb)

    # 生效来源是 Suture 写不到的那一层时，auto/choice 修复写了也不生效：
    #   · 系统环境变量（claude 的 ANTHROPIC_BASE_URL / ANTHROPIC_MODEL 压过文件层）；
    #   · 组织统一下发的托管配置（只读、优先级最高）。
    # 这类问题点自动修复 = 原地循环（conflict 那条更糟，会把只读层里的值复制回文件）；
    # _e2e_fallback 的兜底也会把同一条 no-op 的 base_url 自动修复再塞回来。
    # 三条路互相否定、永远收敛不了。放在 fallback 之后统一处理：整体换成一条外部说明，
    # 并去掉 conflict / 兜底里会复制错误值回文件的自动修复。
    readonly_wrong = {
        i["fix_field"] for i in issues
        if i.get("fix_field") in (FIELD_BASE_URL, FIELD_MODEL)
        and (i["id"] == "base_url" or i["id"] == "model" or i["id"].startswith("model:"))
        and _readonly_source(cfg, i["fix_field"])
    }
    if readonly_wrong:
        # 这一层写不到，所有指向它的原因（含 conflict 和兜底里那条 no-op 的自动修复）
        # 都被下面那一条外部说明取代，留着只会让用户点了没反应。
        issues = [i for i in issues if i.get("fix_field") not in readonly_wrong]
        sources: List[str] = []
        for key in (FIELD_BASE_URL, FIELD_MODEL):
            if key in readonly_wrong:
                sources.extend(_readonly_layers(cfg, key))
        sources = list(dict.fromkeys(sources))
        where = "、".join(sources) or "只读的来源"
        env_style = any(s == "系统环境变量" for s in sources)
        if env_style:
            howto = ("请到「系统设置 → 环境变量」里把对应的变量（ANTHROPIC_BASE_URL / "
                     "ANTHROPIC_MODEL）改成网关的规范地址/想用的模型，或直接删除它让配置文件"
                     "生效；改完再点「开始检查」。")
        else:
            howto = ("这一层由 IT 统一下发、本机改不动，请联系 IT 或网关维护人员修改下发内容；"
                     "本地怎么改都不会生效，所以这里不提供自动修复。")
        issues.append({
            "id": "readonly-source",
            "title": f"该设置来自{where}，在这里改不了",
            "detail": ("当前生效的网关地址/模型来自" + where + "，不是配置文件——它的优先级"
                       "更高，Suture 只能写配置文件、改不到它，所以这里不提供自动修复。"
                       + howto),
            "current_value": "", "severity": "high", "repair_kind": "external",
            "fix_field": None, "fix_value": None, "choices": [], "prompt": None})

    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: (order.get(i.get("severity", "low"), 3), i["id"]))
    return issues
