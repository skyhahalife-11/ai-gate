"""一个只覆盖配置文件常见写法的 YAML 读写器。

标准库没有 YAML 解析器，而引擎要保持只依赖标准库，所以这里自己实现一个
受限子集：嵌套映射、列表、标量、引号字符串、注释。

关键原则是遇到不支持的写法要明确报错，而不是猜一个结果继续跑——把配置读错
再据此去改用户的文件，比读不了更糟。以下写法一律拒绝并说明原因：
锚点与引用（& *）、块标量（| >）、流式写法（{} []）、多文档（---）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


class MiniYamlError(Exception):
    """无法安全解析时抛出，消息里说明是第几行、什么写法不支持。"""


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).rstrip()


def _scalar(raw: str, lineno: int) -> Any:
    s = raw.strip()
    if not s:
        return None
    if s[0] in "&*":
        raise MiniYamlError(f"第 {lineno} 行用到了锚点或引用（{s[0]}），这个读取器不支持这种写法")
    if s[0] in "|>":
        raise MiniYamlError(f"第 {lineno} 行用到了块标量（{s[0]}），这个读取器不支持这种写法")
    if s[0] in "[{":
        # 空集合是例外：`{}` / `[]` 没有歧义，而且写出时本来就会生成它们（见 _emit），
        # 解析器再不认就等于自己写的文件自己读不懂——那会让下一次读取把整份文件
        # 判成"语法错"，进而触发"读不懂就不写"。其余流式写法照旧拒绝。
        if s in ("{}", "[]"):
            return {} if s == "{}" else []
        raise MiniYamlError(f"第 {lineno} 行用到了流式写法（{s[0]}），这个读取器不支持这种写法")
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _split_key(content: str, lineno: int) -> Tuple[str, str]:
    quote = None
    for i, ch in enumerate(content):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        if ch == ":" and (i + 1 == len(content) or content[i + 1] in (" ", "\t")):
            key = content[:i].strip()
            if len(key) >= 2 and key[0] == key[-1] and key[0] in ("'", '"'):
                key = key[1:-1]
            return key, content[i + 1:].strip()
    raise MiniYamlError(f"第 {lineno} 行既不是「键: 值」也不是列表项，无法解析：{content!r}")


_DASH = object()


def _tokenize(text: str) -> List[Tuple[int, Any, int]]:
    """产出 (缩进, 内容或 _DASH, 行号)。列表项拆成一个 DASH 记号加一条
    位于内容实际列位置的记号，这样带首键的列表项能按普通映射解析。"""
    tokens: List[Tuple[int, Any, int]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if raw.strip().startswith("---"):
            raise MiniYamlError(f"第 {lineno} 行出现多文档分隔符，这个读取器只支持单个文档")
        line = _strip_comment(raw)
        if not line.strip():
            continue
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise MiniYamlError(f"第 {lineno} 行用制表符缩进，YAML 不允许，请改成空格")
        indent = len(line) - len(line.lstrip())
        content = line.strip()
        if content == "-" or content.startswith("- "):
            tokens.append((indent, _DASH, lineno))
            rest = content[1:].strip()
            if rest:
                tokens.append((indent + 2, rest, lineno))
        else:
            tokens.append((indent, content, lineno))
    return tokens


def _parse_block(tokens, i: int, indent: int):
    if i >= len(tokens):
        return None, i
    if tokens[i][1] is _DASH:
        return _parse_list(tokens, i, indent)
    return _parse_map(tokens, i, indent)


def _parse_map(tokens, i: int, indent: int):
    out: Dict[str, Any] = {}
    while i < len(tokens):
        ind, content, lineno = tokens[i]
        if ind < indent:
            break
        if ind > indent:
            raise MiniYamlError(f"第 {lineno} 行的缩进对不上，无法确定它属于哪一层")
        if content is _DASH:
            break
        key, value = _split_key(content, lineno)
        i += 1
        if value:
            out[key] = _scalar(value, lineno)
        else:
            if i < len(tokens) and tokens[i][0] > indent:
                child, i = _parse_block(tokens, i, tokens[i][0])
                out[key] = child
            else:
                out[key] = None
    return out, i


def _parse_list(tokens, i: int, indent: int):
    out: List[Any] = []
    while i < len(tokens):
        ind, content, lineno = tokens[i]
        if ind < indent or content is not _DASH:
            break
        i += 1
        if i < len(tokens) and tokens[i][0] > indent:
            child, i = _parse_block(tokens, i, tokens[i][0])
            out.append(child)
        else:
            out.append(None)
    return out, i


def parse(text: str) -> Any:
    tokens = _tokenize(text)
    if not tokens:
        return {}
    value, i = _parse_block(tokens, 0, tokens[0][0])
    if i < len(tokens):
        raise MiniYamlError(f"第 {tokens[i][2]} 行之后的内容缩进结构异常，无法安全解析")
    return value


# ---- 写出。只用于写回 Suture 自己管理的用户层配置 ----

def _quote(s: str) -> str:
    if s == "":
        return "''"
    if any(c in s for c in ":#{}[]&*|>'\"") or s.strip() != s:
        return "'" + s.replace("'", "''") + "'"
    if s.lower() in ("true", "false", "yes", "no", "null", "~"):
        return f"'{s}'"
    try:
        float(s)
        return f"'{s}'"
    except ValueError:
        return s


def _emit(value: Any, indent: int, lines: List[str]) -> None:
    pad = " " * indent
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{_quote(str(k))}:")
                _emit(v, indent + 2, lines)
            elif isinstance(v, (dict, list)):
                lines.append(f"{pad}{_quote(str(k))}: {'{}' if isinstance(v, dict) else '[]'}")
            else:
                lines.append(f"{pad}{_quote(str(k))}: {_render_scalar(v)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item:
                first = True
                for k, v in item.items():
                    prefix = f"{pad}- " if first else f"{pad}  "
                    first = False
                    if isinstance(v, (dict, list)) and v:
                        lines.append(f"{prefix}{_quote(str(k))}:")
                        _emit(v, indent + 4, lines)
                    else:
                        lines.append(f"{prefix}{_quote(str(k))}: {_render_scalar(v)}")
            else:
                lines.append(f"{pad}- {_render_scalar(item)}")


def _render_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if "\n" in s or "\r" in s:
        # 换行在 YAML 里必须用块标量（| / >）才能表达，而这个读取器不支持块标量。
        # 硬写出去会得到一个自己读不回来的文件——下一次读取会把整份配置判成语法错，
        # 再往后就是"读不懂就不写"，用户彻底卡住。宁可在这里明确报错。
        raise MiniYamlError(
            "要写入的值里有换行符，这个受限的 YAML 写出器无法安全表示它。"
            "多半是复制 Key 时把换行一起粘了进来，请去掉首尾空白后重试。")
    return _quote(s)


def dump(value: Any) -> str:
    lines: List[str] = []
    _emit(value, 0, lines)
    return "\n".join(lines) + "\n"
