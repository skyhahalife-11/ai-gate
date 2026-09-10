"""与网关的所有交互都在这里。

分类的关键是把「我这边连不上网」「网关自己报错」「鉴权被拒」分开：
前两类才算网关侧问题、不该去动本地配置；鉴权被拒更可能是本地配置的问题，
正是这个工具要抓的场景，不能一并甩锅给网关。
"""
from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional

# 判定为「网关侧问题、不碰本地配置」的分类
GATEWAY_SIDE = {"timeout", "network_error", "rate_limited", "server_error"}


@dataclass
class ProbeResult:
    ok: bool
    classification: str
    status: Optional[int]
    detail: str
    elapsed_ms: int


def _endpoint(base_url: str, suffix: str) -> str:
    return base_url.strip().rstrip("/") + suffix


def _headers(auth_headers: Dict[str, str]) -> dict:
    """auth_headers 是「请求头名字 -> 值」的完整集合，调用方已经按真实客户端
    的行为拼好了（比如 Authorization 要不要加 Bearer 前缀）——这里只管原样带上，
    不再替调用方决定该发哪一个、不发哪一个。真实客户端往往会同时发好几个
    鉴权相关的头，探活/真实请求也应该照样一起发，而不是先替用户挑一个。"""
    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    headers.update(auth_headers or {})
    return headers


def _classify(status: Optional[int], exc: Optional[BaseException]) -> tuple:
    if exc is not None:
        if isinstance(exc, socket.timeout) or isinstance(exc, TimeoutError):
            return "timeout", "请求超时，网关没有在预期时间内响应。"
        if isinstance(exc, urllib.error.URLError):
            # URLError 有两种：连不上对端（网络/网关侧），和地址本身没法用
            # （unknown url type）——后者是本地配置问题，不能算网关侧。
            reason = getattr(exc, "reason", None)
            if isinstance(reason, str) and "url type" in reason:
                return "local_error", "网关地址不完整（缺少 http:// 或 https:// 前缀），请求没有发出去。"
            return "network_error", "无法连接 AI Gate，请检查网络后重试。"
        # 其余（比如 http.client.InvalidURL：请求头里带了换行、地址里有非法字符）
        # 都是"请求在本机就没构造成功"。这类不能笼统归成网关侧——它的报错原文里
        # 还可能带着请求头的值（也就是 Key），所以既不外传原文、也不归错方向。
        if isinstance(exc, http.client.InvalidURL) or isinstance(exc, ValueError):
            return "local_error", "请求没有发出去：本机这里的地址或鉴权值格式不合法，请检查后重试。"
        return "unknown", f"请求出错：{type(exc).__name__}"
    if status is None:
        return "unknown", "未获取到响应状态。"
    if 200 <= status < 300:
        return "ok", "请求正常返回。"
    if status in (401, 403):
        return "auth_error", f"网关返回 {status}，鉴权被拒绝。"
    if status == 429:
        return "rate_limited", f"网关返回 {status}，请求被限流。"
    if 500 <= status < 600:
        return "server_error", f"网关返回 {status}，网关侧发生错误。"
    return "unknown", f"网关返回未归类的状态码 {status}。"


def _send(base_url: str, auth_headers: Dict[str, str], model: str,
          timeout: float, suffix: str = "/v1/messages", style: str = "anthropic") -> ProbeResult:
    payload = {"model": model, "max_tokens": 16,
               "messages": [{"role": "user", "content": "ping"}]}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(_endpoint(base_url, suffix), data=body,
                                 headers=_headers(auth_headers), method="POST")
    started = time.time()
    status, exc = None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            e.read()
        except Exception:
            pass
    except Exception as e:      # noqa: BLE001 —— 网络异常种类多，统一交给分类函数判断
        exc = e
    elapsed = int((time.time() - started) * 1000)
    classification, detail = _classify(status, exc)
    return ProbeResult(ok=(classification == "ok"), classification=classification,
                       status=status, detail=detail, elapsed_ms=elapsed)


def probe_gateway(base_url: str, auth_headers: Dict[str, str],
                  probe_model: str, timeout: float = 8.0,
                  suffix: str = "/v1/messages", style: str = "anthropic") -> ProbeResult:
    """阶段一自检：用轻量模型确认网关这条路本身通不通。
    这个模型只用来验证网关活不活着，跟用户想用哪个模型是两件事。"""
    return _send(base_url, auth_headers, probe_model, timeout, suffix, style)


def send_real_request(base_url: str, auth_headers: Dict[str, str],
                      model: str, timeout: float = 12.0,
                      suffix: str = "/v1/messages", style: str = "anthropic") -> ProbeResult:
    """用用户实际的配置发一次最小真实请求，验证端到端能不能跑通。
    路径和报文形态必须跟这个 harness 真实会发的一致，鉴权也要把这个 harness
    真实会带上的所有请求头都带上，不只挑其中一个。"""
    return _send(base_url, auth_headers, model, timeout, suffix, style)
