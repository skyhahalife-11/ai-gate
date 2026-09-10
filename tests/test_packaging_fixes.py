"""2026-09-10 第三轮（打包前）审查修掉的一批问题的回归验收。

这一轮的教训是「只修了实例、没修同类」：上一轮给 settings.json 加了「读不懂就不写」、
给 base_url/model 加了只读守卫、给 /api/check 加了脱敏——但同类位置全都漏了
（.credentials.yaml、codex 的 OSError 分支、另外两条返回客户端数据的路由、
auth 类字段的按钮）。下面按「类」各测一遍，含边界输入。
"""
from __future__ import annotations

import json
import os
import shutil
import unittest
import urllib.request
from dataclasses import asdict

from tests.helpers import (
    Sandbox, read_bytes, read_text, subprocess_env, write_json, write_text,
)
from tests.test_flows import cc_settings, make_engine, one_client

from suture import engine as E
from suture import installer, server
from suture.harness import get_adapter
from suture.harness import _minimal_yaml as yaml
from suture.harness.base import FIELD_BASE_URL


class TestUnreadableFileIsNeverOverwritten(unittest.TestCase):
    """类一：读不懂就不写。

    上一轮只覆盖了 claude 的 settings.json 和 deepseek 的 settings.yaml，
    漏了同一个 store_key 里先写的 .credentials.yaml，以及 codex 的 OSError 分支。
    """

    def test_deepseek_credentials_not_clobbered_when_unparseable(self):
        """`.credentials.yaml` 里存着别的厂商的凭据，读不懂时绝不能整份覆盖。"""
        cases = {
            "非 UTF-8（GBK 中文）":
                "version: 1\nrefs:\n  OTHER_VENDOR: sk-abc\n  CHINESE: 中文\n".encode("gbk"),
            "流式写法（读取器不支持花括号）":
                b"version: 1\nrefs: {OTHER_VENDOR: sk-abc, CHINESE: sk-def}\n",
            "refs 是列表而不是映射":
                b"version: 1\nrefs:\n  - OTHER_VENDOR\n  - CHINESE\n",
        }
        for name, raw in cases.items():
            with self.subTest(name), Sandbox() as sb:
                hh = os.path.join(sb.home, ".dsh")
                os.makedirs(hh, exist_ok=True)
                creds = os.path.join(hh, ".credentials.yaml")
                with open(creds, "wb") as f:
                    f.write(raw)
                before = read_bytes(creds)

                eng = make_engine(sb)
                cfg = eng.read_harness("deepseek")
                with self.assertRaises(E.RefuseWrite):
                    get_adapter("deepseek").store_key(
                        cfg, "yotta_pk_newkey1234", env=eng.env,
                        home=sb.home, project_dir=sb.project)
                self.assertEqual(read_bytes(creds), before,
                                 f"{name}：文件被改动了")

    def test_deepseek_credentials_still_merge_when_readable(self):
        """能读懂时必须继续合并——不能为了安全把功能砍掉。"""
        with Sandbox() as sb:
            hh = os.path.join(sb.home, ".dsh")
            os.makedirs(hh, exist_ok=True)
            creds = os.path.join(hh, ".credentials.yaml")
            write_text(creds, "version: 1\nrefs:\n  OTHER_VENDOR: sk-abc\n")
            eng = make_engine(sb)
            get_adapter("deepseek").store_key(
                eng.read_harness("deepseek"), "yotta_pk_newkey1234",
                env=eng.env, home=sb.home, project_dir=sb.project)
            after = read_text(creds)
            self.assertIn("OTHER_VENDOR", after)
            self.assertIn("AI_GATE_API_KEY", after)

    def test_codex_apply_refuses_when_target_unreadable(self):
        """codex 的 config.toml 读不出来时，以前会当成「空文件」整份重写。"""
        with Sandbox() as sb:
            p = os.path.join(sb.home, ".codex", "config.toml")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write('model = "中文"\n'.encode("gbk"))     # 真 GBK → 读不出
            before = read_bytes(p)
            eng = make_engine(sb)
            cfg = eng.read_harness("codex")
            with self.assertRaises(E.RefuseWrite):
                get_adapter("codex").apply(cfg, {FIELD_BASE_URL: "https://x/y"},
                                           env=eng.env, home=sb.home,
                                           project_dir=sb.project)
            self.assertEqual(read_bytes(p), before)


class TestYamlWriterRoundTrip(unittest.TestCase):
    """写出器不能产出自己读不回来的文件——那会让下次读取把整份配置判成语法错，
    进而触发「读不懂就不写」，用户彻底卡住。"""

    def test_values_that_cannot_be_represented_raise(self):
        for value in ("yotta_pk_abc\n", "yotta_pk_abc\r\n", "line1\nline2"):
            with self.subTest(value=value):
                with self.assertRaises(yaml.MiniYamlError):
                    yaml.dump({"refs": {"AI_GATE_API_KEY": value}})

    def test_empty_collections_round_trip(self):
        for value in ({"a": {}, "b": []},
                      {"providers": {"r": {"headers": {}, "models": []}}}):
            with self.subTest(value=value):
                self.assertEqual(yaml.parse(yaml.dump(value)), value)


class TestRawKeyNeverReachesTheBrowser(unittest.TestCase):
    """类二：原始 Key 不出网。

    上一轮只在 report_to_dict（/api/check）里脱敏，另外两条返回客户端数据的
    路由照样把 fix_value 原样发出去。
    """

    def test_every_client_response_is_redacted(self):
        with Sandbox(behavior="token_only") as sb:
            raw = "yotta_pk_SECRETABCD"
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": raw,
                "ANTHROPIC_MODEL": "claude-sonnet-5"}})
            eng = make_engine(sb)
            client = one_client(eng.run(["claude_code"]))
            self.assertTrue(any(i.get("fix_value") for i in client.issues),
                            "这条用例要的就是一个带原始 Key 的 issue")

            # 服务端自己那份必须保留原值（点「修复」时要用它）
            self.assertIn(raw, json.dumps(asdict(client), default=str))
            # 出网的两条序列化路径都要脱敏
            self.assertNotIn(raw, json.dumps(E.redact_client(asdict(client)), default=str))
            self.assertNotIn(raw, json.dumps(E.report_to_dict(eng.run(["claude_code"])),
                                             default=str))

            # 再走一遍真正的 HTTP 路由。必须用 serve_in_background——create_server
            # 只绑端口、不起服务线程，请求会一直没人应答（表现就是卡死到超时）。
            httpd, state, _ = server.serve_in_background(eng)
            try:
                base = "http://127.0.0.1:%d" % httpd.server_address[1]
                headers = {"X-Suture-Token": state.token,
                           "Content-Type": "application/json"}

                def post(path, payload):
                    req = urllib.request.Request(base + path,
                                                 data=json.dumps(payload).encode(),
                                                 headers=headers)
                    return json.loads(urllib.request.urlopen(req, timeout=60).read())

                report = post("/api/check", {})
                self.assertNotIn(raw, json.dumps(report))
                card = report["clients"][0]
                issue = next(i for i in card["issues"] if i["id"] == "auth-header")
                action = post("/api/action", {"client_id": card["client_id"],
                                              "issue_id": issue["id"]})
                self.assertNotIn(raw, json.dumps(action))
            finally:
                httpd.shutdown()


class TestNoDeadButtons(unittest.TestCase):
    """类三：不给点了也没用的按钮。

    上一轮只给 base_url / model 加了只读与「写不进去」守卫，auth 类字段漏了。
    """

    def test_auth_from_env_gets_no_dead_button(self):
        """Key 来自系统环境变量时，往配置文件补 Token 头不会生效。"""
        with Sandbox(behavior="token_only") as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_MODEL": "claude-sonnet-5"}})
            eng = make_engine(sb, ANTHROPIC_API_KEY="yotta_pk_valid1234",
                              ANTHROPIC_CUSTOM_HEADERS="X-Other: 1")
            client = one_client(eng.run(["claude_code"]))
            hdr = next((i for i in client.issues if i["id"] == "auth-header"), None)
            self.assertIsNotNone(hdr, [i["id"] for i in client.issues])
            self.assertEqual(hdr["repair_kind"], "external", hdr["detail"])

    def test_unreadable_config_offers_no_action_buttons_at_all(self):
        with Sandbox(behavior="token_only") as sb:
            p = cc_settings(sb)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            write_text(p, "{ 这不是合法 json ")
            eng = make_engine(sb)
            client = one_client(eng.run(["claude_code"]))
            actionable = [i["id"] for i in client.issues
                          if i["repair_kind"] in ("auto", "choice", "input")]
            self.assertEqual(actionable, [], f"不该给按钮：{actionable}")


class TestLocalRequestFailureIsHandled(unittest.TestCase):
    """类四：请求在本机就没构造成功，不能把整轮检查打崩。"""

    def test_schemeless_base_url_does_not_crash_the_check(self):
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": "tower-ai.yottastudios.com/zi/proxy",
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "claude-sonnet-5"}})
            eng = make_engine(sb)
            client = one_client(eng.run(["claude_code"]))     # 不该抛出去
            self.assertEqual(client.state, E.STATE_BLOCKED)
            self.assertIn("base_url", [i["id"] for i in client.issues])

    def test_action_payload_strips_whitespace_on_auth(self):
        """auto/choice 写鉴权字段时也要 strip——这个载荷可能正是「Key 末尾有换行」
        那个病本身，原样写进去等于把病原封不动搬进新文件。"""
        with Sandbox(behavior="token_only") as sb:
            hdr_env = "Token: yotta_pk_valid1234 "
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_CUSTOM_HEADERS": hdr_env,
                "ANTHROPIC_MODEL": "claude-sonnet-5"}})
            eng = make_engine(sb)
            result = eng.apply_action("claude_code", {
                "id": "auth-header", "repair_kind": "auto",
                "fix_field": "extra_auth_header", "fix_value": "yotta_pk_valid1234 "})
            self.assertEqual(result["result"], E.RESULT_FIXED, result.get("message"))
            written = read_text(cc_settings(sb))
            self.assertNotIn("yotta_pk_valid1234 ", written.replace("\\n", ""))


class TestNpmIsResolvedBeforeExec(unittest.TestCase):
    """类五：Windows 上 npm 只有 npm.cmd，直接 Popen(["npm", ...]) 必抛 WinError 2。
    check_runtime 用 shutil.which 找得到它，执行时却找不到——两边口径必须一致。

    这里用 SUTURE_INSTALL_CMD_* 换成 `npm --version` 来验证「真的能执行起来」：
    它同样经过 cmd[0] 的解析，但不会真的往这台机器上装东西。
    （真跑 `npm install -g` 会让测试变成"有副作用、还依赖网络"。）"""

    def test_install_actually_starts(self):
        if shutil.which("npm") is None:
            self.skipTest("这台机器上没有 npm")
        # 环境用 subprocess_env 拼：npm 是 Node 的启动器，Node 起来时要能拿到
        # SystemRoot/TEMP 这些系统变量，只给 PATH 的话 Node 会在初始化加密随机数
        # 时直接断言崩溃（ncrypto::CSPRNG），看起来像"安装命令跑不起来"，其实
        # 是测试自己把环境剥得太干净。真实运行走的是 os.environ，不会有这个问题。
        env = subprocess_env(SUTURE_INSTALL_CMD_CLAUDE_CODE="npm --version")
        res = installer.run_install(get_adapter("claude_code"), env=env, timeout=90)
        self.assertTrue(res["ran"], res["message"])
        self.assertTrue(res["ok"], res["message"] + res["output"])
        self.assertNotIn("WinError", res["message"])
        self.assertNotIn("系统找不到指定的文件", res["message"])
        self.assertIn(".", res["output"].strip().splitlines()[0])   # 版本号

    def test_missing_binary_gives_a_chinese_message(self):
        """命令行本身就不存在时，不能把 WinError 2 抛给用户，要给能照做的中文指引。

        这里用一个不存在的命令名，而不是把 PATH 置空——PATH 的兜底逻辑是
        「没给 PATH 就用进程自己的」，传空串会被兜底掉，测不出这条分支。"""
        with Sandbox() as sb:
            env = dict(sb.env,
                       SUTURE_INSTALL_CMD_CLAUDE_CODE="suture-no-such-binary-xyz --version")
            res = installer.run_install(get_adapter("claude_code"), env=env, timeout=30)
            self.assertFalse(res["ok"])
            self.assertFalse(res["ran"])
            self.assertIn("suture-no-such-binary-xyz", res["message"])
            self.assertIn("nodejs.org", res["message"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
