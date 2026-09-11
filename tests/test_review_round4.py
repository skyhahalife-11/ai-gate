"""2026-09-10 第四轮审查修掉的问题的回归验收。

这一轮的两类病：
  · 上一轮的「按类修复」本身写出了回归（就地脱敏、层名误判、判定按客户端不按字段）；
  · 写入链路不是原子的——序列化在 open(w) 之后才失败，用户的原文件被清空。
下面按「用户能看到的行为」各测一遍。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from dataclasses import asdict

from tests.helpers import Sandbox, read_bytes, read_text, write_json, write_text
from tests.test_flows import cc_settings, make_engine

from suture import checks, engine as E, fixer, server as S
from suture.harness import get_adapter
from suture.harness.base import FIELD_AUTH, FIELD_BASE_URL


def _deepseek_settings(sb, base_url="https://wrong.example.com", extra=""):
    hh = os.path.join(sb.home, ".dsh")
    os.makedirs(hh, exist_ok=True)
    write_text(os.path.join(hh, "settings.yaml"),
               "version: 1\nllm-pi-ai:\n  providers:\n    tower-ai:\n"
               f"      baseURL: {base_url}\n      models:\n        - id: glm-5.3\n" + extra)
    return hh


class TestRedactionDoesNotEatServerState(unittest.TestCase):
    """修 1：脱敏必须返回副本。

    出网的客户端数据和服务端 last_report 里持有的曾经是同一批 dict。就地 pop
    fix_value 会把服务端那份也抹掉——用户点完一次修复，同一客户端剩下的
    「一键修复」按钮就全拿不到值了。
    """

    def test_server_copy_keeps_fix_value_after_redaction(self):
        with Sandbox() as sb:
            state = S._State(make_engine(sb))
            client = E.ClientState(
                client_id="c", display_name="C", state="blocked", state_label="",
                issues=[{"id": "a", "fix_value": "SECRET_KEY_VALUE"}])
            state.last_report = E.Report(profile_source="p", clients=[])
            holder = type("H", (), {})()
            holder.state = state

            payload = asdict(client)
            S.Handler._record_client(holder, payload)     # 服务端接管这份数据
            out = S._redact_result({"client": payload})   # 同一次响应出网

            kept = state.last_report.clients[0].issues[0].get("fix_value")
            self.assertEqual(kept, "SECRET_KEY_VALUE",
                             "服务端自己那份被脱敏连带抹掉了，剩下的按钮会变死按钮")
            self.assertNotIn("SECRET_KEY_VALUE", json.dumps(out),
                             "出网副本里不该有原始值")

    def test_report_to_dict_does_not_mutate_report(self):
        with Sandbox() as sb:
            eng = make_engine(sb, ANTHROPIC_API_KEY="yotta_pk_valid1234")
            report = eng.run(["claude_code"])
            before = [dict(i) for c in report.clients for i in c.issues]
            E.report_to_dict(report)
            after = [dict(i) for c in report.clients for i in c.issues]
            self.assertEqual(before, after, "report_to_dict 不该改动传入的 report")


class TestEnvLikeLayerScoping(unittest.TestCase):
    """修 2/3：判定层要知道「这次要写的到底是哪个文件」。"""

    def test_key_from_dotenv_still_offers_a_button(self):
        """deepseek 的 Key 来自 .env 时，写 .credentials.yaml 优先级更高、一定生效，
        不该被判成「改不到」。"""
        with Sandbox(behavior="token_only") as sb:
            hh = _deepseek_settings(sb, base_url=sb.base_url)
            write_text(os.path.join(sb.project, ".env"),
                       "AI_GATE_API_KEY=yotta_pk_vendor9999\n")
            write_text(os.path.join(hh, "settings.yaml"),
                       "version: 1\nllm-pi-ai:\n  providers:\n    tower-ai:\n"
                       "      apiKeyEnv: AI_GATE_API_KEY\n"
                       f"      baseURL: {sb.base_url}\n      models:\n        - id: glm-5.3\n")
            cfg = make_engine(sb).read_harness("deepseek")
            self.assertEqual(cfg.field(FIELD_AUTH).source_layer, "项目目录下的 .env")
            self.assertFalse(checks._cannot_write(cfg, [FIELD_AUTH]),
                             ".env 是文件、能覆盖，不该当成写不到的层")

    def test_bad_credentials_file_does_not_block_other_fields(self):
        """凭据文件坏了只影响鉴权——base_url/model 只写 settings.yaml，本该照常能改。"""
        with Sandbox() as sb:
            hh = _deepseek_settings(sb)
            with open(os.path.join(hh, ".credentials.yaml"), "wb") as f:
                f.write("version: 1\nrefs:\n  OTHER: 中文\n".encode("gbk"))
            cfg = make_engine(sb).read_harness("deepseek")
            self.assertFalse(checks._cannot_write(cfg, [FIELD_BASE_URL]),
                             "坏凭据文件不该封掉 base_url 的按钮")
            self.assertTrue(checks._cannot_write(cfg, [FIELD_AUTH]),
                            "auth 确实要写这份坏掉的凭据文件，应当照旧不给按钮")


class TestWriteIsAtomic(unittest.TestCase):
    """修 4：序列化必须在落盘之前完成，失败时原文件逐字节不动。"""

    def test_deepseek_unrepresentable_value_leaves_file_intact(self):
        with Sandbox() as sb:
            p = os.path.join(_deepseek_settings(sb), "settings.yaml")
            before = read_bytes(p)
            eng = make_engine(sb)
            with self.assertRaises(E.RefuseWrite):
                get_adapter("deepseek").apply(
                    eng.read_harness("deepseek"), {FIELD_AUTH: "yotta_pk_ABC\ndef"},
                    env=eng.env, home=sb.home, project_dir=sb.project)
            self.assertEqual(read_bytes(p), before, "原文件被写坏/清空了")

    def test_claude_unencodable_value_leaves_file_intact(self):
        """配置里存着孤立代理字符（半个 emoji 对）时，ensure_ascii=False 写不出去。"""
        with Sandbox() as sb:
            p = cc_settings(sb)
            write_text(p, '{"env": {"ANTHROPIC_BASE_URL": "https://old.example.com"},'
                          ' "note": "\\ud800"}')
            before = read_bytes(p)
            eng = make_engine(sb)
            with self.assertRaises(E.RefuseWrite):
                get_adapter("claude_code").apply(
                    eng.read_harness("claude_code"), {FIELD_BASE_URL: "https://new.example.com"},
                    env=eng.env, home=sb.home, project_dir=sb.project)
            self.assertEqual(read_bytes(p), before, "原文件被写成半截 JSON 了")

    def test_broken_structure_is_refused_not_crashed(self):
        """providers 被写成标量时以前会抛 ValueError（服务端 500 + Python 原文）。"""
        with Sandbox() as sb:
            p = os.path.join(_deepseek_settings(sb), "settings.yaml")
            write_text(p, "version: 1\nllm-pi-ai: 这不是分块\n")
            before = read_bytes(p)
            eng = make_engine(sb)
            with self.assertRaises(E.RefuseWrite):
                get_adapter("deepseek").apply(
                    eng.read_harness("deepseek"), {FIELD_BASE_URL: "https://x/y"},
                    env=eng.env, home=sb.home, project_dir=sb.project)
            self.assertEqual(read_bytes(p), before)


class TestConfigureLeavesNothingHalfDone(unittest.TestCase):
    """修 5/6：一键配置要么配好，要么什么都不留。"""

    def test_key_store_failure_rolls_back_generated_config(self):
        with Sandbox() as sb:
            eng = make_engine(sb)
            model = eng.profile["models"][0]["id"]
            adp = get_adapter("claude_code")
            original = adp.store_key

            def boom(*a, **k):
                raise E.RefuseWrite("测试：凭据文件读不懂")

            adp.store_key = boom
            try:
                res = eng.configure_client("claude_code", model=model,
                                           api_key="yotta_pk_abcdefgh1234")
            finally:
                adp.store_key = original

            self.assertFalse(os.path.exists(cc_settings(sb)),
                             "Key 没存上，却把只有地址没 Key 的半成品配置留下了")
            self.assertIn("放弃", res.get("message", ""))

    def test_unknown_model_writes_nothing(self):
        with Sandbox() as sb:
            eng = make_engine(sb)
            res = eng.configure_client("claude_code", model="this-model-does-not-exist",
                                       api_key="yotta_pk_abcdefgh1234")
            self.assertFalse(os.path.exists(cc_settings(sb)))
            self.assertIn("this-model-does-not-exist", res.get("message", ""))

    def test_empty_model_writes_nothing(self):
        """前端下拉为空（比如 /api/state 失败过）时会发空模型——不能替用户挑一个。"""
        with Sandbox() as sb:
            eng = make_engine(sb)
            eng.configure_client("claude_code", model="", api_key="yotta_pk_abcdefgh1234")
            self.assertFalse(os.path.exists(cc_settings(sb)))

    def test_configure_surfaces_a_note_for_the_ui(self):
        """note 里是「要重启客户端」这类提醒，界面现在会渲染它。"""
        with Sandbox() as sb:
            eng = make_engine(sb)
            model = eng.profile["models"][0]["id"]
            res = eng.configure_client("claude_code", model=model,
                                       api_key="yotta_pk_abcdefgh1234")
            self.assertIn("note", res)
        html = read_text(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "suture", "ui", "index.html"))
        self.assertIn("r.note", html, "界面没有渲染 note，重启提醒会永远看不到")


class TestServerRoutesHaveGuards(unittest.TestCase):
    """修 7/8：/api/state 要有兜底；安装不能并发。"""

    def test_state_returns_json_error_instead_of_dropping_the_connection(self):
        with Sandbox() as sb:
            p = cc_settings(sb)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            # 深到让 json.loads 抛 RecursionError 的嵌套
            with open(p, "w", encoding="utf-8") as f:
                f.write('{"env": ' + "[" * 60000 + "]" * 60000 + "}")
            httpd, state, _ = S.serve_in_background(make_engine(sb))
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:%d/api/state" % httpd.server_address[1],
                    headers={"X-Suture-Token": state.token})
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(req, timeout=30)
                self.assertEqual(cm.exception.code, 500)
                body = json.loads(cm.exception.read())
                self.assertIn("读取本机配置失败", body.get("error", ""))
            finally:
                httpd.shutdown()

    def test_second_install_is_refused_while_one_is_running(self):
        with Sandbox() as sb:
            py = sys.executable.replace("\\", "/")
            eng = make_engine(sb)
            eng.env["SUTURE_INSTALL_CMD_CLAUDE_CODE"] = f'"{py}" -c "import time;time.sleep(3)"'
            httpd, state, _ = S.serve_in_background(eng)
            base = "http://127.0.0.1:%d" % httpd.server_address[1]
            headers = {"X-Suture-Token": state.token, "Content-Type": "application/json"}
            results = {}

            def fire(tag):
                req = urllib.request.Request(
                    base + "/api/install_run",
                    data=json.dumps({"client_id": "claude_code"}).encode(),
                    headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        results[tag] = (r.status, json.loads(r.read()))
                except urllib.error.HTTPError as exc:
                    results[tag] = (exc.code, json.loads(exc.read()))

            try:
                first = threading.Thread(target=fire, args=("a",))
                first.start()
                time.sleep(0.6)          # 让第一个真的进到执行里
                second = threading.Thread(target=fire, args=("b",))
                second.start()
                first.join()
                second.join()
                self.assertEqual(sorted(v[0] for v in results.values()), [200, 409],
                                 "两个安装请求没有被互斥住")
                blocked = next(v[1] for v in results.values() if v[0] == 409)
                # 必须落在 error 这个键上：前端 api() 是 throw new Error(data.error || "请求失败")，
                # 写在 message 里会被当成"没有说明"、换成一句笼统的「请求失败」。
                self.assertIn("已经有一个安装在进行中", blocked.get("error", ""))
            finally:
                httpd.shutdown()


class TestBackupDirectoryIsUnique(unittest.TestCase):
    """修 10：同一秒的两次备份不能落在同一个目录里（否则第一次的内容被覆盖）。"""

    def test_same_second_backups_do_not_overwrite(self):
        with Sandbox() as sb:
            p = cc_settings(sb)
            write_json(p, {"env": {"ANTHROPIC_BASE_URL": "https://ORIGINAL.example.com"}})
            first = fixer.backup_files([p], home=sb.home)
            write_json(p, {"env": {"ANTHROPIC_BASE_URL": "https://CHANGED.example.com"}})
            second = fixer.backup_files([p], home=sb.home)

            self.assertNotEqual(first.directory, second.directory)
            kept = read_text(os.path.join(first.directory, os.listdir(first.directory)[0]))
            self.assertIn("ORIGINAL", kept, "第一份备份被第二次覆盖了，原始内容丢失")


class TestInstallCommandSplitting(unittest.TestCase):
    """Windows 路径不能被 POSIX shlex 吃掉反斜杠。"""

    def test_windows_path_survives(self):
        from suture.installer import split_command
        raw = r'"C:\Program Files\Python312\python.exe" -c "import time"'
        self.assertEqual(split_command(raw),
                         [r"C:\Program Files\Python312\python.exe", "-c", "import time"])

    def test_plain_command_still_splits(self):
        from suture.installer import split_command
        self.assertEqual(split_command("npm install -g @anthropic-ai/claude-code"),
                         ["npm", "install", "-g", "@anthropic-ai/claude-code"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
