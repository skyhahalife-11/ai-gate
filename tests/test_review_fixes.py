"""2026-09-09 三路审查后修掉的问题回归验收。

覆盖：
  · R1 环境变量是生效来源、Suture 写不到时，不再给会失效的 auto/conflict 修复，
    只给一条去环境变量处改的外部说明（不再死循环）；
  · R2 型号在快照里但网关实际拒绝（400/404）且地址本来就对时，不再给 base_url
    的 no-op 自动修复（原地循环），给外部说明；
  · R3 codex 在缺 base_url（没发过 e2e）时也立即给 auth-incompatible 外部说明，
    而不是先引导去配一个没用的地址；
  · C5 configure_client 对已有配置的客户端拒绝整文件重配（避免覆盖不留备份）。
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
import unittest

from tests.helpers import Sandbox, real_profile, write_json, write_text
from tests.test_flows import cc_settings, make_engine, one_client

from suture import engine as E
from suture.checks import _e2e_fallback
from suture.harness.base import FIELD_BASE_URL, FIELD_MODEL
from suture.profile import expected_base_url


class TestEnvSourceNoAutoFix(unittest.TestCase):
    """R1：base_url 的实际生效值来自 OS 环境变量（claude 会被 ANTHROPIC_BASE_URL
    压过文件层）时，自动写配置没用——只给一条去环境变量里改的外部说明，
    绝不再给 conflict / fallback 的自动修复（那会复制错误值或原地循环）。
    同一套逻辑现在也覆盖「生效层是组织托管配置」这种只读来源（见
    test_managed_base_url_gets_external_not_auto）。"""

    def test_env_sourced_base_url_gets_external_not_auto(self):
        with Sandbox() as sb:
            # 只有模型写在文件里（保证 has_any_config 成立）；地址来自"环境变量"层且拼错了 /v1
            write_json(cc_settings(sb), {"env": {"ANTHROPIC_MODEL": "glm-5.3"}})
            eng = make_engine(sb,
                              ANTHROPIC_BASE_URL=sb.base_url + "/v1",
                              ANTHROPIC_API_KEY="yotta_pk_valid1234")
            c = one_client(eng.run(["claude_code"]), "claude_code")
            self.assertEqual(c.state, E.STATE_BLOCKED)
            ids = [i["id"] for i in c.issues]
            self.assertIn("readonly-source", ids, ids)
            self.assertNotIn("conflict:base_url", ids)
            for i in c.issues:
                self.assertEqual(i["repair_kind"], "external",
                                 f"{i['id']} 不该给会写配置文件的自动修复：{i}")
                self.assertIn("环境变量", i["detail"])


class TestModelUnroutedNoOpLoop(unittest.TestCase):
    """R2：真实请求 400/404、地址又确实正确时，不给"把地址改成规范地址"的
    no-op 自动修复（那等于把同一个值重写一遍、原地循环），给外部说明。"""

    def test_fallback_avoids_noop_base_url_fix_when_url_already_correct(self):
        profile = real_profile()
        expected = expected_base_url(profile, "claude_code")
        cfg = SimpleNamespace(
            harness_id="claude_code",
            model_candidates=None,
            field=lambda k: SimpleNamespace(is_set=True, value=expected)
            if k == FIELD_BASE_URL else SimpleNamespace(is_set=True, value="glm-5.3"),
        )
        res = _e2e_fallback(cfg, profile, {"status": 404, "classification": "model"})
        self.assertEqual(res[0]["id"], "model-unrouted", res)
        self.assertEqual(res[0]["repair_kind"], "external")


class TestCodexMissingBaseUrlExternal(unittest.TestCase):
    """R3：codex 在只认 Token 头的网关上连缺 base_url（没发过 e2e）也直接给
    auth-incompatible，不让用户先去配一个配了也没用的地址。"""

    def test_codex_without_base_url_on_token_only_is_external(self):
        with Sandbox(behavior="token_only") as sb:
            write_text(os.path.join(sb.home, ".codex", "config.toml"),
                       'model_provider = "ai-gate"\nmodel = "deepseek-v4-flash"\n\n'
                       '[model_providers.ai-gate]\n'
                       'env_key = "AI_GATE_API_KEY"\n')
            eng = make_engine(sb, AI_GATE_API_KEY="yotta_pk_valid1234")
            c = one_client(eng.run(["codex"]), "codex")
            self.assertEqual(c.state, E.STATE_BLOCKED)
            self.assertEqual([i["id"] for i in c.issues], ["auth-incompatible"])
            self.assertEqual(c.issues[0]["repair_kind"], "external")


class TestConfigureRefusesWhenAlreadyConfigured(unittest.TestCase):
    """C5：已有配置的客户端不再被 configure 整文件覆盖——覆盖会冲掉
    permissions/hooks 且不留备份，只该在"装了但一行没配"时提供。"""

    def test_configure_refuses_and_leaves_existing_config_untouched(self):
        with Sandbox() as sb:
            path = cc_settings(sb)
            write_json(path, {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            before = open(path, encoding="utf-8").read()
            eng = make_engine(sb)
            r = eng.configure_client("claude_code", model="glm-5.3",
                                     api_key="yotta_pk_otherkey1")
            self.assertIn("已有一份配置", r["message"])
            self.assertIsNone(r.get("path"))
            self.assertEqual(open(path, encoding="utf-8").read(), before)


class TestNoPlaintextKeyInReport(unittest.TestCase):
    """出网前必须脱敏：issue 的 fix_value 可能是原始 Key（补 Token 头 / 去掉首尾
    空白这两条修复的载荷），report_to_dict 是唯一的出网出口，必须把它抹掉——
    界面从不读这个字段，值由服务端从自己那份 last_report 里取。"""

    def test_fix_value_stripped_from_wire_payload(self):
        raw = "yotta_pk_SECRET_ABCD1234"
        with Sandbox(behavior="token_only") as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": raw,
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            eng = make_engine(sb)
            report = eng.run(["claude_code"])
            client = one_client(report, "claude_code")
            self.assertEqual(client.state, E.STATE_BLOCKED)
            # 服务端自己那份还留着值（apply_action 要用）
            self.assertTrue(any(i.get("fix_value") for i in client.issues))
            blob = json.dumps(E.report_to_dict(report), ensure_ascii=False)
            self.assertNotIn(raw, blob)
            for c in E.report_to_dict(report)["clients"]:
                for i in c["issues"]:
                    self.assertNotIn("fix_value", i)

    def test_gateway_error_text_does_not_echo_key(self):
        """请求头里带换行时 urllib 抛的异常原文含 Key，不能原样外传。"""
        raw = "yotta_pk_valid1234\n"
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": raw,
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            eng = make_engine(sb)
            report = eng.run(["claude_code"])
            blob = json.dumps(E.report_to_dict(report), ensure_ascii=False)
            self.assertNotIn("yotta_pk_valid1234", blob)


class TestUnparseableFileNotOverwritten(unittest.TestCase):
    """文件读不懂时，自动修复必须拒绝写入。

    解析失败后 data 是空的，按它起草再整文件写回 = 用一份只含本次改动的文件
    覆盖原文件，同一份 settings.json 里的 permissions / hooks 会被静默抹掉。"""

    def test_syntax_error_file_is_not_clobbered(self):
        with Sandbox() as sb:
            path = cc_settings(sb)
            # 多一个逗号 → 整个文件解析失败
            write_text(path, json.dumps({
                "env": {"ANTHROPIC_BASE_URL": sb.base_url,
                        "ANTHROPIC_MODEL": "glm-5.3"},
                "permissions": {"allow": ["Bash"]},
                "hooks": {"Stop": []},
            }).replace('"hooks"', ',"hooks"'))
            before = open(path, encoding="utf-8").read()
            eng = make_engine(sb)
            client = one_client(eng.run(["claude_code"]), "claude_code")
            self.assertEqual(client.state, E.STATE_BLOCKED)
            base_issue = next(i for i in client.issues if i["id"] == "base_url")
            # 判定层就不该给「修复」按钮：这个文件写不进去
            self.assertEqual(base_issue["repair_kind"], "external",
                             "文件读不懂时不该提供会覆盖整份文件的自动修复")
            # 就算硬调，也必须拒绝并保持原文件逐字节不变
            r = eng.apply_action("claude_code", {**base_issue, "repair_kind": "auto",
                                                 "fix_field": "base_url",
                                                 "fix_value": sb.base_url})
            self.assertEqual(r["result"], E.RESULT_MANUAL)
            self.assertIn("读不懂", r["message"])
            self.assertEqual(open(path, encoding="utf-8").read(), before)


class TestNonUtf8ConfigDoesNotCrash(unittest.TestCase):
    """记事本存成 ANSI/GBK 的配置文件曾让整轮检查抛 UnicodeDecodeError，
    界面上只显示「检查失败」。现在应当转成可展示的格式错误、照常出结论。"""

    def test_gbk_settings_json_is_reported_not_raised(self):
        with Sandbox() as sb:
            path = cc_settings(sb)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write('{"env": {"ANTHROPIC_MODEL": "中文模型"}}'.encode("gbk"))
            eng = make_engine(sb)
            client = one_client(eng.run(["claude_code"]), "claude_code")   # 不抛异常
            self.assertEqual(client.state, E.STATE_BLOCKED)
            syntax = [i for i in client.issues if i["id"].startswith("syntax:")]
            self.assertTrue(syntax, [i["id"] for i in client.issues])
            self.assertIn("UTF-8", syntax[0]["detail"])

    def test_gbk_codex_config_is_reported_not_raised(self):
        # codex 的 Key 走环境变量、地址来自默认路由，缺了这两样时 base_url 那条
        # 会先短路；这里补一份完整配置，让检查能跑到语法那一项。
        with Sandbox() as sb:
            path = os.path.join(sb.home, ".codex", "config.toml")
            write_text(path, "# placeholder")
            with open(path, "wb") as f:
                content = ('# 中文注释\nmodel_provider = "ai-gate"\nmodel = "glm-5.3"\n\n'
                           "[model_providers.ai-gate]\n"
                           'base_url = "%s"\n'
                           'env_key = "AI_GATE_API_KEY"\n' % sb.base_url)
                f.write(content.encode("gbk"))
            eng = make_engine(sb, AI_GATE_API_KEY="yotta_pk_valid1234")
            client = one_client(eng.run(["codex"]), "codex")
            # codex 在这座网关上走契约不可达短路（只给一条外部说明），所以这里
            # 要验的是「不抛异常」和「这份文件被如实标成读不懂」，不是列表明细。
            self.assertEqual(client.state, E.STATE_BLOCKED)
            user_file = next(f for f in client.files if f["layer"] == "用户级配置")
            self.assertTrue(user_file["exists"])
            self.assertFalse(user_file["parse_ok"],
                             "非 UTF-8 的文件要标成解析失败，而不是让检查崩掉")

    def test_gbk_deepseek_settings_is_reported_not_raised(self):
        with Sandbox() as sb:
            path = os.path.join(sb.home, ".dsh", "settings.yaml")
            write_text(path, "# placeholder")
            with open(path, "wb") as f:
                f.write("# 中文注释\nllm-pi-ai:\n  providers: {}\n".encode("gbk"))
            eng = make_engine(sb)
            client = one_client(eng.run(["deepseek"]), "deepseek")
            self.assertEqual(client.state, E.STATE_BLOCKED)
            self.assertTrue([i for i in client.issues if i["id"].startswith("syntax:")])


class TestLocalRequestErrorNotBlamedOnGateway(unittest.TestCase):
    """Key 带换行时请求在本机就没发出去，不能报成「AI Gate 网关侧连不上，
    联系网关值班人员」——那是把用户指反方向，而且 severity=high 会盖过
    真正能修的那条。"""

    def test_trailing_newline_key_gets_local_not_gateway_side(self):
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234\n",
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            eng = make_engine(sb)
            client = one_client(eng.run(["claude_code"]), "claude_code")
            self.assertEqual(client.state, E.STATE_BLOCKED)
            self.assertEqual(client.e2e["classification"], "local_error")
            ids = [i["id"] for i in client.issues]
            self.assertIn("auth-whitespace", ids)
            gateway_side = [i for i in client.issues
                            if i["id"] in ("gateway-side", "local-request")
                            and "网关" in i["title"]]
            for i in gateway_side:
                self.assertNotIn("值班", i["detail"],
                                 "本地构造失败不该让用户去联系网关值班")


class TestManagedLayerGetsExternalNotAuto(unittest.TestCase):
    """生效值来自组织统一下发的托管配置（只读、优先级最高）时，
    auto 修复写的是被它压过的那一层——点了不生效、原因原样回来。"""

    def test_managed_base_url_gets_external_not_auto(self):
        with Sandbox() as sb:
            managed = os.path.join(sb.home, "managed-settings.json")
            write_json(managed, {"env": {"ANTHROPIC_BASE_URL": sb.base_url + "/v1",
                                         "ANTHROPIC_MODEL": "glm-5.3"}})
            eng = make_engine(sb, SUTURE_MANAGED_SETTINGS_PATH=managed)
            client = one_client(eng.run(["claude_code"]), "claude_code")
            self.assertEqual(client.state, E.STATE_BLOCKED)
            ids = [i["id"] for i in client.issues]
            self.assertIn("readonly-source", ids, ids)
            for i in client.issues:
                self.assertNotEqual(i["repair_kind"], "auto",
                                    f"{i['id']} 不该给写不进生效层的自动修复")


class TestConflictIssueHasNoNoopAutoFix(unittest.TestCase):
    """多层取值不一致那条曾给 auto（写的就是生效层 = 原样重写一遍），
    点了之后同一条原因原样回来，永远清不掉。"""

    def test_conflict_gets_no_auto_fix(self):
        with Sandbox(behavior="token_only") as sb:
            proj_dir = os.path.join(sb.project, ".claude")
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url + "/v1",
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            write_json(os.path.join(proj_dir, "settings.json"),
                       {"env": {"ANTHROPIC_BASE_URL": sb.base_url}})
            eng = make_engine(sb)
            client = one_client(eng.run(["claude_code"]), "claude_code")
            conflicts = [i for i in client.issues if i["id"].startswith("conflict:")]
            self.assertTrue(conflicts, [i["id"] for i in client.issues])
            for c in conflicts:
                self.assertNotEqual(c["repair_kind"], "auto")


if __name__ == "__main__":
    unittest.main(verbosity=2)
