"""验收「Key 没放进网关要求的请求头」这一类判定的修正。

背景：AI Gate 实际只从 Token 请求头读 Key（x-api-key / Authorization 一律 401）。
之前 401 一律归成「Key 失效 → 重新生成」，对一个 key 有效、只是放错头的用户
是假诊断。这里用 behavior=token_only 的假网关（只认 Token 头）验证：
  · 深挖出真正原因是「没补 Token 头」，给一键修复而不是让用户换 Key；
  · 一键补头后真的能连上；
  · 无关静态问题不会把这条真实主因挤掉；
  · key 本身确实不是网关签发时，不误标成「补头」，仍引导换 Key。
"""
from __future__ import annotations

import os
import unittest

from tests.helpers import Sandbox, write_text
from tests.test_flows import cc_settings, make_engine, one_client

from suture import engine as E
from suture.checks import blocked_reasons
from suture.harness.deepseek import DeepSeekHarnessAdapter

CORDIS_BASE = (
    "plugins:\n"
    "  - name: '@deepseek-ai/dsh-llm-pi-ai'\n"
    "    config:\n"
    "      providers:\n"
    "        tower-ai:\n"
    "          api: anthropic-messages\n"
    "          baseURL: {base_url}\n"
    "          apiKeyEnv: AI_GATE_API_KEY\n"
    "          models:\n"
    "            - id: deepseek-v4-pro\n")


class TestDeepSeekHeaderFix(unittest.TestCase):
    def test_valid_key_missing_token_header_is_auto_fixable_and_connects(self):
        with Sandbox(behavior="token_only") as sb:
            write_text(os.path.join(sb.project, "cordis.yml"),
                       CORDIS_BASE.format(base_url=sb.base_url))
            eng = make_engine(sb, AI_GATE_API_KEY="yotta_pk_valid1234")
            c = one_client(eng.run(["deepseek"]), "deepseek")
            self.assertEqual(c.state, E.STATE_BLOCKED)
            hi = next(i for i in c.issues if i["id"] == "auth-header")
            self.assertEqual(hi["repair_kind"], "auto", hi["detail"])
            self.assertIn("Token", hi["title"])

            result = eng.apply_action("deepseek", hi)
            self.assertEqual(result["result"], E.RESULT_FIXED, result.get("message"))
            self.assertEqual(one_client(eng.run(["deepseek"]), "deepseek").state,
                             E.STATE_CONNECTED)


class TestClaudeHeaderFix(unittest.TestCase):
    def _cc_env(self, sb, key="yotta_pk_valid1234"):
        from tests.helpers import write_json
        write_json(cc_settings(sb), {
            "permission": "whatever",          # 像 permissions 的拼写，会被 unknown-key 检测标出来
            "env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": key,
                "ANTHROPIC_MODEL": "glm-5.3",
            }})

    def test_real_cause_not_crowded_out_by_cosmetic_finding(self):
        with Sandbox(behavior="token_only") as sb:
            self._cc_env(sb)
            eng = make_engine(sb)
            c = one_client(eng.run(["claude_code"]))
            self.assertEqual(c.state, E.STATE_BLOCKED)
            ids = [i["id"] for i in c.issues]
            self.assertIn("auth-header", ids, ids)          # 真主因必须在
            self.assertTrue(any(i.startswith("unknown-key:") for i in ids))
            self.assertEqual(c.issues[0]["id"], "auth-header")   # 且排在最前

            hi = next(i for i in c.issues if i["id"] == "auth-header")
            result = eng.apply_action("claude_code", hi)
            self.assertEqual(result["result"], E.RESULT_FIXED, result.get("message"))
            after = open(cc_settings(sb), encoding="utf-8").read()
            self.assertIn("ANTHROPIC_CUSTOM_HEADERS", after)   # Token 头补进去了

    def test_wrong_key_not_labeled_as_header_problem(self):
        """Key 本身就不是网关签发（sk- 开头）时，主诉是换 Key，不是补请求头。"""
        with Sandbox(behavior="token_only") as sb:
            self._cc_env(sb, key="sk-ant-somebodys-own-key")
            eng = make_engine(sb)
            c = one_client(eng.run(["claude_code"]))
            ids = [i["id"] for i in c.issues]
            self.assertIn("auth", ids, ids)
            self.assertNotIn("auth-header", ids, ids)
            auth_issue = next(i for i in c.issues if i["id"] == "auth")
            self.assertEqual(auth_issue["repair_kind"], "input")


class TestReadSurfacesTokenAsParallelHeader(unittest.TestCase):
    def test_headers_token_is_extra_header_when_primary_from_env(self):
        """同时有 apiKeyEnv(env) 和 headers.Token 时，两者都会被发——
        headers.Token 必须作为平行额外头暴露出来，否则引擎只发主头就永远 401。"""
        with Sandbox() as sb:
            settings = os.path.join(sb.home, ".dsh", "settings.yaml")
            write_text(settings,
                       "llm-pi-ai:\n"
                       "  providers:\n"
                       "    yotta:\n"
                       "      api: anthropic-messages\n"
                       f"      baseURL: {sb.base_url}\n"
                       "      apiKeyEnv: AI_GATE_API_KEY\n"
                       "      headers:\n"
                       "        Token: yotta_pk_headerkey\n"
                       "      models:\n"
                       "        - id: deepseek-v4-pro\n")
            env = dict(sb.env, AI_GATE_API_KEY="yotta_pk_envkey")
            cfg = DeepSeekHarnessAdapter().read(env=env, home=sb.home,
                                                project_dir=sb.project)
            self.assertEqual(cfg.field("auth").value, "yotta_pk_envkey")
            tok = next((e for e in cfg.extra_auth_headers if e.header == "Token"), None)
            self.assertIsNotNone(tok, cfg.extra_auth_headers)
            self.assertEqual(tok.value, "yotta_pk_headerkey")


class TestBlockedReasonsMerge(unittest.TestCase):
    def test_merge_appends_e2e_fallback_when_class_missing(self):
        """构造：静态检查只有一条不相干的 medium，e2e 却是鉴权失败且缺 Token 头——
        blocked_reasons 必须把鉴权类原因也带出来，不能只剩那条无关问题。"""
        with Sandbox(behavior="token_only") as sb:
            write_text(os.path.join(sb.project, "cordis.yml"),
                       CORDIS_BASE.format(base_url=sb.base_url))
            eng = make_engine(sb, AI_GATE_API_KEY="yotta_pk_valid1234")
            c = one_client(eng.run(["deepseek"]), "deepseek")
            # blocked_reasons 直接再算一遍，确认类覆盖逻辑独立于 engine 拼装
            cfg = eng.read_harness("deepseek")
            reasons = blocked_reasons(cfg, eng.profile, c.e2e)
            ids = [i["id"] for i in reasons]
            self.assertIn("auth-header", ids, ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
