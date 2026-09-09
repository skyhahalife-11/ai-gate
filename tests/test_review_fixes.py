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
    绝不再给 conflict / fallback 的自动修复（那会复制错误值或原地循环）。"""

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
            self.assertIn("env-source", ids, ids)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
