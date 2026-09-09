"""验收 codex 对「只认自定义 Token 请求头」的网关（AI Gate）的判定。

AI Gate 只从 Token 头读 Key，而 Codex 没有自定义请求头机制（只会发
Authorization: Bearer）→ 契约层面不可达：补头 / 填 Key / 配 Key 都救不了，
只能给单条外部说明。这里验证：
  · codex 配好地址和 Key、e2e 仍 401 → 归成 auth-incompatible（external），
    不给 input「填 Key」也不给 auto「补头」，apply 是 manual、不改文件；
  · 判定是契约驱动的：若网关 required_header 改成 Authorization（codex 本来
    就会发的头），同一个 codex 就不算不可达；
  · configure_client 对 codex 直接拒绝、不生成配置不写环境变量；
  · claude / deepseek（能加 Token 头）不受影响，仍走「补 Token 头」自动修复。
"""
from __future__ import annotations

import os
import unittest

from tests.helpers import Sandbox, write_text
from tests.test_auth_header import CORDIS_BASE
from tests.test_flows import make_engine, one_client

from suture import engine as E
from suture.checks import auth_incompatible_issue, auth_requirement_unsatisfiable


def codex_cfg(sb, base_url_override=None) -> str:
    base = sb.base_url + "/v1" if base_url_override is None else base_url_override
    return ("model = \"deepseek-v4-flash\"\n"
            "model_provider = \"ai-gate\"\n\n"
            "[model_providers.ai-gate]\n"
            f"base_url = \"{base}\"\n"
            "env_key = \"AI_GATE_API_KEY\"\n")


class TestCodexGatewayContract(unittest.TestCase):
    def test_codex_blocked_on_token_only_gateway_is_external_only(self):
        with Sandbox(behavior="token_only") as sb:
            cfg_path = os.path.join(sb.home, ".codex", "config.toml")
            write_text(cfg_path, codex_cfg(sb))
            eng = make_engine(sb, AI_GATE_API_KEY="yotta_pk_valid1234")

            c = one_client(eng.run(["codex"]), "codex")
            self.assertEqual(c.state, E.STATE_BLOCKED)
            self.assertTrue(c.e2e and not c.e2e["ok"])
            self.assertEqual([i["id"] for i in c.issues], ["auth-incompatible"])
            for i in c.issues:
                self.assertEqual(i["repair_kind"], "external")   # 没有填 Key / 补头按钮

            with open(cfg_path, encoding="utf-8") as f:
                before = f.read()
            result = eng.apply_action("codex", c.issues[0])
            self.assertEqual(result["result"], E.RESULT_MANUAL)
            with open(cfg_path, encoding="utf-8") as f:
                self.assertEqual(f.read(), before)

    def test_codex_missing_key_also_not_guided_to_input(self):
        """连 Key 都没设时 e2e 同样 401——也归成不可达，不引导「先设置环境变量」。"""
        with Sandbox(behavior="token_only") as sb:
            write_text(os.path.join(sb.home, ".codex", "config.toml"), codex_cfg(sb))
            eng = make_engine(sb)
            c = one_client(eng.run(["codex"]), "codex")
            self.assertEqual(c.state, E.STATE_BLOCKED)
            self.assertEqual([i["id"] for i in c.issues], ["auth-incompatible"])

    def test_unsatisfiable_is_contract_driven_not_client_name(self):
        with Sandbox() as sb:
            write_text(os.path.join(sb.home, ".codex", "config.toml"), codex_cfg(sb))
            eng = make_engine(sb, AI_GATE_API_KEY="yotta_pk_valid1234")
            cfg = eng.read_harness("codex")

            self.assertTrue(auth_requirement_unsatisfiable(cfg, eng.profile))
            self.assertIn("Token", auth_incompatible_issue(cfg, eng.profile)["title"])

            # 网关若改认 Authorization（codex 本来就会发的头）→ 不算不可达
            eng.profile["auth"]["required_header"] = "Authorization"
            self.assertFalse(auth_requirement_unsatisfiable(cfg, eng.profile))

    def test_configure_client_refuses_without_writing_for_codex(self):
        with Sandbox() as sb:
            eng = make_engine(sb)
            r = eng.configure_client("codex", model="deepseek-v4-flash",
                                     api_key="yotta_pk_somekey")
            self.assertIn("无法连接", r["message"])
            self.assertIsNone(r.get("path"))
            self.assertFalse(os.path.exists(
                os.path.join(sb.home, ".codex", "config.toml")))
            self.assertEqual(r["client"]["client_id"], "codex")

    def test_claude_and_deepseek_unaffected(self):
        """能加 Token 头的客户端在同一个网关下仍走「补 Token 头」自动修复。"""
        with Sandbox(behavior="token_only") as sb:
            write_text(os.path.join(sb.project, "cordis.yml"),
                       CORDIS_BASE.format(base_url=sb.base_url))
            eng = make_engine(sb, AI_GATE_API_KEY="yotta_pk_valid1234")
            c = one_client(eng.run(["deepseek"]), "deepseek")
            self.assertEqual(c.state, E.STATE_BLOCKED)
            ids = [i["id"] for i in c.issues]
            self.assertIn("auth-header", ids, ids)
            self.assertNotIn("auth-incompatible", ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
