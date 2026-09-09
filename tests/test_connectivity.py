"""连通优先语义的补充验收：真实请求目标模型、网关侧不碰配置、
模型只改目标保留其余、external 原因不能假装修好。"""
from __future__ import annotations

import os
import unittest

from tests.helpers import Sandbox, write_json, write_text
from tests.test_flows import cc_settings, make_engine, one_client

from suture import engine as E


class TestE2eUsesConfiguredModel(unittest.TestCase):
    def test_blocked_client_does_not_fall_back_to_probe_model(self):
        """决定性请求必须用客户端自己配的模型试——模型名配错就要被如实标出来，
        不能拿网关保证认识的探活模型去试、把一个坏模型名掩盖成已连通。"""
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "totally-made-up-model",   # 路由表里没有
            }})
            eng = make_engine(sb)
            c = one_client(eng.run(["claude_code"]))
            self.assertEqual(c.state, E.STATE_BLOCKED)
            self.assertFalse(c.e2e["ok"])
            # 最后失败的那次请求是用配置里的坏模型发的，而不是 probe_model
            self.assertEqual(c.e2e["model"], "totally-made-up-model")
            self.assertNotEqual(c.e2e["model"], eng._probe_model())
            self.assertTrue(any(i["id"].startswith("model:") for i in c.issues))


class TestGatewayDownLeavesConfigAlone(unittest.TestCase):
    def test_gateway_side_issue_is_external_and_applying_it_changes_nothing(self):
        """顶层判定网关侧挂了时，每个客户端只给一条 external 的网关侧原因；
        对这条原因执行动作应当原样返回「需在工具外处理」，绝不写配置文件。"""
        with Sandbox(behavior="server_error") as sb:
            path = cc_settings(sb)
            write_json(path, {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url + "/v1",   # 本地确实也有问题
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            before = open(path, encoding="utf-8").read()
            eng = make_engine(sb)
            report = eng.run(["claude_code"])
            self.assertTrue(report.gateway_side_down)
            c = one_client(report)
            self.assertEqual(c.state, E.STATE_BLOCKED)
            self.assertEqual([i["repair_kind"] for i in c.issues], ["external"])
            result = eng.apply_action("claude_code", c.issues[0])
            self.assertEqual(result["result"], E.RESULT_MANUAL)
            self.assertEqual(open(path, encoding="utf-8").read(), before)


class TestModelChoiceKeepsOtherCandidates(unittest.TestCase):
    def test_replacing_one_bad_model_keeps_the_rest(self):
        """DeepSeek 一次注册多个模型：把坏的那一个换成可用的，其它已经注册的
        模型必须原样保留，不能整份清单被顶掉。"""
        with Sandbox() as sb:
            cordis = os.path.join(sb.project, "cordis.yml")
            write_text(cordis,
                       "plugins:\n"
                       "  - name: '@deepseek-ai/dsh-llm-pi-ai'\n"
                       "    config:\n"
                       "      providers:\n"
                       "        tower-ai:\n"
                       "          api: anthropic-messages\n"
                       f"          baseURL: {sb.base_url}\n"
                       "          apiKeyEnv: AI_GATE_API_KEY\n"
                       "          models:\n"
                       "            - id: zzz-model-a\n"
                       "            - id: zzz-model-b\n")
            eng = make_engine(sb, AI_GATE_API_KEY="yotta_pk_valid1234")
            c = one_client(eng.run(["deepseek"]), "deepseek")
            self.assertEqual(c.state, E.STATE_BLOCKED)
            issue = next(i for i in c.issues if i["id"] == "model:zzz-model-a")
            self.assertEqual(issue["repair_kind"], "choice")

            fix_value = next(v for v in issue["choices"] if v["value"].startswith("glm-5.3"))
            self.assertIn("zzz-model-b", fix_value["value"])   # 另一个模型留在清单里
            result = eng.apply_action("deepseek", issue, value=fix_value["value"])
            self.assertEqual(result["result"], E.RESULT_FIXED, result.get("message"))
            self.assertEqual(one_client(eng.run(["deepseek"]), "deepseek").state, E.STATE_CONNECTED)


class TestStateModelsAgainstProfile(unittest.TestCase):
    def test_state_model_list_equals_gateway_profile(self):
        from suture.profile import load_profile
        profile, _ = load_profile()
        expected = {m["id"] for m in profile["models"]}
        with Sandbox() as sb:
            eng = make_engine(sb)
            models = eng.profile.get("models", [])
            self.assertEqual({m["id"] for m in models}, expected)
            self.assertGreaterEqual(len(models), 1)


class TestRecheckSingleClient(unittest.TestCase):
    def test_recheck_returns_fresh_assessment(self):
        with Sandbox() as sb:
            path = cc_settings(sb)
            write_json(path, {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url + "/v1",   # 需要修
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            eng = make_engine(sb)
            before = one_client(eng.run(["claude_code"]))
            self.assertEqual(before.state, E.STATE_BLOCKED)
            # 用户在工具外手动把配置改对了，recheck 应该立刻反映出来
            write_json(path, {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            client = eng.assess_client("claude_code")
            self.assertEqual(client.state, E.STATE_CONNECTED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
