"""端到端验收：连通优先语义下，客户端四态 + 失败原因的单条修复。"""
from __future__ import annotations

import json
import os
import unittest

from tests.helpers import Sandbox, write_json, write_text

from suture import engine as E
from suture.engine import Engine


def cc_settings(sb) -> str:
    return os.path.join(sb.home, ".claude", "settings.json")


def make_engine(sb, **env_extra) -> Engine:
    return Engine(profile_path=sb.profile_path, env=dict(sb.env, **env_extra),
                  home=sb.home, project_dir=sb.project)


def one_client(report, client_id="claude_code"):
    return next(c for c in report.clients if c.client_id == client_id)


class TestHealthyPath(unittest.TestCase):
    def test_correct_config_is_connected_and_shows_nothing(self):
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            report = make_engine(sb).run(["claude_code"])
            self.assertEqual(report.result, E.RESULT_HEALTHY)
            c = one_client(report)
            self.assertEqual(c.state, E.STATE_CONNECTED)
            self.assertEqual(c.state_label, "可以连接")
            self.assertTrue(c.e2e["ok"])
            self.assertEqual(c.issues, [])          # 成功路径不罗列任何东西


class TestConnectivityFirst(unittest.TestCase):
    def test_connected_ignores_minor_static_issues(self):
        """连通优先：地址/Key/模型都能连，即使有无关紧要的静态问题（比如 settings.json
        里多了一个工具认不得的顶层字段）也不报——通就是通。"""
        with Sandbox() as sb:
            write_json(cc_settings(sb), {
                "bogusUnknownKey": "whatever",      # 会被 unknown-key 类检测标出来
                "env": {
                    "ANTHROPIC_BASE_URL": sb.base_url,
                    "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                    "ANTHROPIC_MODEL": "glm-5.3",
                }})
            report = make_engine(sb).run(["claude_code"])
            self.assertEqual(report.result, E.RESULT_HEALTHY)
            c = one_client(report)
            self.assertEqual(c.state, E.STATE_CONNECTED)
            self.assertEqual(c.issues, [])


class TestActionFix(unittest.TestCase):
    def test_auto_fix_base_url_via_single_action(self):
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url + "/v1",   # 多拼 /v1
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            eng = make_engine(sb)
            report = eng.run(["claude_code"])
            c = one_client(report)
            self.assertEqual(c.state, E.STATE_BLOCKED)
            self.assertFalse(c.e2e["ok"])
            base_issue = next(i for i in c.issues if i["id"] == "base_url")
            self.assertEqual(base_issue["repair_kind"], "auto")

            result = eng.apply_action("claude_code", base_issue)
            self.assertEqual(result["result"], E.RESULT_FIXED, result.get("message"))
            self.assertTrue(result["backup_dir"])
            after = json.load(open(cc_settings(sb), encoding="utf-8"))["env"]
            self.assertEqual(after["ANTHROPIC_BASE_URL"], sb.base_url)
            self.assertTrue(os.path.isdir(result["backup_dir"]))

    def test_auto_fix_model_typo_via_single_action(self):
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "GLM-5.3",            # 大小写不对
            }})
            eng = make_engine(sb)
            c = one_client(eng.run(["claude_code"]))
            model_issue = next(i for i in c.issues if i["id"] == "model:GLM-5.3")
            self.assertEqual(model_issue["repair_kind"], "auto")

            result = eng.apply_action("claude_code", model_issue)
            self.assertEqual(result["result"], E.RESULT_FIXED, result.get("message"))
            after = json.load(open(cc_settings(sb), encoding="utf-8"))["env"]
            self.assertEqual(after["ANTHROPIC_MODEL"], "glm-5.3")

    def test_choice_fix_when_model_not_supported(self):
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "totally-made-up",
            }})
            eng = make_engine(sb)
            c = one_client(eng.run(["claude_code"]))
            model_issue = next(i for i in c.issues if i["id"] == "model:totally-made-up")
            self.assertEqual(model_issue["repair_kind"], "choice")
            self.assertTrue(model_issue["choices"])

            chosen = next(c_ for c_ in model_issue["choices"] if c_["value"] == "glm-5.3")
            result = eng.apply_action("claude_code", model_issue, value=chosen["value"])
            self.assertEqual(result["result"], E.RESULT_FIXED, result.get("message"))
            after = json.load(open(cc_settings(sb), encoding="utf-8"))["env"]
            self.assertEqual(after["ANTHROPIC_MODEL"], "glm-5.3")

    def test_store_key_input_restores_connection(self):
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": "sk-ant-somebodys-own-key",  # 不是网关 Key
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            eng = make_engine(sb)
            c = one_client(eng.run(["claude_code"]))
            auth_issue = next(i for i in c.issues if i["id"] == "auth")
            self.assertEqual(auth_issue["repair_kind"], "input")

            result = eng.apply_action("claude_code", auth_issue, value="yotta_pk_fresh1234")
            self.assertEqual(result["result"], E.RESULT_FIXED, result.get("message"))
            after = json.load(open(cc_settings(sb), encoding="utf-8"))["env"]
            self.assertEqual(after["ANTHROPIC_API_KEY"], "yotta_pk_fresh1234")
            self.assertEqual(one_client(eng.run(["claude_code"])).state, E.STATE_CONNECTED)


class TestGatewaySide(unittest.TestCase):
    def test_gateway_side_failure_stops_before_touching_config(self):
        for behavior in ("server_error", "rate_limited"):
            with self.subTest(behavior=behavior), Sandbox(behavior=behavior) as sb:
                write_json(cc_settings(sb), {"env": {
                    "ANTHROPIC_BASE_URL": sb.base_url + "/v1",   # 本地确实有问题
                    "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                    "ANTHROPIC_MODEL": "glm-5.3",
                }})
                before = open(cc_settings(sb), encoding="utf-8").read()
                report = make_engine(sb).run(["claude_code"])
                self.assertEqual(report.result, E.RESULT_GATEWAY_DOWN)
                self.assertTrue(report.gateway_side_down)
                self.assertEqual(open(cc_settings(sb), encoding="utf-8").read(), before)

    def test_auth_error_is_not_gateway_side(self):
        """401 更可能是本地 Key 配错了，正是工具要抓的场景，不能甩锅给网关。"""
        with Sandbox(behavior="auth_error") as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": "sk-ant-somebodys-own-key",
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            report = make_engine(sb).run(["claude_code"])
            self.assertFalse(report.gateway_side_down)
            c = one_client(report)
            self.assertEqual(c.state, E.STATE_BLOCKED)
            self.assertTrue(any(i["id"] == "auth" for i in c.issues))


class TestWrongLocalAddressNotGateway(unittest.TestCase):
    def test_wrong_address_is_blocked_not_gateway_side(self):
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:9/nonexistent",
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            report = make_engine(sb).run(["claude_code"])
            self.assertFalse(report.gateway_side_down)
            self.assertEqual(report.gateway["classification"], "ok")   # 网关本身是好的
            c = one_client(report)
            self.assertEqual(c.state, E.STATE_BLOCKED)
            self.assertTrue(any(i["id"] == "base_url" for i in c.issues))


class TestFreshUserConfigure(unittest.TestCase):
    def test_configure_client_end_to_end(self):
        with Sandbox() as sb:
            eng = make_engine(sb)
            report = eng.run(["claude_code"])
            self.assertIn(one_client(report).state, (E.STATE_NOT_INSTALLED, E.STATE_UNCONFIGURED))

            result = eng.configure_client("claude_code", model="glm-5.3",
                                          api_key="yotta_pk_fresh1234")
            path = result["path"]
            self.assertTrue(os.path.exists(path))
            text = open(path, encoding="utf-8").read()
            self.assertIn(sb.base_url, text)
            self.assertIn("glm-5.3", text)
            self.assertEqual(result["client"]["state"], E.STATE_CONNECTED,
                             result.get("message"))


class TestCodexFlow(unittest.TestCase):
    def test_fix_writes_user_layer_when_project_untrusted(self):
        with Sandbox() as sb:
            user_cfg = os.path.join(sb.home, ".codex", "config.toml")
            write_text(user_cfg,
                       'model = "glm-5.3"\nmodel_provider = "ai-gate"\n\n'
                       '[model_providers.ai-gate]\n'
                       f'base_url = "{sb.base_url}"\n'          # 少了 /v1
                       'env_key = "AI_GATE_API_KEY"\n')
            project_cfg = os.path.join(sb.project, ".codex", "config.toml")
            write_text(project_cfg, 'model = "kimi-k3"\n')
            before_project = open(project_cfg, encoding="utf-8").read()

            eng = make_engine(sb, AI_GATE_API_KEY="yotta_pk_valid1234")
            c = one_client(eng.run(["codex"]), "codex")
            self.assertEqual(c.state, E.STATE_BLOCKED)
            base_issue = next(i for i in c.issues if i["id"] == "base_url")

            result = eng.apply_action("codex", base_issue)
            self.assertEqual(result["result"], E.RESULT_FIXED, result.get("message"))
            self.assertEqual(open(project_cfg, encoding="utf-8").read(), before_project)
            user_text = open(user_cfg, encoding="utf-8").read()
            self.assertIn(sb.base_url + "/v1", user_text)


class TestDeepSeekFlow(unittest.TestCase):
    def test_connected_with_one_valid_model_among_registered(self):
        with Sandbox() as sb:
            write_text(os.path.join(sb.project, "cordis.yml"),
                       "plugins:\n"
                       "  - name: '@deepseek-ai/dsh-llm-pi-ai'\n"
                       "    config:\n"
                       "      providers:\n"
                       "        tower-ai:\n"
                       "          api: anthropic-messages\n"
                       f"          baseURL: {sb.base_url}\n"
                       "          apiKeyEnv: AI_GATE_API_KEY\n"
                       "          models:\n"
                       "            - id: glm-5.3\n"
                       "            - id: not-a-real-model\n")
            eng = make_engine(sb, AI_GATE_API_KEY="yotta_pk_valid1234")
            c = one_client(eng.run(["deepseek"]), "deepseek")
            self.assertEqual(c.state, E.STATE_CONNECTED)   # 有一个真实能用的模型即可
            self.assertEqual(c.e2e["model"], "glm-5.3")
            self.assertTrue(c.e2e["ok"])


class TestSecrecy(unittest.TestCase):
    def test_key_never_appears_in_report(self):
        secret = "yotta_pk_supersecret"
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": secret,
                "ANTHROPIC_MODEL": "glm-5.3"}})
            report = make_engine(sb).run(["claude_code"])
            blob = json.dumps(E.report_to_dict(report), ensure_ascii=False, default=str)
            self.assertNotIn(secret, blob)
            self.assertNotIn("supersecret", blob)


if __name__ == "__main__":
    unittest.main(verbosity=2)
