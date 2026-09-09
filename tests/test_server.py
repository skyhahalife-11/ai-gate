"""界面接口的验收：令牌保护、各接口行为、界面文件本身。"""
from __future__ import annotations

import json
import os
import unittest
import urllib.error
import urllib.request

from tests.helpers import Sandbox, write_json
from tests.test_flows import cc_settings, make_engine

from suture import server as S


def _get(url, token=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("X-Suture-Token", token)
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, r.read().decode("utf-8")


def _post(url, payload, token=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    if token:
        req.add_header("X-Suture-Token", token)
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


class TestServer(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox().__enter__()
        write_json(cc_settings(self.sb), {"env": {
            "ANTHROPIC_BASE_URL": self.sb.base_url + "/v1",     # 有问题，可修
            "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
            "ANTHROPIC_MODEL": "glm-5.3",
        }})
        self.httpd, self.state, self.url = S.serve_in_background(make_engine(self.sb))
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.token = self.state.token

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.sb.__exit__(None, None, None)

    def test_only_listens_on_loopback(self):
        self.assertEqual(self.httpd.server_address[0], "127.0.0.1")

    def test_requests_without_token_are_refused(self):
        for path in ("/", "/api/state"):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                _get(self.base + path)
            self.assertEqual(cm.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            _post(self.base + "/api/check", {})
        self.assertEqual(cm.exception.code, 403)

    def test_wrong_token_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            _get(self.base + "/api/state", token="not-the-token")
        self.assertEqual(cm.exception.code, 403)

    def test_ui_served_with_token_substituted(self):
        status, html = _get(self.base + "/?token=" + self.token)
        self.assertEqual(status, 200)
        self.assertIn(self.token, html)
        self.assertNotIn("__SUTURE_TOKEN__", html)

    def test_state_lists_all_three_clients_with_meta(self):
        status, raw = _get(self.base + "/api/state", token=self.token)
        state = json.loads(raw)
        self.assertEqual(status, 200)
        ids = [c["client_id"] for c in state["clients"]]
        self.assertEqual(set(ids), {"claude_code", "codex", "deepseek"})
        self.assertTrue(state["models"])
        self.assertEqual(state["model_count"], len(state["models"]))

    def test_check_then_single_action_fix_flow(self):
        _, report = _post(self.base + "/api/check", {}, token=self.token)
        self.assertEqual(report["result"], "blocked")
        client = report["clients"][0]
        self.assertEqual(client["client_id"], "claude_code")
        base_issue = next(i for i in client["issues"] if i["id"] == "base_url")

        _, fixed = _post(self.base + "/api/action",
                         {"client_id": "claude_code", "issue_id": base_issue["id"]},
                         token=self.token)
        self.assertEqual(fixed["result"], "fixed", fixed.get("message"))
        self.assertEqual(fixed["client"]["state"], "connected")
        after = json.load(open(cc_settings(self.sb), encoding="utf-8"))["env"]
        self.assertEqual(after["ANTHROPIC_BASE_URL"], self.sb.base_url)

    def test_action_before_check_is_rejected_clearly(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            _post(self.base + "/api/action",
                  {"client_id": "claude_code", "issue_id": "base_url"},
                  token=self.token)
        self.assertEqual(cm.exception.code, 409)
        self.assertIn("重新检查", json.loads(cm.exception.read())["error"])

    def test_action_with_unknown_issue_is_rejected(self):
        _, _ = _post(self.base + "/api/check", {}, token=self.token)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            _post(self.base + "/api/action",
                  {"client_id": "claude_code", "issue_id": "not-there"},
                  token=self.token)
        self.assertEqual(cm.exception.code, 409)

    def test_key_never_leaves_through_api(self):
        secret = "yotta_pk_valid1234"
        _, report = _post(self.base + "/api/check", {}, token=self.token)
        self.assertNotIn(secret, json.dumps(report, ensure_ascii=False))

    def test_configure_endpoint_builds_fresh_config(self):
        # 这个 sandbox 里 claude_code 已有配置；另起一个干净的测全新用户配置
        with Sandbox() as sb2:
            httpd, state, _ = S.serve_in_background(make_engine(sb2))
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            try:
                _, res = _post(base + "/api/configure",
                               {"client_id": "claude_code", "model": "glm-5.3",
                                "api_key": "yotta_pk_fresh1234"}, token=state.token)
                self.assertEqual(res["client"]["state"], "connected", res.get("message"))
                self.assertIn(".claude", res["path"])
            finally:
                httpd.shutdown()
                httpd.server_close()

    def test_action_after_configure_with_bad_key_is_not_stale(self):
        """全新用户配置连 AI Gate 时把 Key 填错 → 界面就地出现 blocked 原因卡，
        用户接着点那条修复不能因为「还没有过 check」而被 409 挡掉。"""
        with Sandbox() as sb2:
            httpd, state, _ = S.serve_in_background(make_engine(sb2))
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            try:
                _, res = _post(base + "/api/configure",
                               {"client_id": "claude_code", "model": "glm-5.3",
                                "api_key": "sk-fake-not-gateway-key"}, token=state.token)
                self.assertEqual(res["client"]["state"], "blocked")
                auth_issue = next(i for i in res["client"]["issues"] if i["id"] == "auth")
                self.assertEqual(auth_issue["repair_kind"], "input")

                _, fixed = _post(base + "/api/action",
                                 {"client_id": "claude_code", "issue_id": "auth",
                                  "value": "yotta_pk_fresh1234"}, token=state.token)
                self.assertEqual(fixed["result"], "fixed", fixed.get("message"))
                self.assertEqual(fixed["client"]["state"], "connected")
            finally:
                httpd.shutdown()
                httpd.server_close()


class TestUiAsset(unittest.TestCase):
    def test_ui_file_exists_and_has_no_external_dependency(self):
        path = os.path.join(S.UI_DIR, "index.html")
        self.assertTrue(os.path.exists(path))
        html = open(path, encoding="utf-8").read()
        for bad in ("http://", "https://", "cdn.", "<script src"):
            self.assertNotIn(bad, html, f"界面不应该依赖外部资源：{bad}")
        self.assertIn("--accent: #D97757", html)      # Claude 的赤陶主色
        self.assertIn("prefers-color-scheme: dark", html)
        self.assertIn("prefers-reduced-motion", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
