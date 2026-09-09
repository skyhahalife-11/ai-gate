"""安装执行器的验收：真实跑一条命令、缺 Node 给指引、无命令只给引导、
超时会杀掉整棵树而不是卡死、以及服务端 install_run 接口。"""
from __future__ import annotations

import os
import tempfile
import unittest

from tests.helpers import Sandbox
from tests.test_flows import make_engine

from suture import installer
from suture.harness import get_adapter
from suture.harness.codex import CodexAdapter


def _bare_path_env(tmpdir: str) -> dict:
    """一个干净、PATH 里没有 python/node 的环境，用来测“缺运行时”的引导。"""
    return {"PATH": tmpdir}


class TestRunInstall(unittest.TestCase):
    def test_override_command_runs_and_reports_success(self):
        env = {"PATH": os.environ.get("PATH", ""),
               "SUTURE_INSTALL_CMD_CODEX": "python -u -V"}
        r = installer.run_install(CodexAdapter(), env=env)
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["ran"])
        self.assertIn("Python", r["output"])

    def test_timeout_kills_process_tree(self):
        env = {"PATH": os.environ.get("PATH", ""),
               "SUTURE_INSTALL_CMD_CODEX": 'python -u -c "import time;time.sleep(60)"'}
        r = installer.run_install(CodexAdapter(), env=env, timeout=2.0)
        self.assertFalse(r["ok"])
        self.assertTrue(r["timed_out"])
        self.assertIn("超过", r["message"])

    def test_command_failure_is_reported_not_claimed_success(self):
        env = {"PATH": os.environ.get("PATH", ""),
               "SUTURE_INSTALL_CMD_CODEX": 'python -u -c "import sys;sys.exit(3)"'}
        r = installer.run_install(CodexAdapter(), env=env)
        self.assertFalse(r["ok"])
        self.assertTrue(r["ran"])
        self.assertFalse(r["timed_out"])

    def test_none_command_only_gives_guide(self):
        adapter = get_adapter("deepseek")
        self.assertIsNone(adapter.install_command())
        r = installer.run_install(adapter, env={"PATH": os.environ.get("PATH", "")})
        self.assertFalse(r["ok"])
        self.assertFalse(r["ran"])
        self.assertTrue(r["message"])

    def test_npm_install_without_node_guides_to_nodejs(self):
        with tempfile.TemporaryDirectory() as td:
            env = _bare_path_env(td)
            env["SUTURE_INSTALL_CMD_CLAUDE_CODE"] = "npm install -g @anthropic-ai/claude-code"
            adapter = get_adapter("claude_code")
            r = installer.run_install(adapter, env=env)
            self.assertFalse(r["ok"])
            self.assertFalse(r["ran"])
            self.assertIn("Node.js", r["message"])


class TestServerInstallRun(unittest.TestCase):
    def test_install_run_endpoint_via_server(self):
        import json
        import urllib.error
        import urllib.request
        from suture import server as S

        with Sandbox() as sb:
            eng = make_engine(sb, SUTURE_INSTALL_CMD_CODEX="python -u -V")
            httpd, state, _ = S.serve_in_background(eng)
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            try:
                req = urllib.request.Request(
                    base + "/api/install_run",
                    data=json.dumps({"client_id": "codex"}).encode("utf-8"),
                    headers={"Content-Type": "application/json",
                             "X-Suture-Token": state.token}, method="POST")
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(result["ok"], result)
                self.assertIn("Python", result["output"])

                # 没有安装命令的客户端：不报错，返回引导
                req = urllib.request.Request(
                    base + "/api/install_run",
                    data=json.dumps({"client_id": "deepseek"}).encode("utf-8"),
                    headers={"Content-Type": "application/json",
                             "X-Suture-Token": state.token}, method="POST")
                with urllib.request.urlopen(req, timeout=60) as resp:
                    ds = json.loads(resp.read().decode("utf-8"))
                self.assertFalse(ds["ok"])
                self.assertFalse(ds["ran"])

                # 不存在的客户端拒绝
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(urllib.request.Request(
                        base + "/api/install_run",
                        data=json.dumps({"client_id": "nope"}).encode("utf-8"),
                        headers={"Content-Type": "application/json",
                                 "X-Suture-Token": state.token}, method="POST"),
                        timeout=10)
                self.assertEqual(cm.exception.code, 400)
            finally:
                httpd.shutdown()
                httpd.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
