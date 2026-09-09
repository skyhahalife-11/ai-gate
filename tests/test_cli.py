"""命令行的验收：真的以子进程跑一遍，确认输出和退出码。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

from tests.helpers import ROOT, Sandbox, subprocess_env, write_json
from tests.test_flows import cc_settings


def run_cli(sb, *args, env_extra=None):
    env = subprocess_env(HOME=sb.home, PYTHONIOENCODING="utf-8", NO_COLOR="1")
    env.update(env_extra or {})
    # CLI 会真的去跑客户端自证（装了但没配置的客户端）。必须跟引擎层一样
    # 用隔离命令顶掉本机真实客户端，否则结果取决于跑测试这台机器装没装、
    # 登录没登录，而不是取决于被测逻辑。
    env.setdefault("SUTURE_SELFTEST_COMMAND", sb.env.get("SUTURE_SELFTEST_COMMAND", ""))
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "main.py"), "--cli",
         "--profile", sb.profile_path, "--home", sb.home,
         "--project-dir", sb.project, *args],
        cwd=sb.project, env=env, capture_output=True, text=True, encoding="utf-8",
        timeout=90)


class TestCli(unittest.TestCase):
    def test_healthy_config_exits_zero(self):
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "glm-5.3"}})
            r = run_cli(sb, "--yes", "--harness", "claude_code")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("Claude Code CLI", r.stdout)
            self.assertIn("已连接", r.stdout)

    def test_fix_flow_prints_steps_and_succeeds(self):
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url + "/v1",   # 可自动修：多拼 /v1
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "GLM-5.3",                # 可自动修：大小写
            }})
            r = run_cli(sb, "--yes", "--harness", "claude_code")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("已备份原配置", r.stdout)
            after = json.load(open(cc_settings(sb), encoding="utf-8"))["env"]
            self.assertEqual(after["ANTHROPIC_BASE_URL"], sb.base_url)
            self.assertEqual(after["ANTHROPIC_MODEL"], "glm-5.3")

    def test_no_fix_flag_changes_nothing(self):
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url + "/v1",
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "glm-5.3"}})
            before = open(cc_settings(sb), encoding="utf-8").read()
            r = run_cli(sb, "--no-fix", "--harness", "claude_code")
            self.assertEqual(r.returncode, 3, r.stdout)
            self.assertEqual(open(cc_settings(sb), encoding="utf-8").read(), before)

    def test_gateway_down_exit_code_two(self):
        with Sandbox(behavior="server_error") as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "glm-5.3"}})
            r = run_cli(sb, "--yes")
            self.assertEqual(r.returncode, 2, r.stdout)
            self.assertIn("暂时无法连接", r.stdout)

    def test_key_is_masked_in_output(self):
        with Sandbox() as sb:
            write_json(cc_settings(sb), {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url,
                "ANTHROPIC_API_KEY": "yotta_pk_topsecret9",
                "ANTHROPIC_MODEL": "glm-5.3"}})
            r = run_cli(sb, "--yes", "--harness", "claude_code")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertNotIn("topsecret", r.stdout)

    def test_gui_flag_starts_server_without_opening_window(self):
        """--gui 会起本地服务；这里只验证它能起来并给出一次性令牌地址。"""
        with Sandbox() as sb:
            code = (
                "import sys; sys.path.insert(0, %r)\n"
                "from suture.shell import launch\n"
                "launch(profile_path=%r, project_dir=%r, open_ui=False)\n"
            ) % (ROOT, sb.profile_path, sb.project)
            r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, encoding="utf-8", timeout=60,
                               env=subprocess_env(HOME=sb.home))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("http://127.0.0.1:", r.stdout)
            self.assertIn("token=", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
