"""异常路径的验收：写不进去、规则文件坏了这类情况必须如实报告，不能装作成功。"""
from __future__ import annotations

import json
import os
import stat
import unittest

from tests.helpers import Sandbox, write_json, write_text
from tests.test_flows import cc_settings, make_engine

from suture import engine as E
from suture.profile import load_profile


class TestWriteFailure(unittest.TestCase):
    def test_unwritable_config_is_reported_not_claimed_fixed(self):
        """权限不足写不进去时，必须如实说写入失败，
        不能显示「修复成功」但其实什么都没改。"""
        if os.name == "nt":
            self.skipTest("Windows 没有 POSIX 权限位，os.chmod 无法制造真实的写入失败")
        if os.geteuid() == 0:
            self.skipTest("以 root 运行时文件权限不起作用，这条在普通用户下才有意义")
        with Sandbox() as sb:
            path = cc_settings(sb)
            write_json(path, {"env": {
                "ANTHROPIC_BASE_URL": sb.base_url + "/v1",
                "ANTHROPIC_API_KEY": "yotta_pk_valid1234",
                "ANTHROPIC_MODEL": "glm-5.3"}})
            before = open(path, encoding="utf-8").read()
            # 目录只读挡不住改写已存在的文件，要把文件本身设成只读才是真实的写入失败
            os.chmod(path, stat.S_IRUSR)
            try:
                eng = make_engine(sb)
                client = eng.run(["claude_code"]).clients[0]
                issue = next(i for i in client.issues if i["id"] == "base_url")
                result = eng.apply_action("claude_code", issue)
                self.assertNotEqual(result["result"], E.RESULT_FIXED)
                self.assertIn("配置写入失败", result["message"])
                self.assertEqual(open(path, encoding="utf-8").read(), before)
            finally:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    def test_engine_surfaces_write_failure_path_exists(self):
        """至少要保证这条分支在代码里是真的接上了的（root 下也能验证）。"""
        import inspect
        src = inspect.getsource(E.Engine.apply_action)
        self.assertIn("配置写入失败", src)
        self.assertIn("恢复原配置", src)


class TestProfileHandling(unittest.TestCase):
    def test_bundled_profile_is_valid_and_complete(self):
        profile, source = load_profile()
        self.assertIn("内置", source)
        for key in ("base_url", "auth", "models", "probe_model", "per_harness_request"):
            self.assertIn(key, profile)
        ids = [m["id"] for m in profile["models"]]
        self.assertEqual(len(ids), len(set(ids)), "路由表里有重复型号")
        self.assertIn(profile["probe_model"], ids, "探活用的模型必须在路由表里")
        for h in ("claude_code", "codex", "deepseek"):
            self.assertIn(h, profile["base_url"]["per_harness_suffix"])
            self.assertIn(h, profile["per_harness_request"])

    def test_broken_profile_fails_loudly(self):
        with Sandbox() as sb:
            bad = os.path.join(sb.project, "bad.json")
            write_text(bad, "{ not json")
            with self.assertRaises(json.JSONDecodeError):
                load_profile(bad)


class TestRollbackRemovesFilesItCreated(unittest.TestCase):
    def test_generated_file_is_removed_on_rollback(self):
        """修复时新建的文件，回滚就该删掉，不能留一个半成品在用户机器上。"""
        from suture import fixer
        with Sandbox() as sb:
            target = os.path.join(sb.project, "new.json")
            manifest = fixer.backup_files([target], home=sb.home)
            write_text(target, "{}")
            self.assertTrue(os.path.exists(target))
            self.assertEqual(fixer.rollback(manifest), [])
            self.assertFalse(os.path.exists(target))


if __name__ == "__main__":
    unittest.main(verbosity=2)
