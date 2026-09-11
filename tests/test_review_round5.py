"""第五轮审查（CLI / 打包路径）的回归用例。

这一轮的共同点是：**CLI 一条 try 都没有**。界面那边每条路由都有中文 500 兜底，
命令行这边直接把英文 Python 栈甩给用户——而 README 把 CLI 当作"只检测不修改"的
安全入口推荐给非技术同事（check-only.bat 那类玩法）。所以这里的用例都按
「用户会看到什么」写：不是断言某个函数返回什么，而是断言**没有任何异常逃出去**、
以及逃不出去的时候给的是不是中文。

覆盖：
  ① 解析错误原文里夹带明文 Key（安全）
  ② 规则文件读不出来 → 中文而不是 FileNotFoundError / JSONDecodeError
  ③ 备份失败 → 中文而不是 Python 栈
  ④ stdin 不可用 → 中文而不是 EOFError
  ⑤ 修复全部失败 → 不能说「已改完」
  ⑧ 变量名不能当成凭据写进请求头
  ⑨ generate_minimal_config 必须原子写
  ⑩ e2e 候选不能被截断
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.helpers import (                                            # noqa: E402
    Sandbox, read_text, subprocess_env, write_json, write_text,
)
from tests.test_flows import make_engine, one_client                   # noqa: E402
from suture import cli, engine as E                                    # noqa: E402
from suture.harness import get_adapter                                 # noqa: E402
from suture.harness.base import FIELD_AUTH, FIELD_AUTH_ENV             # noqa: E402

SECRET = "yotta_pk_SUPERSECRET9999"


def _run_cli(sb, argv, stdin_devnull=False):
    """在**洗干净的子进程**里跑命令行，返回 (退出码, stdout, 逃出来的异常描述或 None)。

    必须走子进程：`cli.run` 的 Engine 不接 env，用的是 `os.environ`，而开发机上
    通常真实存在 ANTHROPIC_* —— 那是优先级更高的一层，会把沙箱里那份配置整个压掉，
    于是客户端直接判成"已连接"、一条修复都不走，测试看着绿其实什么都没测。
    这里连 stdin 也一并隔离：CLI 的交互分支只有真的没有输入时才会走到。

    异常**不吞**：这一轮每个缺陷都是"异常逃到了用户面前"，吞掉等于把要测的东西
    测没了——所以断言的是 stderr 里不许出现 Traceback。"""
    env = subprocess_env()
    env.update(HOME=sb.home, USERPROFILE=sb.home, PYTHONUTF8="1", PYTHONPATH=ROOT)
    code = "import sys; from suture import cli; sys.exit(cli.run(sys.argv[1:]))"
    proc = subprocess.run(
        [sys.executable, "-c", code] + argv, capture_output=True, text=True,
        env=env, timeout=180,
        stdin=subprocess.DEVNULL if stdin_devnull else None)
    crash = proc.stderr.strip() if "Traceback" in proc.stderr else None
    return proc.returncode, proc.stdout, crash


def _blocked_claude(sb):
    """造一个"连不上、且有可自动修项"的客户端。

    地址少了 https:// —— 这一类在判定层有专门的中文说明，且会给「一键修复」，
    正好走到备份 / 写配置那条链路。"""
    write_json(os.path.join(sb.home, ".claude", "settings.json"),
               {"env": {"ANTHROPIC_BASE_URL": sb.base_url.replace("http://", ""),
                        "ANTHROPIC_API_KEY": "yotta_pk_valid1234"}})


class TestParseErrorDoesNotEchoSecrets(unittest.TestCase):
    """① 读文件失败时把出错那一行抄进说明，等于把 Key 明文印到界面和命令行上。

    README 的承诺是「界面、日志、接口返回里一律只出现掩码后的形式」。这条走的是
    detail → issue → CLI 打印 / 界面渲染，全程没有任何一处做脱敏（redact_client
    只剥 fix_value），所以源头就不能回声。"""

    def test_yaml_error_does_not_echo_the_line(self):
        from suture.harness import _minimal_yaml as y
        with self.assertRaises(y.MiniYamlError) as ctx:
            y.parse("version: 1\nllm-pi-ai:\n  Token:" + SECRET + "\n")
        self.assertNotIn(SECRET, str(ctx.exception))
        # 行号还得留着，否则用户不知道该去看哪一行
        self.assertIn("第 3 行", str(ctx.exception))

    def test_cli_output_never_contains_the_raw_key(self):
        with Sandbox(behavior="token_only") as sb:
            write_text(os.path.join(sb.home, ".dsh", "settings.yaml"),
                       "version: 1\nllm-pi-ai:\n  providers:\n    tower-ai:\n"
                       f"      baseURL: {sb.base_url}\n"
                       f"      Token:{SECRET}\n")          # 冒号后故意少一个空格
            eng = make_engine(sb)
            report = eng.run(["deepseek"])
            blob = json.dumps(E.report_to_dict(report, secrets=eng._known_secrets()),
                              ensure_ascii=False, default=str)
            self.assertNotIn(SECRET, blob)

    def test_redact_scrubs_free_text_not_just_fix_value(self):
        """纵深防御：就算将来又有人在 detail 里拼了原文，出网前也该被擦掉。"""
        payload = {
            "issues": [{"id": "x", "title": "配置读不懂",
                        "detail": f"这一行是 Token:{SECRET}",
                        "fix_value": SECRET, "current_value": SECRET}],
        }
        clean = E.redact_client(payload, secrets=[SECRET])
        text = json.dumps(clean, ensure_ascii=False)
        self.assertNotIn(SECRET, text)
        self.assertNotIn("fix_value", clean["issues"][0])     # 载荷照旧整条删掉
        # 服务端自己那份不能被就地改掉（第四轮踩过的坑）
        self.assertIn(SECRET, payload["issues"][0]["fix_value"])


class TestCliNeverDumpsAPythonStack(unittest.TestCase):
    """②③④ CLI 是全仓唯一没有异常兜底的入口。"""

    def test_missing_profile_gives_chinese(self):
        with Sandbox() as sb:
            code, out, crash = _run_cli(sb, ["--profile", os.path.join(sb.home, "nope.json"),
                                             "--home", sb.home, "--no-fix"])
            self.assertIsNone(crash, f"异常逃到了用户面前：{crash}")
            self.assertNotEqual(code, 0)

    def test_broken_profile_gives_chinese(self):
        with Sandbox() as sb:
            bad = os.path.join(sb.home, "bad.json")
            write_text(bad, "{not json")
            code, out, crash = _run_cli(sb, ["--profile", bad, "--home", sb.home, "--no-fix"])
            self.assertIsNone(crash, f"异常逃到了用户面前：{crash}")

    def test_backup_failure_gives_chinese(self):
        """备份是写入链路的第一步，它失败时用户原文件没被动过——但得说人话。"""
        with Sandbox(behavior="token_only") as sb:
            _blocked_claude(sb)
            # 把 ~/.suture 占成普通文件 → backup_files 的 makedirs 必抛
            with open(os.path.join(sb.home, ".suture"), "w", encoding="utf-8") as f:
                f.write("占位")
            code, out, crash = _run_cli(sb, ["--profile", sb.profile_path, "--home", sb.home,
                                             "--harness", "claude_code", "--yes"])
            self.assertIsNone(crash, f"异常逃到了用户面前：{crash}")
            self.assertIn("备份", out)

    def test_backup_failure_does_not_touch_the_config(self):
        with Sandbox(behavior="token_only") as sb:
            _blocked_claude(sb)
            path = os.path.join(sb.home, ".claude", "settings.json")
            before = read_text(path)
            with open(os.path.join(sb.home, ".suture"), "w", encoding="utf-8") as f:
                f.write("占位")
            _run_cli(sb, ["--profile", sb.profile_path, "--home", sb.home,
                          "--harness", "claude_code", "--yes"])
            self.assertEqual(read_text(path), before, "备份都没做成，配置却被改了")

    def test_no_stdin_does_not_crash(self):
        """check-only.bat 那类用法：读一遍看结论，不该因为"要问一句"就崩掉。"""
        with Sandbox(behavior="token_only") as sb:
            _blocked_claude(sb)
            code, out, crash = _run_cli(sb, ["--profile", sb.profile_path, "--home", sb.home,
                                             "--harness", "claude_code"],
                                        stdin_devnull=True)
            self.assertIsNone(crash, f"异常逃到了用户面前：{crash}")
            self.assertIn("没有可用的输入", out)

    def test_all_repairs_failing_does_not_claim_success(self):
        """⑤ 修复全失败还打印「已按可自动修复的原因改完」，跟上面刚说的失败自相矛盾。"""
        with Sandbox(behavior="token_only") as sb:
            _blocked_claude(sb)
            path = os.path.join(sb.home, ".claude", "settings.json")
            os.chmod(path, 0o444)         # 只读 → 写入必失败
            try:
                code, out, crash = _run_cli(sb, ["--profile", sb.profile_path, "--home", sb.home,
                                                 "--harness", "claude_code", "--yes"])
            finally:
                os.chmod(path, 0o666)
            self.assertIsNone(crash, f"异常逃到了用户面前：{crash}")
            self.assertNotIn("已按可自动修复的原因改完", out)


class TestVariableNameIsNotACredential(unittest.TestCase):
    """⑧ 间接引用式鉴权里，"变量名缺失"要补的是**引用**，不是把名字当 Key 用。

    混用的后果是实测过的：生成的 settings.yaml 里同时出现
        apiKeyEnv: AI_GATE_API_KEY
        headers: {Token: AI_GATE_API_KEY}
    客户端会拿字面量 "AI_GATE_API_KEY" 当 Token 发给网关。"""

    def _deepseek_cfg_missing_env_name(self):
        """deepseek：providers 下没有 apiKeyEnv 字段。"""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = tmp.name
        path = os.path.join(home, ".dsh", "settings.yaml")
        write_text(path, "version: 1\nllm-pi-ai:\n  providers:\n    tower-ai:\n"
                         "      baseURL: https://example.invalid/zi/proxy\n")
        cfg = get_adapter("deepseek").read(env={"HOME": home}, home=home,
                                           project_dir=home)
        return cfg, path

    def test_auth_ref_repair_writes_only_the_reference(self):
        cfg, path = self._deepseek_cfg_missing_env_name()
        get_adapter("deepseek").apply(cfg, {FIELD_AUTH_ENV: "AI_GATE_API_KEY"},
                                      env={"HOME": os.path.dirname(os.path.dirname(path))},
                                      home=os.path.dirname(os.path.dirname(path)))
        text = read_text(path)
        self.assertIn("apiKeyEnv: AI_GATE_API_KEY", text)
        self.assertNotIn("Token: AI_GATE_API_KEY", text,
                         "把变量名当成凭据写进了 Token 请求头")
        self.assertNotIn("headers", text)

    def test_auth_ref_finding_points_at_the_reference_field(self):
        """判定层给出来的 fix_field 必须是 auth_env 而不是 auth。"""
        from suture import checks
        cfg, _ = self._deepseek_cfg_missing_env_name()
        findings = [f for f in checks.run_all_checks(cfg, {"auth": {}})
                    if f.key == "auth-ref" and not f.ok]
        self.assertTrue(findings, "没有产出 auth-ref 这条 finding")
        self.assertEqual(findings[0].fix_field, FIELD_AUTH_ENV)

    def test_codex_reference_still_uses_the_name(self):
        """codex 的 FIELD_AUTH 本来就是"只写引用"，改成按名字写不能破坏它。"""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = tmp.name
        path = os.path.join(home, ".codex", "config.toml")
        write_text(path, 'model = "gpt-5.6"\n\n'
                         '[model_providers.tower-ai]\nbase_url = "https://e.invalid/v1"\n')
        cfg = get_adapter("codex").read(env={"HOME": home}, home=home, project_dir=home)
        get_adapter("codex").apply(cfg, {FIELD_AUTH_ENV: "AI_GATE_API_KEY"},
                                   env={"HOME": home}, home=home, project_dir=home)
        text = read_text(path)
        self.assertIn('env_key = "AI_GATE_API_KEY"', text)
        self.assertNotIn("yotta_pk_", text)


class TestGeneratedConfigIsWrittenAtomically(unittest.TestCase):
    """⑨ 一键配置生成的那两个文件，原先还是先 open(w) 截断、后序列化。"""

    def _assert_atomic(self, module_path, needle):
        src = read_text(module_path)
        self.assertNotIn(needle, src,
                         "生成配置仍走「先截断后写」，序列化失败会把用户原文件清空")
        self.assertIn("write_bytes_atomic", src)

    def test_claude_generate_is_atomic(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._assert_atomic(os.path.join(root, "suture", "harness", "claude_code.py"),
                            'with open(path, "w", encoding="utf-8") as f:')

    def test_deepseek_generate_is_atomic(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._assert_atomic(os.path.join(root, "suture", "harness", "deepseek.py"),
                            'with open(path, "w", encoding="utf-8") as f:')

    def test_a_failed_generate_leaves_the_original_file_intact(self):
        """真跑一次失败路径。旧写法 `open(path, "w")` 会先把文件截断，等序列化那一步
        抛错时，用户原有的配置已经变成 0 字节了——而生成的还是一份"最小配置"，本来
        就只该覆盖自己那一份，不该把别人的东西清掉。"""
        with Sandbox() as sb:
            path = os.path.join(sb.home, ".claude", "settings.json")
            original = json.dumps({"env": {"ANTHROPIC_API_KEY": "yotta_pk_keepme1234"},
                                   "permissions": {"allow": ["Bash"]}}, indent=2)
            write_text(path, original)
            adapter = get_adapter("claude_code")
            # 孤立代理字符：json.dumps 放得过去，编码成 UTF-8 时必抛
            with self.assertRaises((UnicodeEncodeError, ValueError, TypeError)):
                adapter.generate_minimal_config(
                    "https://e.invalid/zi/proxy", "gpt-5.5\ud800",
                    env={"HOME": sb.home}, home=sb.home)
            self.assertEqual(read_text(path), original,
                             "生成失败，用户原有的配置文件却被清掉了")

    def test_deepseek_failed_generate_keeps_the_original(self):
        with Sandbox() as sb:
            path = os.path.join(sb.home, ".dsh", "settings.yaml")
            original = "version: 1\nllm-pi-ai:\n  providers:\n    tower-ai:\n" \
                       "      baseURL: https://e.invalid/zi/proxy\n"
            write_text(path, original)
            adapter = get_adapter("deepseek")
            # 模型名里带换行：受限 YAML 写出器表示不了，会明确报错
            with self.assertRaises(Exception):          # noqa: B017 —— 具体类型由写出器定
                adapter.generate_minimal_config(
                    "https://e.invalid/zi/proxy", "gpt-5.5\ninjected",
                    env={"HOME": sb.home}, home=sb.home)
            self.assertEqual(read_text(path), original,
                             "生成失败，用户原有的配置文件却被清掉了")


class TestBackupFailureIsGuardedAtEveryCallSite(unittest.TestCase):
    """③ 备份有三个调用点：贴 Key（input）、一键修复（auto/choice）、一键配置。
    走 CLI 的用例只能碰到 auto 那条（CLI 只自动应用 repair_kind == "auto"），
    另外两条得直接调引擎——只堵一处等于没堵，这正是前几轮的教训。"""

    @staticmethod
    def _break_backup(sb):
        with open(os.path.join(sb.home, ".suture"), "w", encoding="utf-8") as f:
            f.write("占位")           # ~/.suture 成了文件 → makedirs 必抛

    def test_input_branch(self):
        """贴 Key 那条（界面上「填 Key」输入框走的路径）。"""
        with Sandbox(behavior="token_only") as sb:
            _blocked_claude(sb)
            eng = make_engine(sb)
            self._break_backup(sb)
            res = eng.apply_action("claude_code",
                                   {"id": "auth", "repair_kind": "input",
                                    "fix_field": FIELD_AUTH},
                                   value="yotta_pk_newkey1234")
            self.assertEqual(res["result"], E.RESULT_MANUAL)
            self.assertIn("备份", res["message"])

    def test_configure_branch(self):
        """一键配置那条（全新用户点「配置」）。"""
        with Sandbox() as sb:
            eng = make_engine(sb)
            self._break_backup(sb)
            res = eng.configure_client("claude_code", model="gpt-5.5",
                                       api_key="yotta_pk_newkey1234")
            self.assertIsNone(res["path"], "备份没做成，却写入了配置")
            self.assertIn("备份", res["message"])


class TestEveryE2ECandidateIsTried(unittest.TestCase):
    """⑩ 口径是"有一个能用就算连通"，只试前 4 个会把排后面的正确型号漏掉。

    用 deepseek 造多模型配置：只有它的 providers.<route>.models 是**清单**
    （claude 的 ANTHROPIC_MODEL 是单值，天然只有一个候选）。"""

    MODELS = [f"model-{i}" for i in range(6)]

    def _six_models(self, sb):
        write_text(os.path.join(sb.home, ".dsh", "settings.yaml"),
                   "version: 1\nllm-pi-ai:\n  providers:\n    tower-ai:\n"
                   f"      baseURL: {sb.base_url}\n"
                   "      models:\n" +
                   "".join(f"        - id: {m}\n" for m in self.MODELS))

    def test_candidates_are_not_truncated(self):
        with Sandbox(behavior="token_only") as sb:
            self._six_models(sb)
            eng = make_engine(sb)
            cfg = eng.read_harness("deepseek")
            self.assertEqual(eng._e2e_candidates(cfg), self.MODELS)

    def test_every_candidate_actually_gets_a_request(self):
        """直接盯着出网那一层：6 个候选就必须发 6 次请求（原来是 4 次）。

        不去猜假网关认哪个型号——把发送函数换成记录器，断言的是"引擎有没有把它们
        都送去试"，这正是被 [:4] 砍掉的东西。"""
        with Sandbox(behavior="token_only") as sb:
            self._six_models(sb)
            from suture import gateway
            tried = []
            real = gateway.send_real_request

            def spy(base_url, headers, model, **kw):
                tried.append(model)
                return real(base_url, headers, model, **kw)

            gateway.send_real_request = spy
            try:
                make_engine(sb).run(["deepseek"])
            finally:
                gateway.send_real_request = real
            self.assertEqual(tried, self.MODELS)


if __name__ == "__main__":
    unittest.main()
