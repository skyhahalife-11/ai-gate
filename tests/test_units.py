"""单元层面的验收：YAML 子集读写、掩码、多层解析、各检测项的判断。"""
from __future__ import annotations

import json
import os
import unittest

from tests.helpers import Sandbox, write_json, write_text, real_profile

from suture.harness import _minimal_yaml as yaml
from suture.harness.base import LayerValue, build_resolved, mask_secret
from suture.harness.claude_code import ClaudeCodeAdapter
from suture.harness.codex import CodexAdapter
from suture.harness.deepseek import DeepSeekHarnessAdapter
from suture import checks
from suture.profile import expected_base_url


class TestMinimalYaml(unittest.TestCase):
    def test_nested_and_roundtrip(self):
        text = (
            "llm-pi-ai:\n"
            "  providers:\n"
            "    tower-ai:\n"
            "      baseURL: https://example.com/zi/proxy\n"
            "      apiKeyEnv: AI_GATE_API_KEY\n"
            "      models:\n"
            "        - id: glm-5.3\n"
            "        - id: kimi-k3\n"
        )
        data = yaml.parse(text)
        route = data["llm-pi-ai"]["providers"]["tower-ai"]
        self.assertEqual(route["baseURL"], "https://example.com/zi/proxy")
        self.assertEqual([m["id"] for m in route["models"]], ["glm-5.3", "kimi-k3"])
        self.assertEqual(yaml.parse(yaml.dump(data)), data)

    def test_rejects_unsupported_syntax_loudly(self):
        for text in ("a: &x 1", "a: |\n  block", "a: {b: 1}", "---\na: 1", "a:\n\tb: 1"):
            with self.assertRaises(yaml.MiniYamlError):
                yaml.parse(text)

    def test_comments_and_quotes(self):
        data = yaml.parse("# 注释\nkey: 'a: b # 不是注释'  # 这才是注释\nflag: true\n")
        self.assertEqual(data["key"], "a: b # 不是注释")
        self.assertIs(data["flag"], True)


class TestBasics(unittest.TestCase):
    def test_mask_secret(self):
        self.assertEqual(mask_secret("yotta_pk_abcdefgh"), "yott*********efgh")
        self.assertEqual(mask_secret("short"), "*****")
        self.assertEqual(mask_secret(None), "")
        self.assertNotIn("cdefgh", mask_secret("yotta_pk_abcdefgh"))

    def test_layer_priority_and_differences(self):
        rf = build_resolved("base_url", [
            LayerValue("环境变量", "", "A"),
            LayerValue("项目级配置", "/p", "B"),
            LayerValue("全局配置", "/g", "A"),
        ])
        self.assertEqual(rf.value, "A")
        self.assertEqual(rf.source_layer, "环境变量")
        self.assertEqual([l.layer for l in rf.differing_layers()], ["项目级配置"])


class TestClaudeCodeAdapter(unittest.TestCase):
    def test_reads_env_over_files_and_picks_header(self):
        with Sandbox() as sb:
            write_json(os.path.join(sb.home, ".claude", "settings.json"),
                       {"env": {"ANTHROPIC_BASE_URL": "https://old", "ANTHROPIC_MODEL": "glm-5.3"}})
            env = dict(sb.env, ANTHROPIC_BASE_URL="https://new",
                       ANTHROPIC_AUTH_TOKEN="yotta_pk_tok12345")
            cfg = ClaudeCodeAdapter().read(env=env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg.field("base_url").value, "https://new")
            self.assertEqual(cfg.field("base_url").source_layer, "环境变量")
            self.assertEqual(cfg.field("model").value, "glm-5.3")
            # 设置的是 AUTH_TOKEN，客户端就会发 Authorization 头
            self.assertEqual(cfg.auth_header, "Authorization")

    def test_api_key_wins_and_conflict_reported(self):
        with Sandbox() as sb:
            env = dict(sb.env, ANTHROPIC_API_KEY="yotta_pk_key12345",
                       ANTHROPIC_AUTH_TOKEN="yotta_pk_tok12345")
            cfg = ClaudeCodeAdapter().read(env=env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg.auth_header, "x-api-key")
            self.assertIsNotNone(cfg.auth_conflict)

    def test_broken_json_reports_whole_file_invalid(self):
        with Sandbox() as sb:
            write_text(os.path.join(sb.home, ".claude", "settings.json"),
                       '{"env": {"ANTHROPIC_MODEL": "glm-5.3",}}')
            cfg = ClaudeCodeAdapter().read(env=sb.env, home=sb.home, project_dir=sb.project)
            bad = [f for f in cfg.files if not f.parse_ok]
            self.assertEqual(len(bad), 1)
            self.assertIn("整个文件解析失败", bad[0].parse_error)
            findings = checks.check_config_syntax(cfg)
            self.assertTrue(any(not f.ok for f in findings))

    def test_apply_writes_effective_layer(self):
        with Sandbox() as sb:
            g = os.path.join(sb.home, ".claude", "settings.json")
            write_json(g, {"env": {"ANTHROPIC_BASE_URL": "https://old"}})
            a = ClaudeCodeAdapter()
            cfg = a.read(env=sb.env, home=sb.home, project_dir=sb.project)
            a.apply(cfg, {"base_url": "https://new", "auth": "yotta_pk_new12345"},
                    env=sb.env, home=sb.home, project_dir=sb.project)
            cfg2 = a.read(env=sb.env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg2.field("base_url").value, "https://new")
            self.assertEqual(cfg2.field("auth").value, "yotta_pk_new12345")
            self.assertEqual(cfg2.auth_header, "x-api-key")

    def test_custom_headers_recognized_as_parallel_auth_source(self):
        # ANTHROPIC_CUSTOM_HEADERS 之前完全没被读取过——这里验证它现在能被解析出来，
        # 而且是「跟主字段平行」的关系，不是替代关系。
        accepted = real_profile()["auth"]["accepted_headers"]
        with Sandbox() as sb:
            env = dict(sb.env, ANTHROPIC_CUSTOM_HEADERS=(
                "Token: yotta_pk_tokenvalid\nX-Irrelevant-Header: whatever"))
            cfg = ClaudeCodeAdapter().read(env=env, home=sb.home, project_dir=sb.project,
                                           accepted_headers=accepted)
            self.assertEqual(len(cfg.extra_auth_headers), 1)   # 无关的头名不算数
            self.assertEqual(cfg.extra_auth_headers[0].header, "Token")
            self.assertEqual(cfg.extra_auth_headers[0].value, "yotta_pk_tokenvalid")
            # 只有 Token 头、没设 API_KEY/AUTH_TOKEN 时，鉴权应该被认为是「有配置的」，
            # 而不是报「没有配置鉴权信息」
            self.assertFalse(cfg.field("auth").is_set)
            findings = checks.check_auth(cfg, real_profile())
            self.assertFalse(any(f.key == "auth" and not f.ok
                                 and "没有配置鉴权信息" in f.detail for f in findings))
            token_finding = next(f for f in findings if f.key == "auth-extra:Token")
            self.assertTrue(token_finding.ok, token_finding.detail)

    def test_custom_headers_ignored_without_accepted_list(self):
        # accepted_headers 没传进来时（比如老调用方式）不应该瞎猜，安全地什么都不识别。
        with Sandbox() as sb:
            env = dict(sb.env, ANTHROPIC_CUSTOM_HEADERS="Token: yotta_pk_tokenvalid")
            cfg = ClaudeCodeAdapter().read(env=env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg.extra_auth_headers, [])

    def test_conflicting_api_key_and_custom_header_both_flagged(self):
        # 真实报过的场景：ANTHROPIC_API_KEY 是一个不对的旧值，
        # ANTHROPIC_CUSTOM_HEADERS 里的 Token 才是真正在用的网关 Key。
        # 两个都设置时，客户端会把两个请求头一起发出去，Suture 不该只看 API_KEY
        # 就下结论，也不该假装两者不会冲突。
        accepted = real_profile()["auth"]["accepted_headers"]
        with Sandbox() as sb:
            env = dict(sb.env, ANTHROPIC_API_KEY="sk-ant-stale-real-key",
                       ANTHROPIC_CUSTOM_HEADERS="Token: yotta_pk_realgatekey")
            cfg = ClaudeCodeAdapter().read(env=env, home=sb.home, project_dir=sb.project,
                                           accepted_headers=accepted)
            findings = checks.check_auth(cfg, real_profile())
            conflict = next(f for f in findings if f.key == "auth-multiple-active")
            self.assertFalse(conflict.ok)
            self.assertIn("Token", conflict.detail)
            self.assertIn("更符合网关签发 Key 的特征", conflict.detail)   # 能从前缀看出谁像真的
            self.assertEqual(conflict.fixable, checks.FIXABLE_NO)   # 环境变量，Suture 改不了


class TestManagedSettings(unittest.TestCase):
    """真实反馈：有些用户的机器是公司统一下发配置的（"直连网关"，自己不用填
    base_url），之前 Suture 只读全局/项目级两层，完全看不到这一层，导致对这些
    用户来说"探测网关、模型都探测不到"。这里验证新加的 managed-settings.json
    支持：优先级最高（但真实环境变量还能盖过它）、检测得到、不会被当成可写文件
    去备份/回滚、跟本地文件冲突时不会提供没有意义的"一键修复"。"""

    def _managed_env(self, sb, path):
        return dict(sb.env, SUTURE_MANAGED_SETTINGS_PATH=path)

    def test_managed_settings_wins_over_global_and_project(self):
        with Sandbox() as sb:
            managed_path = os.path.join(sb.home, "managed-settings.json")
            write_json(managed_path, {"env": {
                "ANTHROPIC_BASE_URL": "https://gate.company.internal",
                "ANTHROPIC_MODEL": "glm-5.3",
            }})
            write_json(os.path.join(sb.home, ".claude", "settings.json"),
                       {"env": {"ANTHROPIC_BASE_URL": "https://stale-old-address"}})
            env = self._managed_env(sb, managed_path)
            cfg = ClaudeCodeAdapter().read(env=env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg.field("base_url").value, "https://gate.company.internal")
            self.assertIn("组织托管配置", cfg.field("base_url").source_layer)
            self.assertEqual(cfg.field("model").value, "glm-5.3")

    def test_detected_with_only_managed_settings_present(self):
        # 全新用户,机器是 IT 统一装好的,自己从没碰过 ~/.claude 或者任何环境变量——
        # 这正是被反馈的场景,detect() 之前会在这种情况下判定"没装这个客户端"。
        with Sandbox() as sb:
            managed_path = os.path.join(sb.home, "managed-settings.json")
            write_json(managed_path, {"env": {"ANTHROPIC_BASE_URL": "https://gate.company.internal"}})
            env = self._managed_env(sb, managed_path)
            self.assertTrue(ClaudeCodeAdapter().detect(env=env, home=sb.home, project_dir=sb.project))

    def test_real_env_var_still_beats_managed_settings(self):
        with Sandbox() as sb:
            managed_path = os.path.join(sb.home, "managed-settings.json")
            write_json(managed_path, {"env": {"ANTHROPIC_BASE_URL": "https://gate.company.internal"}})
            env = dict(self._managed_env(sb, managed_path),
                       ANTHROPIC_BASE_URL="https://really-exported-in-shell")
            cfg = ClaudeCodeAdapter().read(env=env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg.field("base_url").value, "https://really-exported-in-shell")
            self.assertEqual(cfg.field("base_url").source_layer, "环境变量")

    def test_managed_settings_excluded_from_writable_paths(self):
        # 备份/回滚不该去碰这个文件——它不属于 Suture 能写的范围，权限通常也不够。
        with Sandbox() as sb:
            managed_path = os.path.join(sb.home, "managed-settings.json")
            write_json(managed_path, {"env": {"ANTHROPIC_BASE_URL": "https://gate.company.internal"}})
            env = self._managed_env(sb, managed_path)
            a = ClaudeCodeAdapter()
            cfg = a.read(env=env, home=sb.home, project_dir=sb.project)
            self.assertNotIn(managed_path, a.writable_paths(cfg))

    def test_conflict_with_managed_layer_has_no_pointless_auto_fix(self):
        # 本地文件跟托管配置不一样是预期内的（托管配置就是用来压过本地设置的），
        # 不是"配置冲突"，而且改本地文件也不会让托管配置生效——不该给"一键修复"。
        with Sandbox() as sb:
            managed_path = os.path.join(sb.home, "managed-settings.json")
            write_json(managed_path, {"env": {"ANTHROPIC_BASE_URL": "https://gate.company.internal"}})
            write_json(os.path.join(sb.home, ".claude", "settings.json"),
                       {"env": {"ANTHROPIC_BASE_URL": "https://stale-old-address"}})
            env = self._managed_env(sb, managed_path)
            cfg = ClaudeCodeAdapter().read(env=env, home=sb.home, project_dir=sb.project)
            findings = checks.check_layer_consistency(cfg)
            conflict = next(f for f in findings if f.key == "conflict:base_url")
            self.assertEqual(conflict.fixable, checks.FIXABLE_NO)
            self.assertIn("托管配置", conflict.detail)


class TestCodexAdapter(unittest.TestCase):
    def test_self_test_command_is_the_documented_non_interactive_mode(self):
        """`codex exec` 是官方文档确认过的非交互模式，不是猜的
        （developers.openai.com/codex/noninteractive）。"""
        cmd = CodexAdapter().self_test_command()
        self.assertEqual(cmd[:2], ["codex", "exec"])
        self.assertIn("--skip-git-repo-check", cmd)

    def _write_user(self, sb, extra=""):
        write_text(os.path.join(sb.home, ".codex", "config.toml"),
                   'model = "glm-5.3"\n'
                   'model_provider = "ai-gate"\n\n'
                   '[model_providers.ai-gate]\n'
                   'base_url = "https://gate/v1"\n'
                   'env_key = "AI_GATE_API_KEY"\n' + extra)

    def test_reads_provider_table_and_indirect_auth(self):
        with Sandbox() as sb:
            self._write_user(sb)
            env = dict(sb.env, AI_GATE_API_KEY="yotta_pk_abc12345")
            cfg = CodexAdapter().read(env=env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg.field("base_url").value, "https://gate/v1")
            self.assertTrue(cfg.auth_is_indirect)
            self.assertEqual(cfg.auth_env_name, "AI_GATE_API_KEY")
            self.assertTrue(cfg.auth_env_resolved)
            self.assertEqual(cfg.field("auth").value, "yotta_pk_abc12345")

    def test_env_key_pointing_at_unset_variable_is_caught(self):
        with Sandbox() as sb:
            self._write_user(sb)
            cfg = CodexAdapter().read(env=sb.env, home=sb.home, project_dir=sb.project)
            self.assertFalse(cfg.auth_env_resolved)
            findings = checks.check_auth(cfg, real_profile())
            ref = [f for f in findings if f.key == "auth-ref"][0]
            self.assertFalse(ref.ok)
            self.assertIn("该变量未设置", ref.detail)

    def test_untrusted_project_layer_is_ignored_and_flagged(self):
        with Sandbox() as sb:
            self._write_user(sb)
            write_text(os.path.join(sb.project, ".codex", "config.toml"),
                       'model = "kimi-k3"\n')
            cfg = CodexAdapter().read(env=sb.env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg.field("model").value, "glm-5.3")   # 项目层不生效
            proj = [f for f in cfg.files if f.layer == "项目级配置"][0]
            self.assertFalse(proj.active)
            findings = checks.check_layer_active(cfg)
            self.assertTrue(any("未被 Codex 标记为可信" in f.detail for f in findings))

    def test_trusted_project_layer_wins(self):
        with Sandbox() as sb:
            # TOML 基本字符串里反斜杠是转义符，Windows 路径必须转义后再塞进去，
            # 否则 "C:\Users\..." 会被当成非法/错误的转义序列，表里的项目路径对不上。
            self._write_user(sb, extra=f'\n[projects.{json.dumps(sb.project)}]\ntrust_level = "trusted"\n')
            write_text(os.path.join(sb.project, ".codex", "config.toml"), 'model = "kimi-k3"\n')
            cfg = CodexAdapter().read(env=sb.env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg.field("model").value, "kimi-k3")

    def test_apply_preserves_other_content(self):
        with Sandbox() as sb:
            self._write_user(sb, extra='\n# 保留这行注释\n[other]\nkeep = true\n')
            a = CodexAdapter()
            cfg = a.read(env=sb.env, home=sb.home, project_dir=sb.project)
            a.apply(cfg, {"base_url": "https://new/v1", "model": "kimi-k3"},
                    env=sb.env, home=sb.home, project_dir=sb.project)
            text = open(os.path.join(sb.home, ".codex", "config.toml"), encoding="utf-8").read()
            self.assertIn("# 保留这行注释", text)
            self.assertIn("keep = true", text)
            cfg2 = a.read(env=sb.env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg2.field("base_url").value, "https://new/v1")
            self.assertEqual(cfg2.field("model").value, "kimi-k3")

    def test_apply_never_writes_key_into_config(self):
        with Sandbox() as sb:
            self._write_user(sb)
            a = CodexAdapter()
            cfg = a.read(env=sb.env, home=sb.home, project_dir=sb.project)
            a.apply(cfg, {"auth": "yotta_pk_secret999"},
                    env=sb.env, home=sb.home, project_dir=sb.project)
            text = open(os.path.join(sb.home, ".codex", "config.toml"), encoding="utf-8").read()
            self.assertNotIn("yotta_pk_secret999", text)
            self.assertIn("AI_GATE_API_KEY", text)


class TestDeepSeekAdapter(unittest.TestCase):
    def _write_base(self, sb, base_url="https://gate/zi/proxy"):
        write_text(os.path.join(sb.project, "cordis.yml"),
                   "plugins:\n"
                   "  - name: '@deepseek-ai/dsh-llm-pi-ai'\n"
                   "    config:\n"
                   "      providers:\n"
                   "        tower-ai:\n"
                   "          api: anthropic-messages\n"
                   f"          baseURL: {base_url}\n"
                   "          apiKeyEnv: AI_GATE_API_KEY\n"
                   "          models:\n"
                   "            - id: glm-5.3\n")

    def test_user_layer_overrides_base(self):
        with Sandbox() as sb:
            self._write_base(sb)
            write_text(os.path.join(sb.home, ".dsh", "settings.yaml"),
                       "llm-pi-ai:\n  providers:\n    tower-ai:\n      baseURL: https://user-layer\n")
            cfg = DeepSeekHarnessAdapter().read(env=sb.env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg.field("base_url").value, "https://user-layer")
            self.assertEqual(cfg.field("base_url").source_layer, "用户层 settings.yaml")
            self.assertEqual([l.layer for l in cfg.field("base_url").differing_layers()],
                             ["组合基线 cordis.yml"])

    def test_key_precedence_env_over_credentials(self):
        with Sandbox() as sb:
            self._write_base(sb)
            write_text(os.path.join(sb.home, ".dsh", ".credentials.yaml"),
                       "version: 1\nrefs:\n  AI_GATE_API_KEY: yotta_pk_fromfile\n")
            cfg = DeepSeekHarnessAdapter().read(env=sb.env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg.field("auth").value, "yotta_pk_fromfile")
            env = dict(sb.env, AI_GATE_API_KEY="yotta_pk_fromenv")
            cfg2 = DeepSeekHarnessAdapter().read(env=env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg2.field("auth").value, "yotta_pk_fromenv")
            self.assertIn("环境变量", cfg2.field("auth").source_layer)

    def test_dsh_home_env_var_respected(self):
        with Sandbox() as sb:
            alt = os.path.join(sb.project, "custom-dsh")
            write_text(os.path.join(alt, "settings.yaml"),
                       "llm-pi-ai:\n  providers:\n    tower-ai:\n      baseURL: https://alt\n")
            env = dict(sb.env, DSH_HOME=alt)
            cfg = DeepSeekHarnessAdapter().read(env=env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg.field("base_url").value, "https://alt")

    def test_headers_token_form_is_understood(self):
        with Sandbox() as sb:
            write_text(os.path.join(sb.project, "cordis.yml"),
                       "plugins:\n"
                       "  - name: '@deepseek-ai/dsh-llm-pi-ai'\n"
                       "    config:\n"
                       "      providers:\n"
                       "        tower-ai:\n"
                       "          baseURL: https://gate/zi/proxy\n"
                       "          headers:\n"
                       "            Token: yotta_pk_inheader\n")
            cfg = DeepSeekHarnessAdapter().read(env=sb.env, home=sb.home, project_dir=sb.project)
            self.assertEqual(cfg.auth_header, "Token")
            self.assertEqual(cfg.field("auth").value, "yotta_pk_inheader")

    def test_all_registered_models_are_checked(self):
        with Sandbox() as sb:
            write_text(os.path.join(sb.project, "cordis.yml"),
                       "plugins:\n"
                       "  - name: '@deepseek-ai/dsh-llm-pi-ai'\n"
                       "    config:\n"
                       "      providers:\n"
                       "        tower-ai:\n"
                       "          baseURL: https://gate/zi/proxy\n"
                       "          models:\n"
                       "            - id: glm-5.3\n"
                       "            - id: GLM-5.3-Flash\n"
                       "            - id: not-a-model\n")
            cfg = DeepSeekHarnessAdapter().read(env=sb.env, home=sb.home, project_dir=sb.project)
            self.assertEqual(len(cfg.model_candidates), 3)
            findings = checks.check_models(cfg, real_profile())
            self.assertEqual(len(findings), 3)
            by_value = {f.current_value: f for f in findings}
            self.assertTrue(by_value["glm-5.3"].ok)
            self.assertFalse(by_value["GLM-5.3-Flash"].ok)
            # fix_value 是「其它已注册的模型原样保留，只把这一个改正确」之后的完整清单，
            # 不能只写这一个模型进去，那样会把 glm-5.3 和 not-a-model 都顶掉。
            self.assertEqual(by_value["GLM-5.3-Flash"].fix_value,
                             "glm-5.3,glm-5.3-flash,not-a-model")
            self.assertFalse(by_value["not-a-model"].ok)

    def test_unparseable_yaml_is_reported_not_guessed(self):
        with Sandbox() as sb:
            write_text(os.path.join(sb.home, ".dsh", "settings.yaml"), "llm-pi-ai: {providers: x}\n")
            cfg = DeepSeekHarnessAdapter().read(env=sb.env, home=sb.home, project_dir=sb.project)
            bad = [f for f in cfg.files if f.exists and not f.parse_ok]
            self.assertEqual(len(bad), 1)
            self.assertIn("流式写法", bad[0].parse_error)


class TestChecks(unittest.TestCase):
    def setUp(self):
        self.profile = real_profile()

    def _cc(self, sb, **env_extra):
        env = dict(sb.env, **env_extra)
        return ClaudeCodeAdapter().read(env=env, home=sb.home, project_dir=sb.project)

    def test_base_url_variants(self):
        expected = expected_base_url(self.profile, "claude_code")
        cases = {
            expected: True,
            expected + "/v1": False,
            expected.replace("https://", "http://"): False,
            "  " + expected + "  ": False,
            "tower-ai.yottastudios.com/zi/proxy": False,
            "https://wrong.example.com/api": False,
        }
        with Sandbox() as sb:
            for value, should_be_ok in cases.items():
                cfg = self._cc(sb, ANTHROPIC_BASE_URL=value)
                f = checks.check_base_url(cfg, self.profile)
                self.assertEqual(f.ok, should_be_ok, f"{value!r} 判断错误：{f.detail}")
                if not should_be_ok:
                    self.assertEqual(f.fix_value, expected)

    def test_codex_expects_v1_suffix(self):
        root = self.profile["base_url"]["canonical_root"]
        with Sandbox() as sb:
            write_text(os.path.join(sb.home, ".codex", "config.toml"),
                       'model_provider = "ai-gate"\n\n[model_providers.ai-gate]\n'
                       f'base_url = "{root}"\n')
            cfg = CodexAdapter().read(env=sb.env, home=sb.home, project_dir=sb.project)
            f = checks.check_base_url(cfg, self.profile)
            self.assertFalse(f.ok)
            self.assertIn("少了一段 /v1", f.detail)
            self.assertEqual(f.fix_value, root + "/v1")

    def test_vendor_key_is_identified(self):
        with Sandbox() as sb:
            cfg = self._cc(sb, ANTHROPIC_API_KEY="sk-ant-api03-xxxxyyyy")
            findings = checks.check_auth(cfg, self.profile)
            auth = [f for f in findings if f.key == "auth"][0]
            self.assertFalse(auth.ok)
            self.assertIn("Anthropic 官方", auth.detail)
            self.assertEqual(auth.fixable, checks.FIXABLE_PARTIAL)
            self.assertNotIn("xxxxyyyy", auth.current_value)   # 展示必须掩码

    def test_variant_models_are_never_auto_corrected(self):
        """gpt-5.6 与 gpt-5.6-sol/terra/luna 是各自独立的型号，
        不能因为「看起来像」就自动改，那会让用户在不知情下连到另一个模型。
        这种「不敢猜」的场景要给一份可以点的候选菜单，不是自动填。"""
        with Sandbox() as sb:
            cfg = self._cc(sb, ANTHROPIC_MODEL="gpt-5.6-so")
            f = checks.check_models(cfg, self.profile)[0]
            self.assertFalse(f.ok)
            self.assertEqual(f.fixable, checks.FIXABLE_NO)
            self.assertIsNone(f.fix_value)
            self.assertIn("gpt-5.6-sol", f.suggested_value)
            self.assertEqual(f.fix_field, "model")
            self.assertIn("gpt-5.6-sol", [c["label"] for c in f.choices])
            chosen = next(c for c in f.choices if c["label"] == "gpt-5.6-sol")
            self.assertEqual(chosen["value"], "gpt-5.6-sol")   # 单模型 harness，替换后就是它自己

    def test_unconfigured_model_is_not_an_error_but_still_offers_the_list(self):
        """不填模型名不是错误：客户端不填就用它自己的默认模型，照样能用。
        所以这一项默认判成正常（界面上会收进折叠区，不平铺几十个型号当报错），
        但候选清单仍然给出来，想指定某个型号的人点开就能选。
        同时 Suture 不该偷偷填探活用的 probe_model 进去——那个模型只是用来测
        网关活不活着，跟用户想用哪个是两回事。"""
        with Sandbox() as sb:
            cfg = self._cc(sb)   # 没设 ANTHROPIC_MODEL
            f = checks.check_models(cfg, self.profile)[0]
            self.assertTrue(f.ok)
            self.assertEqual(f.fixable, checks.FIXABLE_NO)
            self.assertIsNone(f.fix_value)
            self.assertEqual(f.fix_field, "model")
            probe_model = self.profile.get("probe_model")
            labels = [c["label"] for c in f.choices]
            self.assertEqual(set(labels), set(checks.model_ids(self.profile)))
            # probe_model 不能被当成「推荐答案」单独标出来，它只是清单里普通的一项
            self.assertIn(probe_model, labels)

    def test_case_and_separator_typos_are_auto_corrected(self):
        with Sandbox() as sb:
            for typo, target in [("GLM-5.3", "glm-5.3"), ("Claude_Sonnet_5", "claude-sonnet-5"),
                                 ("kimi-k2-6", "kimi-k2.6")]:
                cfg = self._cc(sb, ANTHROPIC_MODEL=typo)
                f = checks.check_models(cfg, self.profile)[0]
                self.assertFalse(f.ok, typo)
                self.assertEqual(f.fix_value, target, typo)

    def test_layer_conflict_flagged_as_error(self):
        with Sandbox() as sb:
            write_json(os.path.join(sb.home, ".claude", "settings.json"),
                       {"env": {"ANTHROPIC_BASE_URL": "https://global"}})
            write_json(os.path.join(sb.project, ".claude", "settings.json"),
                       {"env": {"ANTHROPIC_BASE_URL": "https://project"}})
            cfg = ClaudeCodeAdapter().read(env=sb.env, home=sb.home, project_dir=sb.project)
            findings = checks.check_layer_consistency(cfg)
            self.assertEqual(len(findings), 1)
            self.assertFalse(findings[0].ok)
            # 不给一键修复：写的就是生效的那一层（本来就是这个值），写了等于原地
            # 重写、其它层没被动过，下次检查同一条原因原样回来。
            self.assertEqual(findings[0].fixable, checks.FIXABLE_NO)
            self.assertEqual(findings[0].current_value, "https://project")

    def test_unknown_key_typo_detected(self):
        with Sandbox() as sb:
            write_json(os.path.join(sb.home, ".claude", "settings.json"),
                       {"envs": {"ANTHROPIC_MODEL": "glm-5.3"}})
            cfg = ClaudeCodeAdapter().read(env=sb.env, home=sb.home, project_dir=sb.project,
                                           known_keys=self.profile["known_settings_keys"])
            findings = checks.check_unknown_keys(cfg, self.profile)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].suggested_value, "env")
            self.assertIn("等同于没配置这一项", findings[0].detail)

    def test_protocol_version_check_is_off_by_default(self):
        with Sandbox() as sb:
            cfg = self._cc(sb, ANTHROPIC_MODEL="glm-5.3")
            self.assertEqual(checks.check_protocol_version(cfg, self.profile), [])
            enabled = dict(self.profile)
            enabled["protocol_version"] = dict(self.profile["protocol_version"], check_enabled=True)
            self.assertEqual(len(checks.check_protocol_version(cfg, enabled)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
