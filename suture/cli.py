"""命令行界面。与图形界面共用 engine，不含任何判断逻辑。"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import engine as E
from .checks import FIXABLE_YES

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, RED = "\033[32m", "\033[33m", "\033[31m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _print_findings(findings, only_issues=False) -> None:
    for f in findings:
        if only_issues and f.ok:
            continue
        mark = _c("[正常]", GREEN) if f.ok else _c("[待修复]", YELLOW)
        print(f"  {mark} {f.label}：{f.detail}")
        if f.current_value:
            print(f"         当前值：{f.current_value}")
        if f.suggested_value and not f.ok:
            print(f"         建议值：{f.suggested_value}")
        if f.note:
            print(f"         {_c(f.note, DIM)}")


def run(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="suture", description="AI Gate 配置诊断与一键修复")
    parser.add_argument("--profile", help="网关规则文件路径")
    parser.add_argument("--harness", action="append",
                        choices=["claude_code", "codex", "deepseek"],
                        help="只检查指定的 harness，可重复")
    parser.add_argument("--project-dir", help="项目目录，默认当前目录")
    parser.add_argument("--home", help="覆盖 HOME（测试用）")
    parser.add_argument("--yes", action="store_true", help="发现问题后直接修复，不交互确认")
    parser.add_argument("--no-fix", action="store_true", help="只检测不修复")
    parser.add_argument("--gui", action="store_true", help="打开图形界面")
    args = parser.parse_args(argv)

    if args.gui:
        from .shell import launch
        return launch(profile_path=args.profile, project_dir=args.project_dir)

    eng = E.Engine(profile_path=args.profile, home=args.home, project_dir=args.project_dir)
    print(_c("Suture · AI Gate 配置体检", BOLD))
    print(_c(f"规则来源：{eng.profile_source}", DIM))
    print()

    report = eng.run(harness_ids=args.harness)

    if report.result == E.RESULT_NO_HARNESS:
        print(_c(report.message, YELLOW))
        return 3

    p = report.gateway_probe
    print(f"阶段一：网关连通性自检 —— {p['detail']}（{p['elapsed_ms']}ms）")
    if report.result == E.RESULT_GATEWAY_DOWN:
        print()
        print(_c(report.message, YELLOW))
        return 2
    if p["classification"] == "auth_error":
        print(_c("  网关返回鉴权错误，这更可能是本地配置的问题，继续检查本地配置。", YELLOW))
    print()

    exit_code = 0
    for h in report.harnesses:
        print(_c(f"阶段二：{h.display_name}", BOLD))
        for fs in h.files:
            state = "存在" if fs["exists"] else "不存在（视为该层未配置）"
            if fs["exists"] and not fs["active"]:
                state = "存在但不生效"
            print(f"  {fs['layer']}：{fs['path']} —— {state}")

        if h.result == E.RESULT_NO_CONFIG:
            print(_c("  没有找到任何配置——这不是配错了，是还没配置过。", YELLOW))
            if not args.no_fix:
                path = eng.generate_config(h.harness_id)
                print(f"  已生成一份最小配置：{path}")
                print("  请将其中的 Key 替换为网关后台生成的真实值。")
            print()
            continue

        issues = [f for f in h.findings if not f.ok]
        if issues:
            print(_c(f"  发现 {len(issues)} 项问题：", YELLOW))
            _print_findings(issues)
            oks = [f for f in h.findings if f.ok]
            if oks:
                print(_c(f"  另外 {len(oks)} 项检查通过：", GREEN))
                _print_findings(oks)
        else:
            print(_c("  静态检查全部通过。", GREEN))
        for info in h.infos:
            print(f"  {_c(info, DIM)}")
        e2e_results = h.e2e_results if getattr(h, "e2e_results", None) else ([h.e2e] if h.e2e else [])
        for r in e2e_results:
            label = f"端到端真实请求（模型 {r['model']}）" if len(e2e_results) > 1 else "端到端真实请求"
            print(f"  {label}：{r['detail']}（{r['elapsed_ms']}ms）")

        if h.result == E.RESULT_HEALTHY:
            print(_c("  这个 harness 一切正常。", GREEN))
            print()
            continue

        # 有候选值、但不该替用户猜的项（模型没配置、名字像别的型号）：
        # 摆一个编号菜单让用户自己选，跟下面「一键修复」批量套用是两条路。
        choosable = [f for f in h.findings if not f.ok and f.choices and f.fix_field]
        for f in choosable:
            print()
            print(_c(f"  {f.label}：{f.detail}", YELLOW))
            for i, c in enumerate(f.choices, 1):
                print(f"    {i}. {c['label']}")
            if args.no_fix:
                print(_c("  指定了 --no-fix，不会写入。", DIM))
                exit_code = max(exit_code, 3)
                continue
            if args.yes:
                print(_c("  这一项需要手动选择，--yes 不会替你选，已跳过。", DIM))
                exit_code = max(exit_code, 3)
                continue
            ans = input("  输入序号选择，回车跳过：").strip()
            if not ans:
                print("  已跳过。")
                exit_code = max(exit_code, 3)
                continue
            try:
                choice = f.choices[int(ans) - 1]
            except (ValueError, IndexError):
                print(_c("  输入无效，已跳过。", RED))
                exit_code = max(exit_code, 3)
                continue
            result = eng.apply_choice(h.harness_id, f.fix_field, choice["value"])
            for step in result.get("steps", []):
                print(f"    {step['detail']}")
            if result["result"] == E.RESULT_FIXED:
                print(_c(f"  {result['message']}", GREEN))
            elif result["result"] == E.RESULT_KEPT_FIX:
                print(_c(f"  {result['message']}", YELLOW))
                exit_code = max(exit_code, 4)
            else:
                print(_c(f"  {result['message']}", RED))
                exit_code = max(exit_code, 5)

        fixable = [f for f in h.findings if not f.ok and f.fixable == FIXABLE_YES]
        if not fixable:
            if not choosable:
                print(_c("  没有能自动修复的项，需要人工处理。", YELLOW))
                exit_code = max(exit_code, 3)
            print()
            continue
        if args.no_fix:
            print(_c(f"  有 {len(fixable)} 项可自动修复，但指定了 --no-fix。", DIM))
            exit_code = max(exit_code, 3)
            print()
            continue
        if not args.yes:
            ans = input(f"  发现 {len(fixable)} 项可以自动修复的问题，是否修复？会先备份 [y/N] ")
            if ans.strip().lower() != "y":
                print("  已跳过修复。")
                exit_code = max(exit_code, 3)
                print()
                continue

        print(_c("  阶段三：正在修复…", BOLD))
        result = eng.fix(h.harness_id, h.findings)
        for step in result.get("steps", []):
            print(f"    {step['detail']}")
        print()
        if result["result"] == E.RESULT_FIXED:
            print(_c(f"  {result['message']}", GREEN))
        elif result["result"] == E.RESULT_KEPT_FIX:
            print(_c(f"  {result['message']}", YELLOW))
            exit_code = max(exit_code, 4)
        else:
            print(_c(f"  {result['message']}", RED))
            exit_code = max(exit_code, 5)
        print()

    if report.result == E.RESULT_HEALTHY and report.message:
        print(_c(report.message, GREEN))
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
