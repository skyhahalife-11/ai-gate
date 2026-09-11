"""命令行界面。与图形界面共用 engine，不含任何判断逻辑。"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import engine as E

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, RED = "\033[32m", "\033[33m", "\033[31m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _print_blocked_issues(client) -> None:
    for it in client.issues:
        sev = it.get("severity", "low")
        tag = {"high": "高", "medium": "中", "low": "低"}.get(sev, sev)
        print(f"    [{tag}] {it.get('title') or it.get('id')}")
        print(f"        {it.get('detail', '')}")


def run(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="suture", description="AI Gate 连通体检与安装上手")
    parser.add_argument("--profile", help="网关规则文件路径")
    parser.add_argument("--harness", action="append",
                        choices=["claude_code", "codex", "deepseek"],
                        help="只检查指定的 harness，可重复")
    parser.add_argument("--project-dir", help="项目目录，默认当前目录")
    parser.add_argument("--home", help="覆盖 HOME（测试用）")
    parser.add_argument("--yes", action="store_true",
                        help="连通失败时可自动修的原因直接修，不交互确认")
    parser.add_argument("--no-fix", action="store_true",
                        help="不修改配置文件。注意 Suture 自己一个字节都不写，"
                             "但仍会调用客户端做一次自证（客户端会写它自己的状态目录）")
    parser.add_argument("--gui", action="store_true", help="打开图形界面")
    args = parser.parse_args(argv)

    if args.gui:
        from .shell import launch
        return launch(profile_path=args.profile, project_dir=args.project_dir)

    try:
        eng = E.Engine(profile_path=args.profile, home=args.home, project_dir=args.project_dir)
    except (OSError, ValueError) as exc:
        # 规则文件读不出来（路径不存在、不是 JSON、权限不对）。GUI 侧有中文 500
        # 兜底，这里是全仓唯一没有兜底的入口——不包的话用户看到的是一个英文
        # Python 栈，连"哪个文件出了问题"都得自己从 traceback 里找。
        print(_c(f"读不到网关规则文件：{exc}", RED), file=sys.stderr)
        return 1
    print(_c("Suture · AI Gate 连通体检", BOLD))
    print(_c(f"规则来源：{eng.profile_source}", DIM))
    print()

    report = eng.run(client_ids=args.harness)

    g = report.gateway
    if g:
        print(f"网关连通性：{g['detail']}（{g['elapsed_ms']}ms）")
    if report.gateway_side_down:
        print()
        print(_c(report.message, YELLOW))
        return 2

    changed = False
    exit_code = 0
    for client in report.clients:
        print(_c(f"· {client.display_name} — {client.state_label}", BOLD))
        if client.state == E.STATE_CONNECTED:
            print(_c(f"    {client.detail}", GREEN))
            print()
            continue
        if client.detail:
            print(f"    {client.detail}")
        if client.state == E.STATE_BLOCKED and client.issues:
            print(_c("    可能的原因：", YELLOW))
            _print_blocked_issues(client)
            auto = [it for it in client.issues
                    if it.get("repair_kind") == "auto" and it.get("fix_field") and it.get("fix_value") is not None]
            if auto:
                if args.no_fix:
                    print(_c(f"    有 {len(auto)} 项可自动修复，但指定了 --no-fix，未改动。", DIM))
                    exit_code = max(exit_code, 3)
                else:
                    ok = True if args.yes else None
                    if not args.yes:
                        try:
                            ans = input(f"    发现 {len(auto)} 项可以自动修复的原因，是否处理？会先备份 [y/N] ")
                            ok = ans.strip().lower() == "y"
                        except (EOFError, RuntimeError):
                            # 不在终端里跑（`< NUL`、stdin 被管道接走、打包后无控制台）。
                            # 不接住就直接崩在这儿，而 check-only.bat 那种"跑一遍看结论"
                            # 的用法正好会撞上。当作"不修"继续，并说明怎么让它不必问。
                            print(_c("    当前没有可用的输入，已跳过自动修复"
                                     "（加 --yes 可直接修）。", YELLOW))
                            ok = False
                    if ok:
                        for it in auto:
                            res = eng.apply_action(client.client_id, it)
                            # 只有真改动了才算 changed。无条件置位会让"所有修复都失败"
                            # 也走到下面那句「已按可自动修复的原因改完」，与上面刚打印的
                            # 失败说明自相矛盾。
                            if res.get("result") in (E.RESULT_FIXED, E.RESULT_CHANGED):
                                changed = True
                            for s in res.get("steps", []):
                                print(f"      · {s['detail']}")
                            print(_c(f"      {res['message']}", GREEN if res.get("result") == E.RESULT_FIXED else YELLOW))
        elif client.state in (E.STATE_UNCONFIGURED, E.STATE_NOT_INSTALLED):
            exit_code = max(exit_code, 3)
        print()

    if changed:
        print(_c("已按可自动修复的原因改完，重新检查确认…", DIM))
        report = eng.run(client_ids=args.harness)
        if any(c.state == E.STATE_CONNECTED for c in report.clients):
            print(_c("有客户端现在能连上 AI Gate 了。", GREEN))
        else:
            print(_c("仍然连不上——详情请用图形界面逐条处理（部分原因需要你提供 Key 或手动选模型）。", YELLOW))

    if report.result == E.RESULT_GATEWAY_DOWN:
        exit_code = max(exit_code, 2)
    elif any(c.state == E.STATE_BLOCKED for c in report.clients):
        exit_code = max(exit_code, 3)
    elif all(c.state == E.STATE_NOT_INSTALLED for c in report.clients):
        print(_c("本机还没有这些客户端。图形界面里有「安装 / 上手」页可以装。", YELLOW))
        exit_code = max(exit_code, 3)
    elif any(c.state in (E.STATE_UNCONFIGURED,) for c in report.clients):
        print(_c("有客户端装了但还没配置连 AI Gate。图形界面里可以一键配置。", YELLOW))
        exit_code = max(exit_code, 3)
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
