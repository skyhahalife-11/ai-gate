"""打开界面窗口。

先尝试系统自带的 WebView，拿到一个真正的应用窗口；初始化不成功（主要是
Linux 上没装 WebKitGTK 的情况）就退回到打开系统默认浏览器。两条路径用的是
同一套界面，只是承载的容器不同，不会出现「打开就报错、什么都做不了」。
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import webbrowser
from typing import Optional

from . import engine as E
from . import server as S

WINDOW_TITLE = "Suture · AI Gate 配置体检"


def _report_fatal(message: str) -> None:
    """把启动期的致命错误说给用户听。

    打包成无控制台的窗口程序后，sys.stdout / sys.stderr 都是 None，`print` 等于扔进
    黑洞（不报错、也没人看得见），而 PyInstaller 默认只把 traceback 倒进 stderr、
    不弹窗——合起来就是"双击了一下，什么都没发生"。所以这里必须自己找出口：
    先落一份日志（方便事后排查），Windows 上再弹个中文框。"""
    try:
        log = os.path.join(tempfile.gettempdir(), "suture-error.log")
        with open(log, "a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")
    except OSError:
        pass
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, WINDOW_TITLE, 0x10)
            return
        except Exception:      # noqa: BLE001 —— 弹不出来也不能再抛
            pass
    try:
        print(message, file=sys.stderr)
    except Exception:          # noqa: BLE001
        pass


def _try_webview(url: str) -> bool:
    """pywebview 只在打包时附带，引擎本身不依赖它。装不上或初始化失败都返回 False。"""
    try:
        import webview  # noqa: PLC0415 —— 故意延迟导入，没有它也要能跑
    except Exception:
        return False
    try:
        webview.create_window(WINDOW_TITLE, url, width=860, height=760)
        webview.start()
        return True
    except Exception:
        return False


def launch(profile_path: Optional[str] = None, project_dir: Optional[str] = None,
           port: int = 0, open_ui: bool = True) -> int:
    if project_dir is None and getattr(sys, "frozen", False):
        # 打包后 os.getcwd() 是 exe 自己的启动目录，不是用户的"项目目录"。拿它当
        # 项目目录的话，claude / codex 的**项目级**配置文件会被优先选中——谁把 exe
        # 解到某个项目目录里双击，修复就落到那个项目上。双击场景本来就没有项目
        # 上下文，退到用户主目录（那里项目级路径与全局路径指向同一个文件，写的
        # 就是该写的那一份），而不是把 exe 所在目录当成用户的项目。
        project_dir = os.path.expanduser("~")
    try:
        engine = E.Engine(profile_path=profile_path, project_dir=project_dir)
        httpd, state, url = S.serve_in_background(engine, port=port)
    except Exception as exc:      # noqa: BLE001
        _report_fatal(f"Suture 启动失败：{exc}\n\n详细信息见 "
                      f"{os.path.join(tempfile.gettempdir(), 'suture-error.log')}")
        return 1

    if not open_ui:
        print(url)
        return 0

    if _try_webview(url):
        httpd.shutdown()
        return 0

    # 没有可用的系统 WebView（主要是 Linux 上没装 WebKitGTK）：退回默认浏览器。
    # SUTURE_NO_BROWSER 留给无界面环境和自动化：照常提供服务，只是不主动拉起浏览器。
    skip_browser = os.environ.get("SUTURE_NO_BROWSER") == "1"
    print(f"{'服务已启动' if skip_browser else '已在浏览器中打开'}：{url}", flush=True)
    print("这个地址只在本机有效，带一次性令牌，关掉这个程序就失效。", flush=True)
    print("按 Ctrl+C 退出。", flush=True)
    if not skip_browser:
        opened = False
        try:
            # 返回值要接：没有默认浏览器 / 被系统策略挡掉时它返回 False 而不是抛错。
            # 不接的话下面那三行 print 又没人看得见（无控制台），用户对着一个
            # 转个不停、既没窗口也没提示的进程，只能去任务管理器。
            opened = bool(webbrowser.open(url))
        except Exception:      # noqa: BLE001
            opened = False
        if not opened:
            _report_fatal("没能自动打开界面。请在浏览器里手动访问下面这个地址"
                          f"（只在本机有效）：\n\n{url}")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(launch())
