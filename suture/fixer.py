"""备份、写入、回滚。

这一层只执行动作，不做判断：写哪个字段、写什么值由检测层给出。
备份文件里同样有 Key 明文，所以权限要收紧。
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .harness.base import HarnessAdapter, HarnessConfig


@dataclass
class BackupEntry:
    original_path: str
    backup_path: str
    existed: bool


@dataclass
class BackupManifest:
    timestamp: str
    entries: List[BackupEntry] = field(default_factory=list)
    directory: str = ""


def backup_root(home: Optional[str] = None) -> str:
    base = home or os.path.expanduser("~")
    return os.path.join(base, ".suture", "backups")


def backup_files(paths: List[str], home: Optional[str] = None) -> BackupManifest:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = backup_root(home)
    # 同一秒内可能备份多次（CLI 的 --yes 会连着做几个自动修复）。以前直接复用
    # 同名目录，第二次会把第一次备份的文件**覆盖**掉——备份里留下的成了改过之后
    # 的那版，原始内容反而找不回来，等于白备份。这里保证每次拿到一个新目录。
    directory = os.path.join(root, stamp)
    n = 2
    while os.path.exists(directory):
        directory = os.path.join(root, f"{stamp}-{n}")
        n += 1
    os.makedirs(directory, exist_ok=True)
    try:
        os.chmod(root, 0o700)
        os.chmod(directory, 0o700)
    except OSError:
        pass

    manifest = BackupManifest(timestamp=stamp, directory=directory)
    for i, path in enumerate(paths):
        exists = os.path.exists(path)
        dest = os.path.join(directory, f"{i:02d}-{os.path.basename(path)}")
        if exists:
            shutil.copy2(path, dest)
            try:
                os.chmod(dest, 0o600)
            except OSError:
                pass
        manifest.entries.append(BackupEntry(original_path=path, backup_path=dest, existed=exists))
    return manifest


def rollback(manifest: BackupManifest) -> List[str]:
    """回滚到备份状态。返回回滚失败的文件列表——失败必须如实上报，
    不能让用户以为已经恢复原状。"""
    failed: List[str] = []
    for entry in manifest.entries:
        try:
            if entry.existed:
                os.makedirs(os.path.dirname(entry.original_path), exist_ok=True)
                shutil.copy2(entry.backup_path, entry.original_path)
            elif os.path.exists(entry.original_path):
                os.remove(entry.original_path)      # 修复时新建的文件，回滚就该删掉
        except OSError:
            failed.append(entry.original_path)
    return failed


def apply_fixes(adapter: HarnessAdapter, cfg: HarnessConfig, changes: Dict[str, str],
                env=None, home=None, project_dir=None) -> List[str]:
    return adapter.apply(cfg, changes, env=env, home=home, project_dir=project_dir)
