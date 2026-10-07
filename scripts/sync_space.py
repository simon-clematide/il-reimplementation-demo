#!/usr/bin/env python3
"""
Syncs demo_core/ into hf_space/demo_core/ so the Space repository can be pushed as a self-contained Git directory.
"""

import os
import shutil


def sync():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "demo_core")
    dst = os.path.join(root, "hf_space", "demo_core")

    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Synced {src} -> {dst}")


if __name__ == "__main__":
    sync()
