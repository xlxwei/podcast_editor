"""檢查這台電腦能不能跑這套工具，缺什麼就印出安裝指令。

用法：python3 tools/check_env.py
"""
import importlib.util
import platform
import shutil
import sys

problems = []
if sys.version_info < (3, 10):
    problems.append(f"Python 要 3.10 以上，現在是 {platform.python_version()}。安裝：brew install python")
if platform.system() != "Darwin" or platform.machine() != "arm64":
    problems.append("轉錄用的 mlx-whisper 只能在 Apple Silicon 的 Mac 上跑。")
missing = [pip for mod, pip in (("numpy", "numpy"), ("mlx_whisper", "mlx-whisper"), ("opencc", "opencc-python-reimplemented"))
           if importlib.util.find_spec(mod) is None]
if missing:
    problems.append(f"缺 Python 套件。安裝：{sys.executable} -m pip install {' '.join(missing)}")
for tool in ("ffmpeg", "ffprobe", "git"):
    if not shutil.which(tool):
        problems.append(f"找不到 {tool}。安裝：brew install {'ffmpeg' if tool != 'git' else 'git'}")

if problems:
    print("還不能用：")
    for p in dict.fromkeys(problems):
        print(" -", p)
    sys.exit(1)
print(f"環境沒問題（Python {platform.python_version()}，{sys.executable}）。")
