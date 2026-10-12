"""第三方库 vendor 区。

导入本包即把 third_party/ 插到 sys.path 最前，
使 vendored 的 olmo_core 优先于 site-packages 的 pip 版本。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
