"""
_meta.py - run metadata stored in every result JSON (camera-ready comment R3-15):
hardware, operating system, library versions and the git commit of the code.
"""

import datetime
import os
import platform
import subprocess


def _version(module):
    try:
        mod = __import__(module)
        if module == "gmpy2":
            return mod.version()
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def run_meta(**extra):
    meta = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "versions": {m: _version(m) for m in ("numpy", "pandas", "sklearn", "gmpy2", "psutil")},
        "tree_training": "DecisionTreeClassifier(max_depth=<depth>, random_state=42), other parameters at scikit-learn defaults",
    }
    try:
        import psutil
        meta["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        meta["ram_gb"] = None
    try:
        meta["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or None
    except Exception:
        meta["git_commit"] = None
    meta.update(extra)
    return meta
