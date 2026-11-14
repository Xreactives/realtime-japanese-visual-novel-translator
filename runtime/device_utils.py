import ctypes
import importlib.util
import os
import sys
from pathlib import Path


def register_optional_cuda_dlls() -> None:
    """Adds wheel-bundled NVIDIA CUDA 12 DLL folders to PATH on Windows if present."""
    if sys.platform != "win32":
        return

    for pkg in ("nvidia.cublas", "nvidia.cuda_runtime", "torch"):
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:
            continue

        if spec and spec.submodule_search_locations:
            pkg_root = Path(next(iter(spec.submodule_search_locations)))
            for sub in ("bin", "lib"):
                dll_dir = pkg_root / sub
                if dll_dir.exists():
                    os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")


def can_use_cuda_for_ctranslate2() -> bool:
    """Returns True only if a CUDA GPU exists AND cublas64_12.dll loads cleanly."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() == 0:
            return False

        if sys.platform == "win32":
            ctypes.WinDLL("cublas64_12.dll")

        return True
    except Exception:
        return False