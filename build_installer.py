from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
BUILD_DIR = BASE_DIR / "build"
PAYLOAD_ZIP = BUILD_DIR / "payload.zip"
ICON_PATH = BUILD_DIR / "app.ico"
OUT_EXE = BASE_DIR / "ALICE-Prompt-Challenge-Installer.exe"


def _make_icon() -> None:
    src = BASE_DIR / "static" / "assets" / "magical-scroll.webp"
    img = Image.open(src).convert("RGBA")
    img.save(ICON_PATH, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])


def _build_payload() -> None:
    with zipfile.ZipFile(PAYLOAD_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in ("app.py", "requirements.txt", ".env.example"):
            zf.write(BASE_DIR / name, name)
        zf.write(BASE_DIR / "installer" / "launcher.py", "launcher.py")
        zf.write(ICON_PATH, "app.ico")
        for sub in ("services", "static"):
            root = BASE_DIR / sub
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
                    continue
                zf.write(path, str(path.relative_to(BASE_DIR)))


def main() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    _make_icon()
    _build_payload()

    spec_dir = BUILD_DIR / "spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    work_dir = BUILD_DIR / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        "ALICE-Prompt-Challenge-Installer",
        "--icon",
        str(ICON_PATH),
        "--specpath",
        str(spec_dir),
        "--workpath",
        str(work_dir),
        "--distpath",
        str(BASE_DIR),
        "--add-data",
        f"{PAYLOAD_ZIP};.",
        "--add-data",
        f"{ICON_PATH};.",
        str(BASE_DIR / "installer" / "installer_app.py"),
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)
    size_mb = OUT_EXE.stat().st_size / (1024 * 1024)
    print(f"Built {OUT_EXE} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
