from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import urllib.request
import zipfile

APP_NAME = "ALICE Prompt Challenge"
DEFAULT_INSTALL_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ALICE-Prompt-Challenge"
MIN_PYTHON = (3, 11)
MAX_PYTHON = (3, 14)


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def _no_window_flags() -> int:
    if os.name == "nt":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def _run_logged(cmd: list[str], cwd: Path, log_fn, env: dict[str, str] | None = None) -> int:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_no_window_flags(),
    )
    if proc.stdout:
        for line in proc.stdout:
            clean = line.rstrip()
            if clean:
                log_fn(clean)
    return proc.wait()


def _probe_python(exe: str) -> tuple[bool, bool]:
    try:
        res_v = subprocess.run(
            [exe, "-c", "import sys; v=sys.version_info[:2]; print(int((3,11)<=v<(3,14)))"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=_no_window_flags(),
        )
        if res_v.returncode != 0 or res_v.stdout.strip() != "1":
            return False, False
        res_m = subprocess.run(
            [
                exe,
                "-c",
                (
                    "import importlib.util; "
                    "mods=('fastapi','uvicorn','torch','torchvision','diffusers','transformers','accelerate','peft','safetensors','huggingface_hub','cv2','skimage','PIL','numpy','httpx','dotenv'); "
                    "print(int(all(importlib.util.find_spec(m) is not None for m in mods)))"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=_no_window_flags(),
        )
        has_mods = res_m.returncode == 0 and res_m.stdout.strip() == "1"
        return True, has_mods
    except Exception:
        pass
    return False, False


def _find_system_python() -> tuple[str | None, bool]:
    candidates: list[str] = []
    for exe_name in ("python", "python3"):
        found = shutil.which(exe_name)
        if found:
            candidates.append(found)
    if not getattr(sys, "frozen", False):
        candidates.insert(0, sys.executable)
    local_py = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
    for ver in ("Python313", "Python312", "Python311"):
        p = local_py / ver / "python.exe"
        if p.is_file():
            candidates.append(str(p))
        p2 = Path("C:\\") / ver / "python.exe"
        if p2.is_file():
            candidates.append(str(p2))

    seen: set[str] = set()
    best_exe: str | None = None
    best_has_mods = False
    for cand in candidates:
        norm = str(Path(cand).resolve()).lower()
        if norm in seen or not Path(cand).is_file():
            continue
        seen.add(norm)
        ok_ver, has_mods = _probe_python(cand)
        if ok_ver and has_mods:
            return cand, True
        if ok_ver and best_exe is None:
            best_exe = cand
    return best_exe, best_has_mods


def _has_nvidia_gpu() -> bool:
    smi = shutil.which("nvidia-smi")
    if not smi:
        win_smi = Path(r"C:\Windows\System32\nvidia-smi.exe")
        smi = str(win_smi) if win_smi.is_file() else None
    if not smi:
        return False
    try:
        res = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=5, creationflags=_no_window_flags())
        return res.returncode == 0 and "GPU" in res.stdout
    except Exception:
        return False


def _ensure_uv(install_dir: Path, log_fn) -> str:
    uv_bin = install_dir / "bin" / "uv.exe"
    if uv_bin.is_file():
        return str(uv_bin)
    existing = shutil.which("uv")
    if existing:
        return existing
    log_fn("Python 3.11-3.13 not found; fetching portable uv runtime bootstrapper...")
    (install_dir / "bin").mkdir(parents=True, exist_ok=True)
    zip_path = install_dir / "bin" / "uv.zip"
    url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
    urllib.request.urlretrieve(url, str(zip_path))
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.lower().endswith("uv.exe"):
                uv_bin.write_bytes(zf.read(member))
                break
    try:
        zip_path.unlink()
    except OSError:
        pass
    if not uv_bin.is_file():
        raise RuntimeError("Could not provision standalone uv.exe to install Python.")
    return str(uv_bin)


def _write_uninstaller(install_dir: Path) -> None:
    bat_path = install_dir / "Uninstall.bat"
    script = (
        "@echo off\r\n"
        "echo Uninstalling ALICE Prompt Challenge...\r\n"
        "powershell -NoProfile -Command \"Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('ALICE-Prompt-Challenge') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }\"\r\n"
        "del \"%USERPROFILE%\\Desktop\\ALICE Prompt Challenge.lnk\" 2>nul\r\n"
        "cd /d \"%TEMP%\"\r\n"
        "rmdir /s /q \"%~dp0\"\r\n"
    )
    bat_path.write_text(script, encoding="utf-8")


def _create_desktop_shortcut(install_dir: Path, log_fn) -> None:
    if os.name != "nt":
        return
    venv_pythonw = install_dir / ".venv" / "Scripts" / "pythonw.exe"
    target_exe = venv_pythonw if venv_pythonw.is_file() else (install_dir / ".venv" / "Scripts" / "python.exe")
    launcher_py = install_dir / "launcher.py"
    icon_file = install_dir / "app.ico"
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    if not desktop.exists():
        return
    lnk_path = desktop / "ALICE Prompt Challenge.lnk"
    ps_cmd = (
        "$ws = New-Object -ComObject WScript.Shell; "
        "$s = $ws.CreateShortcut($env:APC_LNK); "
        "$s.TargetPath = $env:APC_TARGET; "
        "$s.Arguments = $env:APC_ARGS; "
        "$s.WorkingDirectory = $env:APC_WORKDIR; "
        "if (Test-Path $env:APC_ICON) { $s.IconLocation = $env:APC_ICON }; "
        "$s.Save()"
    )
    env = {
        **os.environ,
        "APC_LNK": str(lnk_path),
        "APC_TARGET": str(target_exe),
        "APC_ARGS": f'"{launcher_py}"',
        "APC_WORKDIR": str(install_dir),
        "APC_ICON": str(icon_file),
    }
    res = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
        env=env,
        capture_output=True,
        text=True,
        creationflags=_no_window_flags(),
    )
    if res.returncode == 0:
        log_fn(f"Created Desktop shortcut: {lnk_path}")
    else:
        log_fn(f"Desktop shortcut warning: {res.stderr.strip()}")


def run_installation(
    install_dir: Path,
    install_models: bool,
    make_shortcut: bool,
    log_fn,
    progress_fn,
) -> Path:
    progress_fn(5)
    install_dir.mkdir(parents=True, exist_ok=True)
    log_path = install_dir / "install.log"
    log_path.write_text("", encoding="utf-8")

    def emit(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
        try:
            log_fn(msg)
        except Exception:
            pass

    bundle = _bundle_root()
    payload_zip = bundle / "payload.zip"
    if not payload_zip.is_file():
        raise FileNotFoundError(f"Embedded payload.zip not found in {bundle}")

    emit(f"Extracting frontend UI, backend server & assets to {install_dir}...")
    with zipfile.ZipFile(payload_zip, "r") as zf:
        zf.extractall(install_dir)
    env_example = install_dir / ".env.example"
    env_target = install_dir / ".env"
    if env_example.is_file() and not env_target.exists():
        env_target.write_text(env_example.read_text(encoding="utf-8"), encoding="utf-8")
    _write_uninstaller(install_dir)
    progress_fn(20)

    venv_dir = install_dir / ".venv"
    venv_py = venv_dir / "Scripts" / "python.exe"
    sys_py, has_all_mods = _find_system_python()

    if sys_py and has_all_mods:
        emit(f"Detected ready Python environment with AI dependencies: {sys_py}")
        if not venv_py.is_file():
            rc = _run_logged([sys_py, "-m", "venv", "--system-site-packages", str(venv_dir)], install_dir, emit)
            if rc != 0:
                raise RuntimeError("Failed to create virtual environment.")
        progress_fn(60)
    else:
        if sys_py:
            emit(f"Creating virtual environment using {sys_py}...")
            if not venv_py.is_file():
                rc = _run_logged([sys_py, "-m", "venv", str(venv_dir)], install_dir, emit)
                if rc != 0:
                    raise RuntimeError("Failed to create virtual environment.")
            progress_fn(32)
            has_gpu = _has_nvidia_gpu()
            if has_gpu:
                emit("NVIDIA GPU detected: installing CUDA 12.6 PyTorch runtime...")
                rc = _run_logged(
                    [str(venv_py), "-m", "pip", "install", "--upgrade", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126"],
                    install_dir,
                    emit,
                )
                if rc != 0:
                    emit("CUDA wheel install fell back to standard PyPI torch...")
            progress_fn(48)
            emit("Installing backend & AI dependencies from requirements.txt...")
            rc = _run_logged([str(venv_py), "-m", "pip", "install", "-r", "requirements.txt"], install_dir, emit)
            if rc != 0:
                raise RuntimeError("pip install -r requirements.txt failed.")
        else:
            uv_exe = _ensure_uv(install_dir, emit)
            emit("Installing standalone Python 3.12 via uv...")
            if _run_logged([uv_exe, "python", "install", "3.12"], install_dir, emit) != 0:
                raise RuntimeError("uv python install 3.12 failed.")
            progress_fn(35)
            if not venv_py.is_file():
                if _run_logged([uv_exe, "venv", "--python", "3.12", str(venv_dir)], install_dir, emit) != 0:
                    raise RuntimeError("uv venv creation failed.")
            has_gpu = _has_nvidia_gpu()
            if has_gpu:
                emit("NVIDIA GPU detected: installing CUDA 12.6 PyTorch runtime via uv...")
                _run_logged(
                    [uv_exe, "pip", "install", "--python", str(venv_py), "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126"],
                    install_dir,
                    emit,
                )
            progress_fn(50)
            emit("Installing backend & AI packages via uv...")
            if _run_logged([uv_exe, "pip", "install", "--python", str(venv_py), "-r", "requirements.txt"], install_dir, emit) != 0:
                raise RuntimeError("uv pip install failed.")
        progress_fn(65)

    if install_models:
        emit("Verifying & caching local AI models (ResNet50 similarity + local GPU diffusion if available)...")
        model_script = (
            "from services.similarity import warmup_similarity; "
            "print('Loading/caching ResNet50 similarity model...', flush=True); "
            "warmup_similarity(); "
            "print('ResNet50 similarity model ready.', flush=True); "
            "import torch; "
            "print('CUDA available:', torch.cuda.is_available(), flush=True); "
            "from services.image_gen import warmup_engine; "
            "warmup_engine(); "
            "print('Local AI model warmup finished.', flush=True)"
        )
        rc = _run_logged([str(venv_py), "-c", model_script], install_dir, emit)
        if rc != 0:
            emit("Warning: Local AI model warmup returned non-zero; app remains usable with offline synthesis/cloud.")
    progress_fn(90)

    if make_shortcut:
        _create_desktop_shortcut(install_dir, emit)

    progress_fn(100)
    emit("Installation complete! Use the Desktop shortcut or click LAUNCH GAME below.")
    return venv_py


class InstallerGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("ALICE Prompt Challenge — Setup")
        self.root.geometry("640x480")
        self.root.resizable(False, False)
        self.root.configure(bg="#22150A")
        ico = _bundle_root() / "app.ico"
        if ico.is_file():
            try:
                self.root.iconbitmap(str(ico))
            except Exception:
                pass

        tk.Label(
            self.root,
            text="ALICE PROMPT CHALLENGE INSTALLER",
            font=("Georgia", 18, "bold"),
            fg="#E8C76A",
            bg="#22150A",
        ).pack(pady=(18, 4))

        tk.Label(
            self.root,
            text="Installs the Wonderland UI, FastAPI backend, Chrome Kiosk lock, and Local AI models.",
            font=("Segoe UI", 9),
            fg="#D7C39B",
            bg="#22150A",
        ).pack(pady=(0, 12))

        path_frame = tk.Frame(self.root, bg="#22150A")
        path_frame.pack(fill="x", padx=24, pady=(0, 8))

        tk.Label(path_frame, text="Install Folder:", font=("Segoe UI", 9, "bold"), fg="#F5EBD0", bg="#22150A").pack(anchor="w")
        row = tk.Frame(path_frame, bg="#22150A")
        row.pack(fill="x", pady=(4, 0))

        self.dir_var = tk.StringVar(value=str(DEFAULT_INSTALL_DIR))
        self.dir_entry = tk.Entry(row, textvariable=self.dir_var, font=("Segoe UI", 9), bg="#F5EBD0", fg="#1A120B", relief="solid", bd=1)
        self.dir_entry.pack(side="left", fill="x", expand=True, ipady=4)

        self.btn_browse = tk.Button(
            row,
            text="Browse...",
            font=("Segoe UI", 9),
            bg="#3E2A17",
            fg="#F5EBD0",
            relief="flat",
            padx=10,
            command=self._choose_dir,
        )
        self.btn_browse.pack(side="left", padx=(8, 0), ipady=2)

        opts_frame = tk.Frame(self.root, bg="#22150A")
        opts_frame.pack(fill="x", padx=24, pady=(4, 8))

        self.models_var = tk.BooleanVar(value=True)
        self.shortcut_var = tk.BooleanVar(value=True)

        tk.Checkbutton(
            opts_frame,
            text="Download & cache Local AI Models (ResNet50 + DreamShaper-8 / Hyper-SD on CUDA GPU)",
            variable=self.models_var,
            fg="#F5EBD0",
            bg="#22150A",
            selectcolor="#3E2A17",
            activebackground="#22150A",
            activeforeground="#E8C76A",
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        tk.Checkbutton(
            opts_frame,
            text="Create Desktop Shortcut (ALICE Prompt Challenge)",
            variable=self.shortcut_var,
            fg="#F5EBD0",
            bg="#22150A",
            selectcolor="#3E2A17",
            activebackground="#22150A",
            activeforeground="#E8C76A",
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        self.pbar = ttk.Progressbar(self.root, mode="determinate", maximum=100)
        self.pbar.pack(fill="x", padx=24, pady=(4, 8))

        self.log_box = scrolledtext.ScrolledText(
            self.root,
            height=11,
            bg="#150D06",
            fg="#EFE0BC",
            font=("Consolas", 8),
            relief="solid",
            bd=1,
            state="disabled",
        )
        self.log_box.pack(fill="both", expand=True, padx=24, pady=(0, 12))

        btn_bar = tk.Frame(self.root, bg="#22150A")
        btn_bar.pack(fill="x", padx=24, pady=(0, 16))

        self.btn_install = tk.Button(
            btn_bar,
            text="INSTALL ALICE PROMPT CHALLENGE",
            font=("Segoe UI", 10, "bold"),
            bg="#D4AF37",
            fg="#1A120B",
            activebackground="#E8C76A",
            relief="flat",
            padx=18,
            pady=8,
            command=self._start_install,
        )
        self.btn_install.pack(side="right")

        self.installed_dir: Path | None = None

    def _choose_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.dir_var.get())
        if chosen:
            self.dir_var.set(chosen)

    def _append_log(self, msg: str) -> None:
        def _do() -> None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.root.after(0, _do)

    def _set_progress(self, val: int) -> None:
        self.root.after(0, lambda v=val: self.pbar.configure(value=v))

    def _start_install(self) -> None:
        install_dir = Path(self.dir_var.get().strip() or str(DEFAULT_INSTALL_DIR))
        self.btn_install.configure(state="disabled", text="INSTALLING...")
        self.btn_browse.configure(state="disabled")
        self.dir_entry.configure(state="disabled")

        def _worker() -> None:
            try:
                run_installation(
                    install_dir=install_dir,
                    install_models=self.models_var.get(),
                    make_shortcut=self.shortcut_var.get(),
                    log_fn=self._append_log,
                    progress_fn=self._set_progress,
                )
                self.installed_dir = install_dir
                self.root.after(0, self._on_success)
            except Exception as exc:
                self._append_log(f"ERROR: {exc}")
                self.root.after(0, lambda e=str(exc): self._on_error(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_success(self) -> None:
        self.btn_install.configure(state="normal", text="LAUNCH GAME", command=self._launch_installed)

    def _on_error(self, err: str) -> None:
        self.btn_install.configure(state="normal", text="RETRY INSTALL", command=self._start_install)
        self.btn_browse.configure(state="normal")
        self.dir_entry.configure(state="normal")
        messagebox.showerror("Installation Failed", err)

    def _launch_installed(self) -> None:
        if self.installed_dir is None:
            return
        pyw = self.installed_dir / ".venv" / "Scripts" / "pythonw.exe"
        exe = pyw if pyw.is_file() else (self.installed_dir / ".venv" / "Scripts" / "python.exe")
        subprocess.Popen(
            [str(exe), str(self.installed_dir / "launcher.py")],
            cwd=str(self.installed_dir),
            creationflags=_no_window_flags(),
        )
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_INSTALL_DIR))
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--no-shortcut", action="store_true")
    args = parser.parse_args()

    if args.silent:
        def _safe_out(msg: str) -> None:
            if sys.stdout is not None:
                sys.stdout.write(msg + "\n")
                sys.stdout.flush()

        run_installation(
            install_dir=Path(args.dir),
            install_models=not args.skip_models,
            make_shortcut=not args.no_shortcut,
            log_fn=_safe_out,
            progress_fn=lambda _: None,
        )
        return

    InstallerGUI().run()


if __name__ == "__main__":
    main()
