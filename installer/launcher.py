from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
import urllib.request
import json
import webbrowser

INSTALL_DIR = Path(__file__).resolve().parent
DATA_DIR = INSTALL_DIR / "data"
ICON_PATH = INSTALL_DIR / "app.ico"
SERVER_URL = "http://127.0.0.1:8000"


def _find_python_exe() -> str:
    venv_py = INSTALL_DIR / ".venv" / "Scripts" / "python.exe"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def _find_chrome_exe() -> str | None:
    candidates = [
        shutil.which("chrome"),
        shutil.which("chrome.exe"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
    ]
    for cand in candidates:
        if cand and Path(cand).is_file():
            return str(Path(cand))
    return None


def _check_health() -> dict | None:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/api/health", timeout=1.5) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass
    return None


def _open_url_in_chrome(url: str, kiosk: bool = False) -> None:
    chrome = _find_chrome_exe()
    if chrome:
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if kiosk:
            profile_dir = DATA_DIR / "chrome_kiosk_profile"
            profile_dir.mkdir(parents=True, exist_ok=True)
            args = [
                chrome,
                f"--user-data-dir={profile_dir}",
                "--kiosk",
                f"{SERVER_URL}/?kiosk=1",
                "--new-window",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-pinch",
                "--overscroll-history-navigation=0",
                "--disable-session-crashed-bubble",
                "--disable-infobars",
            ]
        else:
            args = [chrome, url]
        subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
        return
    webbrowser.open(url)


class LauncherWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("ALICE Prompt Challenge")
        self.root.geometry("500x290")
        self.root.resizable(False, False)
        self.root.configure(bg="#22150A")
        if ICON_PATH.is_file():
            try:
                self.root.iconbitmap(str(ICON_PATH))
            except Exception:
                pass

        self.server_proc: subprocess.Popen | None = None
        self.owned_server = False
        self.opened_once = False
        self.closing = False

        header = tk.Label(
            self.root,
            text="ALICE PROMPT CHALLENGE",
            font=("Georgia", 18, "bold"),
            fg="#E8C76A",
            bg="#22150A",
        )
        header.pack(pady=(22, 6))

        sub = tk.Label(
            self.root,
            text="Local AI Backend & Wonderland Web Arena",
            font=("Segoe UI", 10),
            fg="#D7C39B",
            bg="#22150A",
        )
        sub.pack(pady=(0, 14))

        self.status_var = tk.StringVar(value="Starting local server & warming AI models...")
        self.status_lbl = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 10, "bold"),
            fg="#F5EBD0",
            bg="#2F1E10",
            width=54,
            height=3,
            relief="solid",
            bd=1,
            wraplength=440,
        )
        self.status_lbl.pack(padx=24, pady=(0, 18))

        btn_row = tk.Frame(self.root, bg="#22150A")
        btn_row.pack(pady=(0, 12))

        self.btn_open = tk.Button(
            btn_row,
            text="OPEN GAME",
            font=("Segoe UI", 10, "bold"),
            bg="#D4AF37",
            fg="#1A120B",
            activebackground="#E8C76A",
            relief="flat",
            padx=14,
            pady=8,
            state="disabled",
            command=lambda: _open_url_in_chrome(SERVER_URL, kiosk=False),
        )
        self.btn_open.pack(side="left", padx=6)

        self.btn_kiosk = tk.Button(
            btn_row,
            text="LAUNCH CHROME KIOSK",
            font=("Segoe UI", 10, "bold"),
            bg="#3E2A17",
            fg="#F5EBD0",
            activebackground="#5C3E18",
            relief="flat",
            padx=14,
            pady=8,
            state="disabled",
            command=self._enter_kiosk,
        )
        self.btn_kiosk.pack(side="left", padx=6)

        self.btn_stop = tk.Button(
            self.root,
            text="STOP SERVER & EXIT",
            font=("Segoe UI", 9, "bold"),
            bg="#4A2018",
            fg="#FDF6E3",
            activebackground="#6B2E22",
            relief="flat",
            padx=14,
            pady=6,
            command=self._shutdown,
        )
        self.btn_stop.pack(pady=(2, 8))

        self.root.protocol("WM_DELETE_WINDOW", self._shutdown)
        self.root.after(150, self._start_backend)

    def _enter_kiosk(self) -> None:
        try:
            req = urllib.request.Request(
                f"{SERVER_URL}/api/kiosk/enter",
                data=json.dumps({"origin": SERVER_URL}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            _open_url_in_chrome(f"{SERVER_URL}/?kiosk=1", kiosk=True)

    def _start_backend(self) -> None:
        health = _check_health()
        if health is not None:
            mode = (health.get("engine") or {}).get("active_mode", "Ready")
            self._mark_ready(mode)
            return

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log_path = DATA_DIR / "server.log"
        py_exe = _find_python_exe()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            log_file = open(log_path, "w", encoding="utf-8")
            self.server_proc = subprocess.Popen(
                [py_exe, "app.py"],
                cwd=str(INSTALL_DIR),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            self.owned_server = True
        except Exception as exc:
            messagebox.showerror("Launch Error", f"Could not start backend server:\n{exc}")
            self._shutdown()
            return

        threading.Thread(target=self._poll_ready, daemon=True).start()

    def _poll_ready(self) -> None:
        while not self.closing:
            if self.server_proc is not None and self.server_proc.poll() is not None:
                self.root.after(
                    0,
                    lambda: self.status_var.set("Server stopped unexpectedly. Check data/server.log."),
                )
                return
            health = _check_health()
            if health is not None:
                mode = (health.get("engine") or {}).get("active_mode", "Ready")
                self.root.after(0, lambda m=mode: self._mark_ready(m))
                return
            time.sleep(0.5)

    def _mark_ready(self, mode: str) -> None:
        self.status_var.set(f"Running on {SERVER_URL}\nActive Engine: {mode}")
        self.btn_open.configure(state="normal")
        self.btn_kiosk.configure(state="normal")
        if not self.opened_once:
            self.opened_once = True
            _open_url_in_chrome(SERVER_URL, kiosk=False)

    def _shutdown(self) -> None:
        self.closing = True
        if self.owned_server and self.server_proc is not None and self.server_proc.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(self.server_proc.pid)],
                        capture_output=True,
                        timeout=4,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                else:
                    self.server_proc.terminate()
            except Exception:
                pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    LauncherWindow().run()
