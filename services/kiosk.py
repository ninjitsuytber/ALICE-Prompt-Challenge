from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KIOSK_PROFILE_DIR = DATA_DIR / "chrome_kiosk_profile"
SECRET_PHRASE = "utmsanjunganbangsa"

_KIOSK_LOCK = threading.Lock()
_KIOSK_PROCESS: subprocess.Popen | None = None


def _find_chrome_executable() -> str | None:
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


def _sanitize_local_kiosk_url(origin: str | None) -> str:
    default_url = "http://127.0.0.1:8000/?kiosk=1"
    if not origin:
        return default_url
    parsed = urlparse(origin.strip())
    if parsed.scheme not in ("http", "https"):
        return default_url
    if parsed.hostname not in ("127.0.0.1", "localhost"):
        return default_url
    return f"{parsed.scheme}://{parsed.netloc}/?kiosk=1"


def _ensure_clean_preferences() -> None:
    pref_path = KIOSK_PROFILE_DIR / "Default" / "Preferences"
    if not pref_path.exists():
        return
    try:
        data = json.loads(pref_path.read_text(encoding="utf-8"))
        profile = data.setdefault("profile", {})
        profile["exit_type"] = "Normal"
        profile["exited_cleanly"] = True
        pref_path.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def _find_profile_pids(root_only: bool = False) -> list[int]:
    if os.name != "nt":
        return []
    marker = str(KIOSK_PROFILE_DIR).lower()
    if root_only:
        filter_clause = (
            "$_.CommandLine -and $_.CommandLine.ToLower().Contains($env:KIOSK_MARKER) "
            "-and -not $_.CommandLine.Contains('--type=')"
        )
    else:
        filter_clause = "$_.CommandLine -and $_.CommandLine.ToLower().Contains($env:KIOSK_MARKER)"
    ps_script = (
        f"Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
        f"Where-Object {{ {filter_clause} }} | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            env={**os.environ, "KIOSK_MARKER": marker},
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        pids: list[int] = []
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return pids
    except Exception:
        return []


def _terminate_pid(pid: int) -> None:
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            ["taskkill", "/PID", str(pid)],
            capture_output=True,
            timeout=3,
            creationflags=flags,
        )
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            if not _find_profile_pids(root_only=False):
                return
            time.sleep(0.1)
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            timeout=3,
            creationflags=flags,
        )
    else:
        os.kill(pid, 15)


def get_kiosk_status() -> dict:
    with _KIOSK_LOCK:
        active = False
        pid = None
        if _KIOSK_PROCESS is not None and _KIOSK_PROCESS.poll() is None:
            active = True
            pid = _KIOSK_PROCESS.pid
        else:
            pids = _find_profile_pids()
            if pids:
                active = True
                pid = pids[0]
        return {"active": active, "pid": pid}


def launch_kiosk(origin: str | None = None) -> dict:
    global _KIOSK_PROCESS
    target_url = _sanitize_local_kiosk_url(origin)
    with _KIOSK_LOCK:
        if _KIOSK_PROCESS is not None and _KIOSK_PROCESS.poll() is None:
            return {
                "launched": True,
                "already_running": True,
                "pid": _KIOSK_PROCESS.pid,
                "url": target_url,
            }
        existing_pids = _find_profile_pids()
        if existing_pids:
            return {
                "launched": True,
                "already_running": True,
                "pid": existing_pids[0],
                "url": target_url,
            }

        chrome_exe = _find_chrome_executable()
        if not chrome_exe:
            raise RuntimeError("Google Chrome executable not found.")

        KIOSK_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        _ensure_clean_preferences()

        args = [
            chrome_exe,
            f"--user-data-dir={KIOSK_PROFILE_DIR}",
            "--kiosk",
            target_url,
            "--new-window",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-pinch",
            "--overscroll-history-navigation=0",
            "--disable-session-crashed-bubble",
            "--disable-infobars",
        ]
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )
        time.sleep(0.25)
        _KIOSK_PROCESS = proc
        return {
            "launched": True,
            "already_running": False,
            "pid": proc.pid,
            "url": target_url,
        }


def verify_secret_phrase(phrase: str) -> bool:
    candidate = phrase.strip().lower().encode("utf-8")
    expected = SECRET_PHRASE.encode("utf-8")
    return hmac.compare_digest(candidate, expected)


def _stop_kiosk_processes() -> bool:
    global _KIOSK_PROCESS
    with _KIOSK_LOCK:
        closed = False
        if _KIOSK_PROCESS is not None and _KIOSK_PROCESS.poll() is None:
            _terminate_pid(_KIOSK_PROCESS.pid)
            try:
                _KIOSK_PROCESS.wait(timeout=2)
            except Exception:
                pass
            closed = True
        _KIOSK_PROCESS = None
        remaining = _find_profile_pids(root_only=True) or _find_profile_pids(root_only=False)
        for pid in remaining:
            try:
                _terminate_pid(pid)
                closed = True
            except Exception:
                pass
        _ensure_clean_preferences()
        return closed


def schedule_kiosk_exit(delay_seconds: float = 0.12) -> dict:
    had_process = get_kiosk_status()["active"]
    if delay_seconds <= 0:
        closed = _stop_kiosk_processes()
        return {"unlocked": True, "closing": closed}
    timer = threading.Timer(delay_seconds, _stop_kiosk_processes)
    timer.daemon = True
    timer.start()
    return {"unlocked": True, "closing": had_process}
