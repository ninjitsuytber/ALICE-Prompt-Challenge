from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import re
import threading
import time

from dotenv import load_dotenv
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = DATA_DIR / "generated"
CONFIG_PATH = DATA_DIR / "config.json"

load_dotenv(BASE_DIR / ".env")

_CONFIG_KEYS = (
    "HF_TOKEN",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_API_TOKEN",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
)

_HF_SPACES: tuple[dict, ...] = (
    {
        "label": "Z-Image-Turbo",
        "subdomain": "mrfakename-z-image-turbo",
        "endpoint": "/generate_image",
        "timeout": 20.0,
        "build": lambda p, s: [p, 512, 512, 8, s, False],
    },
    {
        "label": "FLUX.1-schnell",
        "subdomain": "black-forest-labs-flux-1-schnell",
        "endpoint": "/infer",
        "timeout": 22.0,
        "build": lambda p, s: [p, s, False, 512, 512, 4],
    },
    {
        "label": "Z-Image-Turbo",
        "subdomain": "akhaliq-z-image-turbo",
        "endpoint": "/generate_image",
        "timeout": 20.0,
        "build": lambda p, s: [p, 512, 512, 8, s, False],
    },
    {
        "label": "Z-Image-Turbo",
        "subdomain": "laruss5-z-image-turbo",
        "endpoint": "/generate_image",
        "timeout": 20.0,
        "build": lambda p, s: [p, 512, 512, 8, s, False],
    },
    {
        "label": "Z-Image-Turbo Realism",
        "subdomain": "linoyts-z-image-turbo-realism",
        "endpoint": "/generate_image",
        "timeout": 20.0,
        "build": lambda p, s: [p, 512, 512, 8, s, False],
    },
    {
        "label": "Tongyi Z-Image-Turbo",
        "subdomain": "tongyi-mai-z-image-turbo",
        "endpoint": "/generate",
        "timeout": 28.0,
        "build": lambda p, s: [p, "1024x1024 ( 1:1 )", s, 8, 3.0, False, []],
    },
)

_pool_lock = threading.Lock()
_space_cursor = 0
_space_cooldowns: dict[str, float] = {}
_anon_zerogpu_cooldown_until = 0.0

_local_load_lock = threading.Lock()
_gpu_infer_lock = threading.Lock()
_local_pipe = None
_local_gpu_name: str | None = None
_local_init_failed = False


@dataclass(frozen=True)
class GenerationResult:
    file_path: Path
    public_url: str
    source: str
    fallback_reason: str | None = None


def _load_runtime_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    if CONFIG_PATH.exists():
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cfg = {str(k): str(v) for k, v in raw.items() if k in _CONFIG_KEYS and v}
        except Exception:
            cfg = {}
    for key in _CONFIG_KEYS:
        env_val = os.getenv(key, "").strip()
        if env_val and key not in cfg:
            cfg[key] = env_val
    return cfg


def save_runtime_keys(
    hf_token: str | None = None,
    cloudflare_account_id: str | None = None,
    cloudflare_api_token: str | None = None,
    gemini_key: str | None = None,
    openai_key: str | None = None,
) -> dict[str, bool | str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current = {}
    if CONFIG_PATH.exists():
        try:
            current = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            current = {}

    current = {k: v for k, v in current.items() if k in _CONFIG_KEYS and v}
    updates = {
        "HF_TOKEN": hf_token,
        "CLOUDFLARE_ACCOUNT_ID": cloudflare_account_id,
        "CLOUDFLARE_API_TOKEN": cloudflare_api_token,
        "GEMINI_API_KEY": gemini_key,
        "OPENAI_API_KEY": openai_key,
    }
    for k, val in updates.items():
        if val is not None:
            cleaned = val.strip()
            if cleaned:
                current[k] = cleaned
            else:
                current.pop(k, None)

    CONFIG_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return get_engine_status()


def _detect_local_gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return None


def _get_local_pipe():
    global _local_pipe, _local_gpu_name, _local_init_failed
    if _local_pipe is not None:
        return _local_pipe
    if _local_init_failed:
        return None
    with _local_load_lock:
        if _local_pipe is not None:
            return _local_pipe
        if _local_init_failed:
            return None
        try:
            import torch
            from diffusers import DDIMScheduler, DiffusionPipeline
            from huggingface_hub import hf_hub_download

            if not torch.cuda.is_available():
                _local_init_failed = True
                return None

            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            _local_gpu_name = str(torch.cuda.get_device_name(0))
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    "Lykon/dreamshaper-8",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                )
                lora_path = hf_hub_download(
                    "ByteDance/Hyper-SD",
                    "Hyper-SD15-8steps-CFG-lora.safetensors",
                    local_files_only=True,
                )
            except Exception:
                pipe = DiffusionPipeline.from_pretrained(
                    "Lykon/dreamshaper-8",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                lora_path = hf_hub_download(
                    "ByteDance/Hyper-SD",
                    "Hyper-SD15-8steps-CFG-lora.safetensors",
                )

            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora()
            pipe.unload_lora_weights()
            pipe.scheduler = DDIMScheduler.from_config(
                pipe.scheduler.config,
                timestep_spacing="trailing",
            )
            pipe.set_progress_bar_config(disable=True)
            pipe = pipe.to("cuda")

            gen = torch.Generator(device="cuda").manual_seed(0)
            with torch.inference_mode():
                _ = pipe(
                    prompt="minimalist white studio",
                    num_inference_steps=2,
                    guidance_scale=5.0,
                    width=512,
                    height=512,
                    generator=gen,
                ).images[0]
            torch.cuda.synchronize()
            _local_pipe = pipe
            return _local_pipe
        except Exception:
            _local_init_failed = True
            return None


def warmup_engine() -> None:
    if _detect_local_gpu_name():
        _get_local_pipe()


def _try_local_gpu(prompt: str, seed: int, output_path: Path) -> tuple[bool, str | None, str | None]:
    with _gpu_infer_lock:
        pipe = _get_local_pipe()
        if pipe is None:
            return False, None, "Local CUDA GPU model unavailable"
        try:
            import torch

            gen = torch.Generator(device="cuda").manual_seed(seed % 2147483647)
            with torch.inference_mode():
                image = pipe(
                    prompt=prompt,
                    negative_prompt="blurry, low quality, distorted, deformed, ugly, watermark, text, signature",
                    num_inference_steps=8,
                    guidance_scale=5.0,
                    width=512,
                    height=512,
                    generator=gen,
                ).images[0]
            torch.cuda.synchronize()
            buf = BytesIO()
            image.convert("RGB").save(buf, format="JPEG", quality=92)
            if _save_image_bytes(buf.getvalue(), output_path):
                gpu_label = _local_gpu_name or "CUDA GPU"
                short_gpu = gpu_label.replace("NVIDIA GeForce ", "")
                return True, f"Local GPU ({short_gpu} · Hyper-SD)", None
            return False, None, "Local GPU image save failed"
        except Exception as exc:
            return False, None, f"Local GPU error ({type(exc).__name__})"


def get_engine_status() -> dict[str, bool | str]:
    cfg = _load_runtime_config()
    has_hf = bool(cfg.get("HF_TOKEN"))
    has_cf = bool(cfg.get("CLOUDFLARE_ACCOUNT_ID") and cfg.get("CLOUDFLARE_API_TOKEN"))
    has_gemini = bool(cfg.get("GEMINI_API_KEY"))
    has_openai = bool(cfg.get("OPENAI_API_KEY"))
    gpu_name = _detect_local_gpu_name()
    has_local_gpu = gpu_name is not None

    if has_cf:
        active = "Cloudflare FLUX + Local GPU" if has_local_gpu else "Cloudflare FLUX + Free Multi-AI"
    elif has_hf:
        active = "HF ZeroGPU + Local GPU" if has_local_gpu else "HF ZeroGPU + Free Multi-AI"
    elif has_gemini:
        active = "Gemini Imagen + Local GPU" if has_local_gpu else "Gemini Imagen + Free Multi-AI"
    elif has_openai:
        active = "OpenAI DALL-E + Local GPU" if has_local_gpu else "OpenAI DALL-E + Free Multi-AI"
    elif has_local_gpu:
        short_gpu = gpu_name.replace("NVIDIA GeForce ", "")
        active = f"Local {short_gpu} (DreamShaper-8 Hyper-SD)"
    else:
        active = "Free Multi-AI (Z-Image / FLUX / SDXL)"

    return {
        "active_mode": active,
        "has_local_gpu": has_local_gpu,
        "has_hf_token": has_hf,
        "has_cloudflare_key": has_cf,
        "has_gemini_key": has_gemini,
        "has_openai_key": has_openai,
    }


def _save_image_bytes(raw_bytes: bytes, output_path: Path) -> bool:
    try:
        im = Image.open(BytesIO(raw_bytes)).convert("RGB")
        im = im.resize((512, 512), Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(output_path, format="JPEG", quality=92)
        return True
    except Exception:
        return False


def _extract_gradio_image_url(obj, base_url: str) -> str | None:
    if isinstance(obj, dict):
        if obj.get("url"):
            return str(obj["url"])
        if obj.get("path"):
            return f"{base_url}/gradio_api/file={obj['path']}"
        for v in obj.values():
            found = _extract_gradio_image_url(v, base_url)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _extract_gradio_image_url(item, base_url)
            if found:
                return found
    return None


def _call_hf_space(
    spec: dict,
    prompt: str,
    seed: int,
    output_path: Path,
    hf_token: str | None = None,
) -> tuple[bool, bool, str | None]:
    base = f"https://{spec['subdomain']}.hf.space"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    payload = {"data": spec["build"](prompt, seed % 2147483647)}
    try:
        with httpx.Client(timeout=spec["timeout"], follow_redirects=True) as client:
            resp = client.post(f"{base}/gradio_api/call{spec['endpoint']}", json=payload, headers=headers)
            if resp.status_code != 200:
                is_quota = resp.status_code == 429 or "quota" in resp.text.lower()
                return False, is_quota, f"{spec['label']} HTTP {resp.status_code}"
            event_id = resp.json().get("event_id")
            if not event_id:
                return False, False, f"{spec['label']} missing event_id"

            result_data = None
            with client.stream("GET", f"{base}/gradio_api/call{spec['endpoint']}/{event_id}", headers=headers) as stream:
                event_type = None
                for line in stream.iter_lines():
                    if line.startswith("event:"):
                        event_type = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        raw_data = line.split(":", 1)[1].strip()
                        if event_type == "error":
                            lower_err = raw_data.lower()
                            is_quota = any(tok in lower_err for tok in ("quota", "exceeded", "runs limit"))
                            return False, is_quota, f"{spec['label']} quota/error"
                        if event_type == "complete":
                            result_data = json.loads(raw_data)
                            break

            if result_data is None:
                return False, False, f"{spec['label']} incomplete stream"

            img_url = _extract_gradio_image_url(result_data, base)
            if not img_url:
                return False, False, f"{spec['label']} returned no image URL"
            dl_headers = headers if ".hf.space" in img_url else {}
            img_resp = client.get(img_url, headers=dl_headers)
            if img_resp.status_code == 200 and _save_image_bytes(img_resp.content, output_path):
                return True, False, None
            return False, False, f"{spec['label']} image fetch HTTP {img_resp.status_code}"
    except Exception as exc:
        return False, False, f"{spec['label']} error ({type(exc).__name__})"


def _try_hf_pool(prompt: str, seed: int, output_path: Path, hf_token: str | None = None) -> tuple[bool, str | None, str | None]:
    global _space_cursor, _anon_zerogpu_cooldown_until
    now = time.monotonic()

    with _pool_lock:
        if not hf_token and now < _anon_zerogpu_cooldown_until:
            return False, None, "ZeroGPU anonymous cooldown active"
        start = _space_cursor
        _space_cursor = (_space_cursor + 1) % len(_HF_SPACES)
        ordered = [
            _HF_SPACES[(start + i) % len(_HF_SPACES)]
            for i in range(len(_HF_SPACES))
            if now >= _space_cooldowns.get(_HF_SPACES[(start + i) % len(_HF_SPACES)]["subdomain"], 0.0)
        ]

    if not ordered:
        ordered = list(_HF_SPACES)

    max_attempts = 3 if hf_token else 2
    last_reason = "Hugging Face Spaces unavailable"
    for spec in ordered[:max_attempts]:
        ok, is_quota, reason = _call_hf_space(spec, prompt, seed, output_path, hf_token=hf_token)
        if ok:
            source_name = f"HF {spec['label']}"
            return True, source_name, None
        if reason:
            last_reason = reason
        with _pool_lock:
            if is_quota and not hf_token:
                _anon_zerogpu_cooldown_until = time.monotonic() + 300.0
                break
            _space_cooldowns[spec["subdomain"]] = time.monotonic() + (180.0 if is_quota else 90.0)

    return False, None, last_reason


def _try_cloudflare(
    prompt: str,
    seed: int,
    account_id: str,
    api_token: str,
    output_path: Path,
) -> tuple[bool, str | None]:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                url,
                headers={"Authorization": f"Bearer {api_token}"},
                json={"prompt": prompt, "steps": 4, "seed": seed % 2147483647},
            )
        if resp.status_code != 200:
            return False, f"Cloudflare HTTP {resp.status_code}"
        ct = resp.headers.get("content-type", "")
        if ct.startswith("image/"):
            if _save_image_bytes(resp.content, output_path):
                return True, None
        data = resp.json()
        b64 = (data.get("result") or {}).get("image")
        if b64 and _save_image_bytes(base64.b64decode(b64), output_path):
            return True, None
        return False, "Cloudflare returned no image data"
    except Exception as exc:
        return False, f"Cloudflare error ({type(exc).__name__})"


def _try_ai_horde(prompt: str, seed: int, output_path: Path) -> tuple[bool, str | None, str | None]:
    headers = {
        "apikey": "0000000000",
        "Client-Agent": "UNBOCS:2.0:arena@unbocs.local",
    }
    payload = {
        "prompt": prompt,
        "params": {
            "width": 512,
            "height": 512,
            "steps": 14,
            "cfg_scale": 6.5,
            "n": 1,
            "seed": str(seed % 2147483647),
        },
        "models": [
            "Juggernaut XL",
            "AlbedoBase XL (SDXL)",
            "AlbedoBase XL 3.1",
            "Deliberate",
        ],
        "nsfw": False,
        "censor_nsfw": True,
        "r2": True,
    }
    try:
        deadline = time.monotonic() + 38.0
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            submit_resp = client.post("https://aihorde.net/api/v2/generate/async", headers=headers, json=payload)
            if submit_resp.status_code == 403:
                time.sleep(0.6)
                payload["params"]["steps"] = 12
                payload["params"]["sampler_name"] = "k_euler"
                payload["models"] = ["Deliberate"]
                submit_resp = client.post("https://aihorde.net/api/v2/generate/async", headers=headers, json=payload)
            if submit_resp.status_code not in (200, 202):
                return False, None, f"AI Horde submit HTTP {submit_resp.status_code}"
            job_id = submit_resp.json().get("id")
            if not job_id:
                return False, None, "AI Horde missing job id"

            while time.monotonic() < deadline:
                time.sleep(1.5)
                chk = client.get(f"https://aihorde.net/api/v2/generate/check/{job_id}").json()
                if chk.get("done"):
                    status = client.get(f"https://aihorde.net/api/v2/generate/status/{job_id}").json()
                    generations = status.get("generations") or []
                    if not generations:
                        return False, None, "AI Horde returned no generations"
                    gen = generations[0]
                    img_url = gen.get("img")
                    model_name = gen.get("model") or "SDXL"
                    if img_url:
                        img_resp = client.get(img_url)
                        if img_resp.status_code == 200 and _save_image_bytes(img_resp.content, output_path):
                            return True, f"AI Horde ({model_name})", None
                    return False, None, "AI Horde image download failed"
                if not chk.get("is_possible", True) or chk.get("faulted"):
                    return False, None, "AI Horde job impossible or faulted"

        return False, None, "AI Horde request timed out"
    except Exception as exc:
        return False, None, f"AI Horde error ({type(exc).__name__})"


def _try_gemini(prompt: str, api_key: str, output_path: Path) -> tuple[bool, str | None]:
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        res = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config={"number_of_images": 1, "aspect_ratio": "1:1"},
        )
        if res.generated_images:
            raw = res.generated_images[0].image.image_bytes
            if _save_image_bytes(raw, output_path):
                return True, None
        return False, "Gemini returned no images"
    except Exception as exc:
        return False, f"Gemini error ({type(exc).__name__})"


def _try_openai(prompt: str, api_key: str, output_path: Path) -> tuple[bool, str | None]:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        res = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            response_format="b64_json",
            n=1,
        )
        b64 = res.data[0].b64_json
        if b64 and _save_image_bytes(base64.b64decode(b64), output_path):
            return True, None
        return False, "OpenAI returned no image data"
    except Exception as exc:
        return False, f"OpenAI error ({type(exc).__name__})"


_COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "cyan": (40, 225, 245),
    "teal": (30, 175, 185),
    "turquoise": (48, 200, 210),
    "magenta": (245, 55, 185),
    "pink": (245, 120, 185),
    "neon": (0, 240, 255),
    "purple": (145, 65, 225),
    "violet": (130, 75, 215),
    "lavender": (195, 175, 235),
    "blue": (40, 105, 225),
    "cobalt": (28, 78, 205),
    "indigo": (35, 38, 120),
    "red": (215, 45, 45),
    "crimson": (195, 30, 45),
    "orange": (245, 130, 45),
    "peach": (250, 195, 165),
    "sunset": (248, 145, 78),
    "yellow": (245, 210, 45),
    "gold": (240, 185, 65),
    "golden": (240, 185, 65),
    "amber": (235, 150, 40),
    "green": (45, 155, 85),
    "emerald": (22, 115, 75),
    "moss": (90, 130, 55),
    "white": (242, 242, 240),
    "cream": (245, 235, 218),
    "grey": (135, 140, 145),
    "gray": (135, 140, 145),
    "concrete": (205, 204, 198),
    "black": (20, 22, 26),
    "dark": (24, 28, 36),
    "slate": (34, 38, 44),
    "brown": (125, 78, 42),
    "wood": (135, 84, 48),
    "brass": (205, 165, 75),
}


def _lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _synthesize_prompt_image(prompt: str, output_path: Path, seed: int) -> None:
    text = prompt.lower()
    tokens = set(re.findall(r"[a-z0-9-]+", text))
    rng = random.Random(seed)
    w, h = 512, 512

    found_colors = [_COLOR_MAP[k] for k in _COLOR_MAP if k in tokens]
    is_dark_scene = bool(
        tokens & {"night", "midnight", "dark", "cyberpunk", "neon", "synthwave", "slate", "space", "galaxy", "velvet"}
    )
    is_bright_minimal = bool(
        tokens & {"white", "minimalist", "bauhaus", "salt", "concrete", "sunlit", "plaster", "pastel"}
    )

    if is_dark_scene:
        top_col = found_colors[0] if found_colors else (16, 22, 42)
        top_col = tuple(max(10, int(c * 0.28)) for c in top_col)
        bot_col = (14, 16, 24)
        accent_1 = found_colors[0] if len(found_colors) >= 1 else (0, 230, 255)
        accent_2 = found_colors[1] if len(found_colors) >= 2 else (245, 60, 180)
        accent_3 = found_colors[2] if len(found_colors) >= 3 else (245, 185, 65)
    elif is_bright_minimal:
        top_col = (238, 237, 234)
        bot_col = (225, 224, 220)
        if "peach" in tokens or "lavender" in tokens:
            top_col = (210, 196, 235)
            bot_col = (250, 212, 188)
        accent_1 = found_colors[0] if len(found_colors) >= 1 else (32, 82, 200)
        accent_2 = found_colors[1] if len(found_colors) >= 2 else (200, 38, 46)
        accent_3 = found_colors[2] if len(found_colors) >= 3 else (240, 205, 48)
    else:
        top_col = found_colors[0] if len(found_colors) >= 1 else (120, 175, 205)
        bot_col = found_colors[1] if len(found_colors) >= 2 else (215, 225, 230)
        accent_1 = found_colors[0] if len(found_colors) >= 1 else (235, 145, 70)
        accent_2 = found_colors[1] if len(found_colors) >= 2 else (55, 155, 175)
        accent_3 = found_colors[2] if len(found_colors) >= 3 else (240, 205, 95)

    arr = np.zeros((h, w, 3), dtype=np.uint8)
    ys = np.linspace(0.0, 1.0, h)[:, None]
    for ch in range(3):
        arr[:, :, ch] = np.clip(top_col[ch] * (1 - ys) + bot_col[ch] * ys, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    if tokens & {"overhead", "top-down", "espresso", "coffee", "cup", "koi", "pond", "water", "pebbles"}:
        if tokens & {"espresso", "coffee", "cup", "galaxy", "nebula"}:
            draw.ellipse([96, 96, 416, 416], fill=(242, 242, 240, 255))
            draw.polygon([(400, 345), (458, 385), (438, 412), (382, 372)], fill=(228, 228, 226, 255))
            draw.ellipse([126, 126, 386, 386], fill=(38, 26, 18, 255))
            for r in range(120, 12, -8):
                t = 1.0 - r / 120.0
                col = _lerp_color((185, 125, 65), (255, 235, 185), t)
                draw.arc([256 - r, 256 - r, 256 + r, 256 + r], r * 3, r * 3 + 220, fill=(*col, 190), width=5)
            for _ in range(90):
                ang = rng.uniform(0, math.tau)
                rad = rng.uniform(5, 115)
                px, py = int(256 + math.cos(ang) * rad), int(256 + math.sin(ang) * rad)
                draw.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(255, 245, 210, 220))
        else:
            draw.rectangle([0, 0, w, h], fill=(75, 168, 182, 255))
            for _ in range(36):
                px, py = rng.randint(20, 492), rng.randint(20, 492)
                rx, ry = rng.randint(14, 42), rng.randint(12, 34)
                shade = rng.randint(95, 155)
                draw.ellipse([px - rx, py - ry, px + rx, py + ry], fill=(shade, shade + 12, shade + 18, 210))
            for idx in range(3):
                fx, fy = 140 + idx * 115, 220 + (idx % 2) * 75
                draw.polygon([(fx - 35, fy), (fx + 10, fy - 24), (fx + 45, fy), (fx + 10, fy + 24)], fill=(238, 235, 235, 245))
                draw.polygon([(fx - 10, fy - 12), (fx + 18, fy - 16), (fx + 14, fy + 10)], fill=(205, 42, 52, 235))
    else:
        horizon_y = int(h * 0.56)
        if tokens & {"wireframe", "grid", "synthwave", "retro", "highway", "car"}:
            sun_r = 115
            sx, sy = w // 2, horizon_y - 48
            for y in range(sy - sun_r, sy + sun_r):
                dy = y - sy
                if abs(dy) >= sun_r:
                    continue
                half_w = int(math.sqrt(sun_r * sun_r - dy * dy))
                if y > sy + 12 and (y // 10) % 2 == 0:
                    continue
                t = (y - (sy - sun_r)) / (2 * sun_r)
                col = _lerp_color((255, 195, 70), (240, 45, 155), t)
                draw.line([(sx - half_w, y), (sx + half_w, y)], fill=(*col, 245))
            draw.rectangle([0, horizon_y, w, h], fill=(22, 12, 42, 255))
            for i in range(-10, 11):
                draw.line([(w // 2 + i * 10, horizon_y), (w // 2 + i * 75, h)], fill=(185, 70, 255, 190), width=2)
            for row in range(1, 10):
                ry = horizon_y + int((row / 10.0) ** 1.8 * (h - horizon_y))
                draw.line([(0, ry), (w, ry)], fill=(185, 70, 255, 175), width=2)
            if tokens & {"car", "sports", "vehicle"}:
                draw.polygon([(145, 430), (195, 360), (317, 360), (367, 430)], fill=(18, 18, 22, 255))
                draw.rectangle([160, 405, 352, 420], fill=(255, 45, 55, 235))

        elif tokens & {"planet", "moon", "astronaut", "salt", "desert"}:
            px, py, pr = w // 2, int(h * 0.40), 110
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(*accent_1, 170))
            draw.arc([px - 185, py - 45, px + 185, py + 45], -10, 190, fill=(245, 235, 245, 190), width=8)
            draw.rectangle([0, horizon_y, w, h], fill=(245, 245, 248, 255))
            draw.rectangle([240, 275, 272, 345], fill=(248, 248, 250, 255), outline=(120, 120, 130, 255))
            draw.ellipse([242, 245, 270, 273], fill=(248, 248, 250, 255), outline=(100, 100, 110, 255))
            draw.ellipse([248, 252, 264, 265], fill=(195, 145, 55, 255))
            draw.polygon([(245, 345), (410, 410), (430, 404), (268, 345)], fill=(85, 95, 115, 110))

        elif tokens & {"bauhaus", "sphere", "pyramid", "arch", "plinth", "geometric"}:
            draw.rectangle([64, 345, 448, 465], fill=(250, 250, 250, 255), outline=(210, 210, 210, 255))
            draw.ellipse([105, 245, 215, 355], fill=(*accent_1, 255))
            draw.polygon([(195, 365), (255, 235), (315, 365)], fill=(*accent_2, 255))
            draw.arc([285, 195, 405, 355], 180, 360, fill=(*accent_3, 255), width=28)
            draw.rectangle([285, 275, 313, 355], fill=(*accent_3, 255))
            draw.rectangle([377, 275, 405, 355], fill=(*accent_3, 255))

        elif tokens & {"alley", "ramen", "lantern", "lanterns", "cyberpunk", "rain", "street"}:
            draw.rectangle([0, 0, 165, h], fill=(22, 18, 28, 255))
            draw.rectangle([365, 0, w, h], fill=(18, 22, 34, 255))
            draw.rectangle([45, 185, 210, 415], fill=(88, 46, 24, 255))
            draw.rectangle([75, 215, 195, 330], fill=(245, 175, 85, 225))
            draw.rectangle([0, 395, w, h], fill=(18, 22, 32, 245))
            for i in range(6):
                lx = 155 + i * 48
                ly = 110 + (i % 2) * 35
                col = accent_1 if i % 2 == 0 else accent_2
                draw.ellipse([lx - 14, ly - 18, lx + 14, ly + 18], fill=(*col, 240))
                draw.rectangle([lx - 10, 410, lx + 10, 495], fill=(*col, 70))

        elif tokens & {"forest", "mushroom", "mushrooms", "pine", "fireflies", "trees"}:
            draw.rectangle([0, 340, w, h], fill=(14, 34, 24, 255))
            for tx in (35, 130, 365, 455):
                draw.rectangle([tx, 0, tx + 44, 420], fill=(16, 20, 22, 240))
            for mx, my, mr in ((105, 405, 26), (175, 385, 32), (245, 425, 22), (385, 410, 28)):
                draw.rectangle([mx - 4, my, mx + 4, my + 48], fill=(145, 225, 245, 220))
                draw.chord([mx - mr, my - mr, mx + mr, my + mr], 180, 360, fill=(*accent_1, 240))
            for _ in range(30):
                fx, fy = rng.randint(25, 487), rng.randint(80, 440)
                draw.ellipse([fx - 3, fy - 3, fx + 3, fy + 3], fill=(250, 225, 120, 220))

        elif tokens & {"fox", "clockwork", "steampunk", "mechanical", "gears"}:
            draw.rectangle([60, 395, 452, 485], fill=(34, 36, 40, 255))
            draw.polygon([(195, 410), (225, 195), (295, 245), (345, 410)], fill=(235, 232, 225, 255))
            draw.polygon([(170, 175), (220, 120), (255, 175)], fill=(235, 232, 225, 255))
            draw.polygon([(195, 135), (205, 85), (225, 135)], fill=(205, 165, 75, 255))
            draw.ellipse([245, 280, 305, 340], fill=(195, 155, 65, 255))
            draw.ellipse([195, 150, 209, 162], fill=(245, 155, 30, 255))

        elif tokens & {"tea", "house", "floating", "island", "clouds", "waterfall"}:
            for cx, cy, cr in ((110, 355, 75), (410, 330, 80), (140, 140, 65)):
                draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(250, 220, 185, 190))
            draw.polygon([(145, 250), (365, 250), (320, 375), (245, 415), (170, 335)], fill=(65, 78, 52, 255))
            draw.rectangle([215, 170, 315, 248], fill=(120, 74, 42, 255))
            draw.rectangle([232, 188, 298, 235], fill=(255, 205, 105, 240))
            draw.polygon([(195, 172), (265, 128), (335, 172)], fill=(48, 42, 40, 255))
            draw.rectangle([195, 265, 214, 465], fill=(195, 230, 250, 210))

        elif tokens & {"library", "concrete", "staircase", "stairs", "atrium", "brutalist", "sunbeams"}:
            draw.rectangle([0, 0, 120, h], fill=(195, 194, 188, 255))
            draw.rectangle([392, 0, w, h], fill=(190, 189, 183, 255))
            draw.polygon([(115, 390), (395, 220), (395, 255), (115, 425)], fill=(235, 233, 228, 255))
            draw.polygon([(115, 210), (395, 345), (395, 380), (115, 245)], fill=(218, 216, 210, 255))
            draw.polygon([(120, 0), (260, 0), (420, 512), (250, 512)], fill=(255, 238, 195, 85))

        else:
            cx, cy = w // 2, h // 2
            draw.ellipse([cx - 125, cy - 125, cx + 125, cy + 125], fill=(*accent_1, 210))
            draw.polygon([(cx - 110, cy + 120), (cx, cy - 95), (cx + 110, cy + 120)], fill=(*accent_2, 185))
            draw.rectangle([cx - 150, cy + 85, cx + 150, cy + 145], fill=(*accent_3, 205))

    soft = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    out = Image.blend(img, soft, alpha=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path, format="JPEG", quality=90)


def generate_image_from_prompt(prompt: str, round_id: str) -> GenerationResult:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{round_id.lower()}.jpg"
    output_path = GENERATED_DIR / safe_name
    public_url = f"/generated/{safe_name}"

    seed = int(hashlib.sha256(f"{round_id}:{prompt}".encode("utf-8")).hexdigest()[:8], 16)
    cfg = _load_runtime_config()
    reasons: list[str] = []

    if cfg.get("CLOUDFLARE_ACCOUNT_ID") and cfg.get("CLOUDFLARE_API_TOKEN"):
        ok, err = _try_cloudflare(
            prompt,
            seed,
            cfg["CLOUDFLARE_ACCOUNT_ID"],
            cfg["CLOUDFLARE_API_TOKEN"],
            output_path,
        )
        if ok:
            return GenerationResult(output_path, public_url, "Cloudflare FLUX.1-schnell")
        if err:
            reasons.append(err)

    hf_token = cfg.get("HF_TOKEN")
    if hf_token:
        ok, src, err = _try_hf_pool(prompt, seed, output_path, hf_token=hf_token)
        if ok and src:
            return GenerationResult(output_path, public_url, src)
        if err:
            reasons.append(err)

    if cfg.get("GEMINI_API_KEY"):
        ok, err = _try_gemini(prompt, cfg["GEMINI_API_KEY"], output_path)
        if ok:
            return GenerationResult(output_path, public_url, "Gemini Imagen")
        if err:
            reasons.append(err)

    if cfg.get("OPENAI_API_KEY"):
        ok, err = _try_openai(prompt, cfg["OPENAI_API_KEY"], output_path)
        if ok:
            return GenerationResult(output_path, public_url, "OpenAI DALL-E")
        if err:
            reasons.append(err)

    ok, src, err = _try_local_gpu(prompt, seed, output_path)
    if ok and src:
        return GenerationResult(output_path, public_url, src)
    if err:
        reasons.append(err)

    if not hf_token:
        ok, src, err = _try_hf_pool(prompt, seed, output_path, hf_token=None)
        if ok and src:
            return GenerationResult(output_path, public_url, src)
        if err:
            reasons.append(err)

    ok, src, err = _try_ai_horde(prompt, seed, output_path)
    if ok and src:
        return GenerationResult(output_path, public_url, src)
    if err:
        reasons.append(err)

    _synthesize_prompt_image(prompt, output_path, seed)
    return GenerationResult(
        file_path=output_path,
        public_url=public_url,
        source="Offline Synthesis Fallback",
        fallback_reason="; ".join(reasons) if reasons else None,
    )
