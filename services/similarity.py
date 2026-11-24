from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.models as tv_models


@dataclass(frozen=True)
class SimilarityResult:
    overall: float
    semantic: float
    style: float
    color: float
    composition: float

    def to_dict(self) -> dict[str, float]:
        return {
            "overall": self.overall,
            "semantic": self.semantic,
            "style": self.style,
            "color": self.color,
            "composition": self.composition,
        }


class _ResNetFeatureExtractor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: torch.nn.Module | None = None
        self._transform = None
        self._feats: dict[str, torch.Tensor] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        weights = tv_models.ResNet50_Weights.DEFAULT
        model = tv_models.resnet50(weights=weights).eval()
        model.layer2.register_forward_hook(
            lambda _, __, out: self._feats.__setitem__("l2", out.detach())
        )
        model.avgpool.register_forward_hook(
            lambda _, __, out: self._feats.__setitem__("pool", out.detach())
        )
        self._model = model
        self._transform = weights.transforms()

    def extract(self, img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            self._ensure_loaded()
            assert self._model is not None and self._transform is not None
            tensor = self._transform(img).unsqueeze(0)
            with torch.no_grad():
                _ = self._model(tensor)

            sem = self._feats["pool"].flatten().cpu().numpy().astype(np.float32)
            sem /= np.linalg.norm(sem) + 1e-8

            f2 = self._feats["l2"].squeeze(0)
            c, h, w = f2.shape
            flat = f2.view(c, h * w)
            gram = (flat @ flat.t()) / float(c * h * w)
            g_vec = gram.flatten().cpu().numpy().astype(np.float32)
            g_vec = g_vec - g_vec.mean()
            g_vec /= np.linalg.norm(g_vec) + 1e-8
            return sem, g_vec


_EXTRACTOR = _ResNetFeatureExtractor()
_REF_CACHE_LOCK = threading.Lock()
_REF_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def warmup_similarity() -> None:
    with _EXTRACTOR._lock:
        _EXTRACTOR._ensure_loaded()


def _get_reference_features(reference_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = str(reference_path.resolve())
    with _REF_CACHE_LOCK:
        cached = _REF_CACHE.get(key)
    if cached is not None:
        return cached
    ref_img = Image.open(reference_path).convert("RGB")
    ref_256 = np.array(ref_img.resize((256, 256), Image.Resampling.LANCZOS))
    sem_ref, gram_ref = _EXTRACTOR.extract(ref_img)
    val = (ref_256, sem_ref, gram_ref)
    with _REF_CACHE_LOCK:
        _REF_CACHE[key] = val
    return val


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _stretch(value: float, low: float, high: float, gamma: float = 1.0) -> float:
    norm = _clamp01((value - low) / (high - low))
    return float(norm ** gamma)


def _compute_color_score(ref_rgb: np.ndarray, gen_rgb: np.ndarray) -> float:
    ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gen_lab = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    h1 = cv2.calcHist([ref_lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    h2 = cv2.calcHist([gen_lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    h1 /= h1.sum() + 1e-8
    h2 /= h2.sum() + 1e-8
    bhatt = float(np.sum(np.sqrt(np.maximum(h1 * h2, 0.0))))
    hist_sim = _stretch(bhatt, 0.20, 0.96, gamma=0.9)

    grid1 = ref_lab.reshape(4, 64, 4, 64, 3).mean(axis=(1, 3)).flatten()
    grid2 = gen_lab.reshape(4, 64, 4, 64, 3).mean(axis=(1, 3)).flatten()
    rmse = float(np.linalg.norm(grid1 - grid2) / np.sqrt(len(grid1)))
    spatial_sim = float(np.exp(-rmse / 42.0))

    return _clamp01(0.60 * hist_sim + 0.40 * spatial_sim)


def _compute_composition_score(ref_rgb: np.ndarray, gen_rgb: np.ndarray) -> float:
    ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gen_gray = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    l1 = cv2.resize(ref_gray, (16, 16), interpolation=cv2.INTER_AREA).flatten()
    l2 = cv2.resize(gen_gray, (16, 16), interpolation=cv2.INTER_AREA).flatten()
    l1 = (l1 - l1.mean()) / (l1.std() + 1e-5)
    l2 = (l2 - l2.mean()) / (l2.std() + 1e-5)
    lum_corr = float(np.dot(l1, l2) / len(l1))
    lum_sim = _stretch(lum_corr, -0.10, 0.98, gamma=0.95)

    def edge_vec(g: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        vec = cv2.resize(mag, (16, 16), interpolation=cv2.INTER_AREA).flatten()
        return vec / (np.linalg.norm(vec) + 1e-8)

    edge_cos = float(np.dot(edge_vec(ref_gray), edge_vec(gen_gray)))
    edge_sim = _stretch(edge_cos, 0.30, 0.97, gamma=0.95)

    ssim_val = float(
        ssim(
            cv2.resize(ref_gray, (128, 128), interpolation=cv2.INTER_AREA),
            cv2.resize(gen_gray, (128, 128), interpolation=cv2.INTER_AREA),
            data_range=255.0,
        )
    )
    ssim_sim = _stretch(ssim_val, 0.05, 0.92, gamma=0.85)

    return _clamp01(0.45 * lum_sim + 0.35 * edge_sim + 0.20 * ssim_sim)


def compare_images(reference_path: str | Path, generated_path: str | Path) -> SimilarityResult:
    ref_p = Path(reference_path)
    ref_256, sem_ref, gram_ref = _get_reference_features(ref_p)
    gen_img = Image.open(generated_path).convert("RGB")
    gen_256 = np.array(gen_img.resize((256, 256), Image.Resampling.LANCZOS))

    if np.mean(np.abs(ref_256.astype(np.int16) - gen_256.astype(np.int16))) < 0.5:
        return SimilarityResult(
            overall=100.0,
            semantic=100.0,
            style=100.0,
            color=100.0,
            composition=100.0,
        )

    sem_gen, gram_gen = _EXTRACTOR.extract(gen_img)

    raw_sem = float(np.dot(sem_ref, sem_gen))
    raw_gram = float(np.dot(gram_ref, gram_gen))

    semantic = _stretch(raw_sem, 0.12, 0.95, gamma=0.85)
    style = _stretch(raw_gram, 0.72, 0.995, gamma=0.95)
    color = _compute_color_score(ref_256, gen_256)
    composition = _compute_composition_score(ref_256, gen_256)

    overall = _clamp01(
        0.45 * semantic + 0.20 * style + 0.20 * color + 0.15 * composition
    )

    return SimilarityResult(
        overall=round(overall * 100.0, 1),
        semantic=round(semantic * 100.0, 1),
        style=round(style * 100.0, 1),
        color=round(color * 100.0, 1),
        composition=round(composition * 100.0, 1),
    )
