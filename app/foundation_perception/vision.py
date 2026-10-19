from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class VisionEmbedding:
    vector: np.ndarray
    features: dict[str, float]
    backend: str
    uncertainty: float


def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).ravel()
    p = hist / max(float(hist.sum()), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / 6.0)


def _spatial_pyramid(gray: np.ndarray) -> list[float]:
    out: list[float] = []
    for cells in (1, 2, 4):
        h, w = gray.shape
        for y in range(cells):
            for x in range(cells):
                tile = gray[y*h//cells:(y+1)*h//cells, x*w//cells:(x+1)*w//cells]
                out.extend([float(tile.mean()) / 255.0, float(tile.std()) / 128.0])
    return out


def encode_frame(frame_bgr: np.ndarray) -> VisionEmbedding:
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Empty image frame")
    frame = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 60, 160)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    edge_density = float((edges > 0).mean())
    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 128.0)
    saturation = float(hsv[..., 1].mean() / 255.0)
    entropy = _entropy(gray)
    vertical = float(np.abs(gx).mean() / 255.0)
    horizontal = float(np.abs(gy).mean() / 255.0)
    texture = float(magnitude.mean() / 255.0)
    overexposed = float((gray > 245).mean())
    underexposed = float((gray < 15).mean())
    features = {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "edge_density": edge_density,
        "focus_laplacian": lap_var,
        "visual_entropy": entropy,
        "vertical_structure": vertical,
        "horizontal_structure": horizontal,
        "texture_energy": texture,
        "overexposed_fraction": overexposed,
        "underexposed_fraction": underexposed,
    }
    color_hist: list[float] = []
    for channel in cv2.split(hsv):
        hist = cv2.calcHist([channel], [0], None, [16], [0, 256]).ravel()
        hist = hist / max(float(hist.sum()), 1.0)
        color_hist.extend(hist.astype(float).tolist())
    vector = np.asarray(
        list(features.values()) + _spatial_pyramid(gray) + color_hist,
        dtype=np.float32,
    )
    vector /= max(float(np.linalg.norm(vector)), 1e-8)
    quality_penalty = min(1.0, underexposed + overexposed + (0.25 if lap_var < 25 else 0.0))
    uncertainty = float(np.clip(0.15 + 0.7 * quality_penalty, 0.05, 0.95))
    return VisionEmbedding(vector=vector, features=features, backend="opencv-governed-v1", uncertainty=uncertainty)


def encode_image(path: str | Path) -> VisionEmbedding:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Unable to decode image: {path}")
    return encode_frame(frame)


def inspect_video(path: str | Path, interval_s: float = 2.0, max_frames: int = 300) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = count / fps if fps > 0 else 0.0
    step = max(1, int(round(max(fps, 1.0) * interval_s)))
    rows, vectors = [], []
    index = 0
    while len(rows) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break
        emb = encode_frame(frame)
        rows.append({"time_s": index / max(fps, 1.0), **emb.features, "uncertainty": emb.uncertainty})
        vectors.append(emb.vector)
        index += step
        if count and index >= count:
            break
    cap.release()
    if not vectors:
        raise ValueError("Video yielded no decodable frames")
    matrix = np.vstack(vectors)
    pooled = matrix.mean(axis=0)
    pooled /= max(float(np.linalg.norm(pooled)), 1e-8)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {
        "backend": "opencv-governed-v1",
        "video_sha256": digest,
        "fps": fps,
        "frame_count": count,
        "duration_s": duration,
        "samples": rows,
        "embedding": pooled.astype(float).tolist(),
        "mean_uncertainty": float(np.mean([r["uncertainty"] for r in rows])),
        "semantic_scope": "operational visual representation; not autonomous heritage meaning",
    }
