from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.foundation_perception.vision import encode_frame


@dataclass(frozen=True)
class FramePlaceState:
    time_s: float
    attributes: dict[str, float]
    cues: list[dict[str, Any]]
    uncertainty: float


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _line_structure(gray: np.ndarray) -> dict[str, float]:
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45,
                            minLineLength=max(20, gray.shape[1] // 10), maxLineGap=12)
    vertical = horizontal = diagonal = 0.0
    convergence_x: list[float] = []
    if lines is not None:
        for item in lines[:, 0, :]:
            x1, y1, x2, y2 = map(float, item)
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180
            length = math.hypot(x2 - x1, y2 - y1)
            if 70 <= angle <= 110:
                vertical += length
            elif angle <= 20 or angle >= 160:
                horizontal += length
            else:
                diagonal += length
                if abs(y2 - y1) > 1:
                    convergence_x.append((x1 + x2) / 2 / gray.shape[1])
    denom = max(vertical + horizontal + diagonal, 1.0)
    convergence = 0.0 if len(convergence_x) < 2 else 1.0 - min(float(np.std(convergence_x)) * 2.0, 1.0)
    return {
        "vertical_line_ratio": vertical / denom,
        "horizontal_line_ratio": horizontal / denom,
        "diagonal_line_ratio": diagonal / denom,
        "perspective_convergence_proxy": _clip(convergence),
    }


def _segmentation_proxies(frame: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    vegetation = ((h >= 30) & (h <= 95) & (s > 45) & (v > 35)).mean()
    sky = ((h >= 85) & (h <= 135) & (s > 25) & (v > 80)).mean()
    stone = ((h >= 5) & (h <= 35) & (s < 115) & (v > 45) & (v < 235)).mean()
    dark = (v < 45).mean()
    bright = (v > 210).mean()
    warm = (((h <= 25) | (h >= 170)) & (s > 35)).mean()
    return {
        "vegetation_fraction": float(vegetation),
        "sky_fraction": float(sky),
        "stone_like_fraction": float(stone),
        "dark_region_fraction": float(dark),
        "bright_region_fraction": float(bright),
        "warm_material_fraction": float(warm),
    }


def _arch_proxy(gray: np.ndarray) -> tuple[float, list[int] | None]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    best = 0.0
    best_box = None
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area < 0.015 * w * h or area > 0.55 * w * h or ch < cw * 0.75:
            continue
        roi = contour[:, 0, :]
        upper = roi[roi[:, 1] < y + ch * 0.55]
        if len(upper) < 6:
            continue
        center_x = x + cw / 2
        curvature = 1.0 - min(float(np.mean(np.abs(upper[:, 0] - center_x))) / max(cw / 2, 1), 1.0)
        symmetry = 1.0 - min(abs(float(np.mean(upper[:, 0])) - center_x) / max(cw / 2, 1), 1.0)
        score = _clip(0.55 * curvature + 0.45 * symmetry) * min(area / (0.12 * w * h), 1.0)
        if score > best:
            best, best_box = score, [int(x), int(y), int(cw), int(ch)]
    return best, best_box


def analyze_frame(frame_bgr: np.ndarray, time_s: float) -> FramePlaceState:
    small = cv2.resize(frame_bgr, (320, 240), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    foundation = encode_frame(small)
    line = _line_structure(gray)
    seg = _segmentation_proxies(small)
    arch_score, arch_box = _arch_proxy(gray)

    center = gray[:, gray.shape[1] // 3: 2 * gray.shape[1] // 3]
    side = np.concatenate([gray[:, : gray.shape[1] // 4], gray[:, -gray.shape[1] // 4:]], axis=1)
    center_brightness = float(center.mean() / 255.0)
    side_darkness = 1.0 - float(side.mean() / 255.0)
    openness = _clip(0.38 * seg["sky_fraction"] * 3.0 + 0.25 * center_brightness + 0.22 * line["perspective_convergence_proxy"] + 0.15 * seg["bright_region_fraction"])
    enclosure = _clip(0.42 * side_darkness + 0.30 * line["vertical_line_ratio"] + 0.18 * (1.0 - seg["sky_fraction"] * 4.0) + 0.10 * foundation.features["edge_density"])
    depth = _clip(0.55 * line["perspective_convergence_proxy"] + 0.25 * center_brightness + 0.20 * line["diagonal_line_ratio"])
    complexity = _clip(0.44 * foundation.features["visual_entropy"] + 0.36 * min(foundation.features["edge_density"] * 5.0, 1.0) + 0.20 * foundation.features["texture_energy"] * 3.0)
    symmetry = _clip(1.0 - float(np.mean(np.abs(gray[:, :160].astype(float) - np.fliplr(gray[:, 160:]).astype(float)))) / 128.0)

    attributes = {
        "openness_proxy": openness,
        "enclosure_proxy": enclosure,
        "visual_depth_proxy": depth,
        "visual_complexity_proxy": complexity,
        "bilateral_symmetry_proxy": symmetry,
        "illumination": foundation.features["brightness"],
        "edge_density": foundation.features["edge_density"],
        "focus_quality": _clip(math.log1p(foundation.features["focus_laplacian"]) / 8.0),
        **line,
        **seg,
    }
    cues: list[dict[str, Any]] = []
    if arch_score >= 0.48:
        cues.append({"cue": "arched_opening_candidate", "confidence": round(arch_score, 4), "bbox_xywh": arch_box, "status": "probable_visual_cue"})
    if seg["stone_like_fraction"] >= 0.22:
        cues.append({"cue": "stone_like_surface", "confidence": round(_clip(seg["stone_like_fraction"] * 1.8), 4), "status": "appearance_only"})
    if seg["vegetation_fraction"] >= 0.08:
        cues.append({"cue": "vegetation_visible", "confidence": round(_clip(seg["vegetation_fraction"] * 3.0), 4), "status": "visual_proxy"})
    if openness >= 0.62:
        cues.append({"cue": "open_spatial_condition", "confidence": round(openness, 4), "status": "spatial_proxy"})
    if enclosure >= 0.68:
        cues.append({"cue": "enclosed_spatial_condition", "confidence": round(enclosure, 4), "status": "spatial_proxy"})
    quality_uncertainty = foundation.uncertainty
    cue_uncertainty = 0.25 if cues else 0.42
    uncertainty = _clip(0.65 * quality_uncertainty + 0.35 * cue_uncertainty)
    return FramePlaceState(time_s=float(time_s), attributes=attributes, cues=cues, uncertainty=uncertainty)


def _vector(state: FramePlaceState) -> np.ndarray:
    keys = ["openness_proxy", "enclosure_proxy", "visual_depth_proxy", "visual_complexity_proxy", "illumination", "vegetation_fraction", "stone_like_fraction"]
    return np.asarray([state.attributes[k] for k in keys], dtype=np.float32)


def _zones(states: list[FramePlaceState], transition_threshold: float = 0.42) -> list[dict[str, Any]]:
    if not states:
        return []
    zones, start = [], 0
    for idx in range(1, len(states)):
        distance = float(np.linalg.norm(_vector(states[idx]) - _vector(states[idx - 1])))
        if distance >= transition_threshold:
            zones.append(_summarize_zone(states[start:idx], len(zones) + 1))
            start = idx
    zones.append(_summarize_zone(states[start:], len(zones) + 1))
    return zones


def _summarize_zone(states: list[FramePlaceState], number: int) -> dict[str, Any]:
    keys = states[0].attributes.keys()
    means = {k: float(np.mean([s.attributes[k] for s in states])) for k in keys}
    cues: dict[str, list[float]] = {}
    for state in states:
        for cue in state.cues:
            cues.setdefault(cue["cue"], []).append(float(cue["confidence"]))
    label = "mixed_spatial_condition"
    if means["openness_proxy"] > means["enclosure_proxy"] + 0.15:
        label = "comparatively_open_zone"
    elif means["enclosure_proxy"] > means["openness_proxy"] + 0.15:
        label = "comparatively_enclosed_zone"
    return {
        "zone_id": f"zone-{number:03d}",
        "start_s": states[0].time_s,
        "end_s": states[-1].time_s,
        "label": label,
        "mean_attributes": means,
        "persistent_cues": [{"cue": key, "mean_confidence": float(np.mean(vals)), "observations": len(vals)} for key, vals in sorted(cues.items())],
        "mean_uncertainty": float(np.mean([s.uncertainty for s in states])),
    }


def _scene_graph(zones: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = []
    edges = []
    for zone in zones:
        nodes.append({"id": zone["zone_id"], "type": "observed_route_zone", "label": zone["label"], "attributes": zone["mean_attributes"], "cues": zone["persistent_cues"], "uncertainty": zone["mean_uncertainty"]})
    for left, right in zip(zones, zones[1:]):
        changes = {k: right["mean_attributes"][k] - left["mean_attributes"][k] for k in ("openness_proxy", "enclosure_proxy", "illumination", "visual_depth_proxy")}
        edges.append({"source": left["zone_id"], "target": right["zone_id"], "relation": "temporal_transition", "attribute_changes": changes})
    return {"nodes": nodes, "edges": edges, "semantic_scope": "visual and spatial observations only; no experiential or cultural meaning inferred"}


def inspect_place_video(path: str | Path, interval_s: float = 1.0, max_frames: int = 900) -> dict[str, Any]:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    step = max(1, int(round(max(fps, 1.0) * interval_s)))
    states: list[FramePlaceState] = []
    index = 0
    while len(states) < max_frames and (frame_count <= 0 or index < frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break
        states.append(analyze_frame(frame, index / max(fps, 1.0)))
        index += step
    cap.release()
    if not states:
        raise ValueError("Video yielded no decodable frames")
    zones = _zones(states)
    return {
        "schema": "resonant-place-perception-1.0",
        "backend": "opencv-governed-place-v1",
        "video_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": duration,
        "sample_interval_s": interval_s,
        "samples": [{"time_s": s.time_s, **s.attributes, "cues": s.cues, "uncertainty": s.uncertainty} for s in states],
        "zones": zones,
        "scene_graph": _scene_graph(zones),
        "limitations": [
            "Architectural and material labels are probabilistic visual cues, not authenticated heritage classifications.",
            "Spatial attributes are image-derived proxies and do not directly measure experienced place.",
            "Temporal correspondence with EEG does not establish causation.",
        ],
    }


def export_analysis(analysis: dict[str, Any], directory: str | Path) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "place_perception.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    samples = analysis.get("samples", [])
    if samples:
        scalar_keys = [k for k, v in samples[0].items() if not isinstance(v, (dict, list))]
        with (out / "place_perception_timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_keys)
            writer.writeheader()
            for row in samples:
                writer.writerow({k: row.get(k) for k in scalar_keys})
    (out / "route_zones.json").write_text(json.dumps(analysis.get("zones", []), indent=2), encoding="utf-8")
    (out / "scene_graph.json").write_text(json.dumps(analysis.get("scene_graph", {}), indent=2), encoding="utf-8")
    return out
