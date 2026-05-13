from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


Array = np.ndarray


@dataclass
class TechniqueResult:
    """Resultado visual y metadatos breves para mostrar en la interfaz."""

    title: str
    image: Array
    description: str
    metrics: Dict[str, float | int | str]


def load_image(path: str | Path) -> Array:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    return image


def to_gray(image: Array) -> Array:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_for_display(image: Array, max_size: Tuple[int, int] = (620, 520)) -> Array:
    h, w = image.shape[:2]
    max_w, max_h = max_size
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale == 1.0:
        return image.copy()
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: Array) -> Array:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def reconstruction(image: Array) -> TechniqueResult:
    """Reconstruccion con limpieza de ruido e inpainting de grietas/manchas."""
    gray = to_gray(image)
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 8, 8, 7, 21)
    denoised_gray = to_gray(denoised)

    dark_defects = cv2.threshold(denoised_gray, 34, 255, cv2.THRESH_BINARY_INV)[1]
    bright_defects = cv2.threshold(denoised_gray, 245, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(denoised_gray, 55, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_mask = cv2.bitwise_or(dark_defects, bright_defects)
    defect_mask = cv2.bitwise_or(defect_mask, cv2.dilate(edges, kernel, iterations=1))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)

    restored = cv2.inpaint(denoised, defect_mask, 3, cv2.INPAINT_TELEA)
    sharpen = cv2.GaussianBlur(restored, (0, 0), 1.2)
    restored = cv2.addWeighted(restored, 1.35, sharpen, -0.35, 0)

    return TechniqueResult(
        title="Reconstruccion - Autoencoder/Inpainting",
        image=restored,
        description=(
            "Elimina ruido con Non-Local Means y reconstruye zonas faltantes con "
            "inpainting guiado por una mascara de defectos."
        ),
        metrics={
            "pixeles_reconstruidos": int(np.count_nonzero(defect_mask)),
            "porcentaje_mascara": round(float(np.mean(defect_mask > 0) * 100), 2),
            "topologia": "Autoencoder convolucional simulado + inpainting",
        },
    )


def segmentation(image: Array) -> TechniqueResult:
    """Segmenta texto/fondo y resalta posibles sellos y firmas."""
    gray = to_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    text_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        12,
    )
    text_mask = cv2.morphologyEx(
        text_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_1 = cv2.inRange(hsv, (0, 35, 40), (12, 255, 255))
    red_2 = cv2.inRange(hsv, (165, 35, 40), (180, 255, 255))
    blue = cv2.inRange(hsv, (88, 30, 35), (135, 255, 255))
    stamp_mask = cv2.bitwise_or(cv2.bitwise_or(red_1, red_2), blue)
    stamp_mask = cv2.morphologyEx(
        stamp_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )

    signature_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 3))
    signature_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, signature_kernel)
    signature_mask = _filter_components(signature_mask, min_area=180, max_area=20000, wide_only=True)

    overlay = image.copy()
    overlay[text_mask > 0] = (40, 40, 40)
    overlay[stamp_mask > 0] = (30, 60, 220)
    overlay[signature_mask > 0] = (45, 170, 45)
    visual = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)

    return TechniqueResult(
        title="Segmentacion - U-Net conceptual",
        image=visual,
        description=(
            "Separa texto y fondo con umbral adaptativo; resalta sellos por color "
            "y firmas por componentes alargados."
        ),
        metrics={
            "pixeles_texto": int(np.count_nonzero(text_mask)),
            "pixeles_sello": int(np.count_nonzero(stamp_mask)),
            "pixeles_firma": int(np.count_nonzero(signature_mask)),
            "topologia": "Segmentador tipo U-Net / encoder-decoder",
        },
    )


def clustering(image: Array, k: int = 4) -> TechniqueResult:
    """Agrupa estilos de tinta/caligrafia sin etiquetas."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = to_gray(image)
    text_mask = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0),
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    ys, xs = np.where(text_mask > 0)
    if len(xs) < 20:
        raise ValueError("No se detecto suficiente tinta/texto para agrupar.")

    features = np.column_stack(
        [
            lab[ys, xs, 0] / 255.0,
            lab[ys, xs, 1] / 255.0,
            lab[ys, xs, 2] / 255.0,
            xs / max(image.shape[1], 1),
            ys / max(image.shape[0], 1),
        ]
    ).astype(np.float32)

    labels = _kmeans(features, k)
    palette = np.array(
        [
            (230, 90, 40),
            (40, 170, 70),
            (40, 120, 230),
            (200, 70, 180),
            (30, 190, 190),
            (230, 170, 40),
        ],
        dtype=np.uint8,
    )
    visual = image.copy()
    for cluster_id in range(k):
        mask = labels == cluster_id
        visual[ys[mask], xs[mask]] = palette[cluster_id % len(palette)]
    visual = cv2.addWeighted(image, 0.35, visual, 0.65, 0)

    counts = np.bincount(labels, minlength=k)
    metrics: Dict[str, float | int | str] = {
        "clusters": k,
        "pixeles_agrupados": int(len(labels)),
        "topologia": "SOM/K-Means no supervisado",
    }
    for index, count in enumerate(counts):
        metrics[f"cluster_{index + 1}_pct"] = round(float(count / len(labels) * 100), 2)

    return TechniqueResult(
        title="Agrupamiento - Estilos de tinta",
        image=visual,
        description=(
            "Agrupa pixeles de tinta usando color LAB y posicion espacial para "
            "aproximar estilos de caligrafia o tipos de tinta sin etiquetas."
        ),
        metrics=metrics,
    )


def pattern_recognition(image: Array) -> TechniqueResult:
    """Detecta caracteres/componentes y marca posibles anomalias."""
    gray = to_gray(image)
    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if 8 <= area <= 2500 and h >= 5 and w >= 2:
            aspect = w / max(h, 1)
            components.append((x, y, w, h, area, aspect))

    if not components:
        raise ValueError("No se encontraron componentes tipo caracter.")

    areas = np.array([c[4] for c in components], dtype=np.float32)
    aspects = np.array([c[5] for c in components], dtype=np.float32)
    area_z = _zscore(areas)
    aspect_z = _zscore(aspects)

    visual = image.copy()
    anomalies = 0
    for idx, (x, y, w, h, area, aspect) in enumerate(components):
        score = abs(area_z[idx]) + abs(aspect_z[idx])
        if score > 3.2:
            color = (35, 35, 230)
            anomalies += 1
        else:
            color = (35, 180, 65)
        cv2.rectangle(visual, (x, y), (x + w, y + h), color, 1)

    return TechniqueResult(
        title="Reconocimiento de patrones - CNN conceptual",
        image=visual,
        description=(
            "Extrae componentes conectados como candidatos a caracteres y detecta "
            "anomalias por tamano/proporcion frente al patron dominante."
        ),
        metrics={
            "caracteres_detectados": len(components),
            "anomalias": anomalies,
            "topologia": "CNN clasificadora + detector de anomalias",
        },
    )


def residual_super_resolution(image: Array, scale: int = 2) -> TechniqueResult:
    """Aumenta resolucion con interpolacion y refinamiento residual."""
    h, w = image.shape[:2]
    upscaled = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    smooth = cv2.GaussianBlur(upscaled, (0, 0), 1.1)
    residual = cv2.subtract(upscaled, smooth)
    detail = cv2.addWeighted(upscaled, 1.0, residual, 1.8, 0)

    lab = cv2.cvtColor(detail, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([enhanced_l, a, b]), cv2.COLOR_LAB2BGR)

    return TechniqueResult(
        title="Interpolacion/Extrapolacion - Red residual",
        image=enhanced,
        description=(
            "Aplica interpolacion bicubica x2 y suma un mapa residual de detalles, "
            "imitando el bloque de salto de una red residual de superresolucion."
        ),
        metrics={
            "escala": f"{scale}x",
            "resolucion_original": f"{w}x{h}",
            "resolucion_resultado": f"{w * scale}x{h * scale}",
            "topologia": "SRResNet / bloques residuales",
        },
    )


def full_pipeline(image: Array) -> TechniqueResult:
    restored = reconstruction(image).image
    segmented = segmentation(restored).image
    clustered = clustering(restored).image
    pattern = pattern_recognition(restored).image
    sr = residual_super_resolution(restored).image

    target_h = max(restored.shape[0], segmented.shape[0], clustered.shape[0], pattern.shape[0])
    thumbnails = [
        _fit_to_height(restored, target_h),
        _fit_to_height(segmented, target_h),
        _fit_to_height(clustered, target_h),
        _fit_to_height(pattern, target_h),
        _fit_to_height(cv2.resize(sr, (restored.shape[1], restored.shape[0])), target_h),
    ]
    visual = cv2.hconcat(thumbnails)
    return TechniqueResult(
        title="Flujo completo",
        image=visual,
        description="Ejecuta las cinco tecnicas sobre una copia derivada del documento base.",
        metrics={"etapas": 5, "archivo_base": "sin modificar"},
    )


def _filter_components(mask: Array, min_area: int, max_area: int, wide_only: bool = False) -> Array:
    output = np.zeros_like(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if min_area <= area <= max_area and (not wide_only or w > h * 2.5):
            output[labels == label] = 255
    return output


def _kmeans(features: Array, k: int, iterations: int = 35) -> Array:
    sample_count = len(features)
    k = min(k, sample_count)
    quantiles = np.linspace(0, sample_count - 1, k).astype(int)
    order = np.argsort(features[:, 0])
    centers = features[order[quantiles]].copy()

    labels = np.zeros(sample_count, dtype=np.int32)
    for _ in range(iterations):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            members = features[labels == cluster_id]
            if len(members):
                centers[cluster_id] = members.mean(axis=0)
    return labels


def _zscore(values: Array) -> Array:
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def _fit_to_height(image: Array, target_h: int) -> Array:
    h, w = image.shape[:2]
    if h == target_h:
        return image
    new_w = max(1, int(w * target_h / max(h, 1)))
    return cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)
