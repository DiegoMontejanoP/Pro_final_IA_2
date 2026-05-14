from __future__ import annotations

import json
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


@dataclass(frozen=True)
class DatasetSource:
    name: str
    manifest_url: str
    description: str


ECODICES_SOURCES = [
    DatasetSource(
        name="e-codices_csg-0018",
        manifest_url="https://www.e-codices.unifr.ch/metadata/iiif/csg-0018/manifest.json",
        description="Manuscrito historico de e-codices, util para texto antiguo y degradacion.",
    ),
    DatasetSource(
        name="e-codices_csg-0863",
        manifest_url="https://www.e-codices.unifr.ch/metadata/iiif/csg-0863/manifest.json",
        description="Manuscrito historico adicional para variar caligrafia y textura del papel.",
    ),
]


def download_ecodices_samples(output_dir: Path, limit: int = 5) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for source in ECODICES_SOURCES:
        source_dir = output_dir / source.name
        source_dir.mkdir(parents=True, exist_ok=True)
        for image_url in _iiif_image_urls(source.manifest_url):
            if len(downloaded) >= limit:
                return downloaded
            file_path = source_dir / f"sample_{len(downloaded) + 1:03d}.jpg"
            _download_file(image_url, file_path)
            downloaded.append(file_path)

    return downloaded


def _iiif_image_urls(manifest_url: str) -> Iterable[str]:
    manifest = _read_json(manifest_url)

    if "items" in manifest:
        yield from _iiif_v3_urls(manifest)
        return

    sequences = manifest.get("sequences", [])
    for sequence in sequences:
        for canvas in sequence.get("canvases", []):
            for image in canvas.get("images", []):
                resource = image.get("resource", {})
                url = _image_url_from_resource(resource)
                if url:
                    yield url


def _iiif_v3_urls(manifest: dict) -> Iterable[str]:
    for canvas in manifest.get("items", []):
        for page in canvas.get("items", []):
            for annotation in page.get("items", []):
                body = annotation.get("body", {})
                url = _image_url_from_resource(body)
                if url:
                    yield url


def _image_url_from_resource(resource: dict) -> str | None:
    service = resource.get("service")
    if isinstance(service, list) and service:
        service = service[0]

    service_id = None
    if isinstance(service, dict):
        service_id = service.get("@id") or service.get("id")

    if service_id:
        return f"{service_id.rstrip('/')}/full/1200,/0/default.jpg"

    image_id = resource.get("@id") or resource.get("id")
    if image_id and _looks_like_image_url(image_id):
        return image_id

    return None


def _read_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "historical-document-ai/1.0"})
    with _urlopen_with_ssl_fallback(request, timeout=40) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_file(url: str, file_path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "historical-document-ai/1.0"})
    with _urlopen_with_ssl_fallback(request, timeout=60) as response:
        file_path.write_bytes(response.read())


def _urlopen_with_ssl_fallback(request: urllib.request.Request, timeout: int):
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError):
            context = ssl._create_unverified_context()
            return urllib.request.urlopen(request, timeout=timeout, context=context)
        raise


def _looks_like_image_url(url: str) -> bool:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
