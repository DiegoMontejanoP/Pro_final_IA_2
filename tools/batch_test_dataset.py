from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import historical_document_ai as hdoc


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


TECHNIQUES = {
    "reconstruction": hdoc.reconstruction,
    "segmentation": hdoc.segmentation,
    "clustering": hdoc.clustering,
    "patterns": hdoc.pattern_recognition,
    "super_resolution": hdoc.residual_super_resolution,
}


def iter_images(input_dir: Path):
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def safe_name(path: Path) -> str:
    return "_".join(path.with_suffix("").parts[-4:])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta las tecnicas forenses sobre una carpeta de imagenes."
    )
    parser.add_argument("input_dir", type=Path, help="Carpeta con imagenes del dataset.")
    parser.add_argument("output_dir", type=Path, help="Carpeta donde guardar resultados.")
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Cantidad maxima de imagenes a procesar. Usa 0 para procesar todas.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"No existe la carpeta de entrada: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(iter_images(args.input_dir))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        raise SystemExit("No se encontraron imagenes compatibles.")

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path}")
        image = hdoc.load_image(image_path)
        document_dir = args.output_dir / safe_name(image_path)
        document_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(document_dir / "00_original_copy.png"), image)
        for technique_name, technique in TECHNIQUES.items():
            try:
                result = technique(image.copy())
                cv2.imwrite(str(document_dir / f"{technique_name}.png"), result.image)
            except Exception as exc:
                (document_dir / f"{technique_name}_error.txt").write_text(
                    str(exc), encoding="utf-8"
                )

    print(f"Listo. Resultados en: {args.output_dir}")


if __name__ == "__main__":
    main()
