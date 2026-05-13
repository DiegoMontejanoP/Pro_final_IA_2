from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, X, Button, Frame, Label, StringVar, Tk, filedialog, messagebox

from PIL import Image, ImageTk

import historical_document_ai as hdoc


class HistoricalDocumentApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Sistema Inteligente de Restauracion y Analisis Forense")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)

        self.base_path: Path | None = None
        self.base_image = None
        self.current_result: hdoc.TechniqueResult | None = None
        self.base_photo = None
        self.result_photo = None

        self.status = StringVar(value="Carga una imagen historica para iniciar.")
        self.metrics = StringVar(value="El archivo base nunca se modifica.")

        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = Frame(self.root, padx=10, pady=8)
        toolbar.pack(fill=X)

        Button(toolbar, text="Cargar imagen", command=self.load_image, width=16).pack(side=LEFT, padx=4)
        Button(toolbar, text="Reconstruccion", command=self.run_reconstruction, width=16).pack(side=LEFT, padx=4)
        Button(toolbar, text="Segmentacion", command=self.run_segmentation, width=16).pack(side=LEFT, padx=4)
        Button(toolbar, text="Agrupamiento", command=self.run_clustering, width=16).pack(side=LEFT, padx=4)
        Button(toolbar, text="Patrones", command=self.run_patterns, width=14).pack(side=LEFT, padx=4)
        Button(toolbar, text="Superresolucion", command=self.run_super_resolution, width=16).pack(side=LEFT, padx=4)
        Button(toolbar, text="Flujo completo", command=self.run_full_pipeline, width=14).pack(side=LEFT, padx=4)
        Button(toolbar, text="Guardar resultado", command=self.save_result, width=16).pack(side=RIGHT, padx=4)

        content = Frame(self.root, padx=12, pady=8)
        content.pack(fill=BOTH, expand=True)

        left_panel = Frame(content)
        left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 8))
        right_panel = Frame(content)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=(8, 0))

        Label(left_panel, text="Documento base", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        self.base_label = Label(left_panel, bg="#eeeeee", relief="groove")
        self.base_label.pack(fill=BOTH, expand=True)

        Label(right_panel, text="Resultado de la tecnica", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        self.result_label = Label(right_panel, bg="#eeeeee", relief="groove")
        self.result_label.pack(fill=BOTH, expand=True)

        footer = Frame(self.root, padx=12, pady=8)
        footer.pack(fill=X)
        Label(footer, textvariable=self.status, anchor="w", font=("Segoe UI", 10, "bold")).pack(fill=X)
        Label(footer, textvariable=self.metrics, anchor="w", justify=LEFT, wraplength=1120).pack(fill=X)

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Selecciona un documento historico",
            filetypes=[
                ("Imagenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.base_path = Path(path)
            self.base_image = hdoc.load_image(self.base_path)
            self.current_result = None
            self._show_base()
            self.result_label.configure(image="", text="")
            self.status.set(f"Imagen cargada: {self.base_path.name}")
            self.metrics.set("Listo. El procesamiento se hara sobre copias en memoria.")
        except Exception as exc:
            messagebox.showerror("Error al cargar", str(exc))

    def run_reconstruction(self) -> None:
        self._run_technique(hdoc.reconstruction)

    def run_segmentation(self) -> None:
        self._run_technique(hdoc.segmentation)

    def run_clustering(self) -> None:
        self._run_technique(hdoc.clustering)

    def run_patterns(self) -> None:
        self._run_technique(hdoc.pattern_recognition)

    def run_super_resolution(self) -> None:
        self._run_technique(hdoc.residual_super_resolution)

    def run_full_pipeline(self) -> None:
        self._run_technique(hdoc.full_pipeline)

    def save_result(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("Sin resultado", "Primero aplica una tecnica.")
            return

        default_name = f"resultado_forense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = filedialog.asksaveasfilename(
            title="Guardar resultado",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            import cv2

            cv2.imwrite(path, self.current_result.image)
            self.status.set(f"Resultado guardado: {path}")
        except Exception as exc:
            messagebox.showerror("Error al guardar", str(exc))

    def _run_technique(self, technique) -> None:
        if self.base_image is None:
            messagebox.showinfo("Falta imagen", "Carga primero una imagen base.")
            return
        try:
            result = technique(self.base_image.copy())
            self.current_result = result
            self._show_result(result)
            self.status.set(result.title)
            metric_text = " | ".join(f"{key}: {value}" for key, value in result.metrics.items())
            self.metrics.set(f"{result.description}\n{metric_text}")
        except Exception as exc:
            messagebox.showerror("Error de procesamiento", str(exc))

    def _show_base(self) -> None:
        preview = hdoc.resize_for_display(self.base_image)
        self.base_photo = self._to_photo(preview)
        self.base_label.configure(image=self.base_photo, text="")

    def _show_result(self, result: hdoc.TechniqueResult) -> None:
        preview = hdoc.resize_for_display(result.image)
        self.result_photo = self._to_photo(preview)
        self.result_label.configure(image=self.result_photo, text="")

    @staticmethod
    def _to_photo(image) -> ImageTk.PhotoImage:
        return ImageTk.PhotoImage(Image.fromarray(hdoc.bgr_to_rgb(image)))


def main() -> None:
    root = Tk()
    app = HistoricalDocumentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
