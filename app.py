from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, X, Button, Canvas, Frame, Label, Scrollbar, StringVar, Tk, filedialog, messagebox
import webbrowser

from PIL import Image, ImageTk

import dataset_downloader
import historical_document_ai as hdoc


class ZoomableImageView(Frame):
    def __init__(self, parent, background: str = "#eeeeee") -> None:
        super().__init__(parent)
        self.image = None
        self.photo = None
        self.zoom = 1.0
        self.canvas = Canvas(self, bg=background, highlightthickness=0)
        self.x_scroll = Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.y_scroll = Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.x_scroll.set, yscrollcommand=self.y_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.y_scroll.grid(row=0, column=1, sticky="ns")
        self.x_scroll.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.canvas.bind("<Double-Button-1>", self.reset_zoom)

    def set_image(self, image) -> None:
        self.image = image.copy()
        self.zoom = 1.0
        self._render()

    def clear(self) -> None:
        self.image = None
        self.photo = None
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, 0, 0))

    def reset_zoom(self, _event=None) -> None:
        if self.image is not None:
            self.zoom = 1.0
            self._render()

    def _on_mouse_wheel(self, event) -> None:
        if self.image is None:
            return
        if getattr(event, "num", None) == 5 or event.delta < 0:
            factor = 0.9
        else:
            factor = 1.1
        self.zoom = min(5.0, max(0.2, self.zoom * factor))
        self._render()

    def _render(self) -> None:
        rgb = hdoc.bgr_to_rgb(self.image)
        pil_image = Image.fromarray(rgb)
        w, h = pil_image.size
        display_size = (max(1, int(w * self.zoom)), max(1, int(h * self.zoom)))
        resized = pil_image.resize(display_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, display_size[0], display_size[1]))


class HistoricalDocumentApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Sistema Inteligente de Restauracion y Analisis Forense")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)

        self.base_path: Path | None = None
        self.base_image = None
        self.current_result: hdoc.TechniqueResult | None = None

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
        Button(toolbar, text="Descargar muestras", command=self.download_samples, width=18).pack(side=LEFT, padx=4)
        Button(toolbar, text="Enlaces datasets", command=self.open_dataset_links, width=16).pack(side=LEFT, padx=4)
        Button(toolbar, text="Guardar resultado", command=self.save_result, width=16).pack(side=RIGHT, padx=4)

        content = Frame(self.root, padx=12, pady=8)
        content.pack(fill=BOTH, expand=True)

        left_panel = Frame(content)
        left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 8))
        right_panel = Frame(content)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=(8, 0))

        Label(left_panel, text="Documento base", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        self.base_view = ZoomableImageView(left_panel)
        self.base_view.pack(fill=BOTH, expand=True)

        Label(right_panel, text="Resultado de la tecnica", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        self.result_view = ZoomableImageView(right_panel)
        self.result_view.pack(fill=BOTH, expand=True)

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
            self.result_view.clear()
            self.status.set(f"Imagen cargada: {self.base_path.name}")
            self.metrics.set("Listo. Usa la rueda del mouse para zoom y doble clic para reiniciar.")
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

    def download_samples(self) -> None:
        try:
            output_dir = Path("datasets") / "e_codices_samples"
            self.status.set("Descargando muestras publicas de e-codices...")
            self.root.update_idletasks()
            files = dataset_downloader.download_ecodices_samples(output_dir, limit=5)
            if not files:
                messagebox.showwarning("Sin descargas", "No se encontraron imagenes en los manifiestos.")
                return
            self.status.set(f"Se descargaron {len(files)} imagenes en {output_dir}")
            self.metrics.set(
                "Puedes cargarlas con 'Cargar imagen' o procesarlas en lote con "
                f"tools\\batch_test_dataset.py {output_dir} outputs\\e_codices"
            )
        except Exception as exc:
            messagebox.showerror(
                "Error al descargar",
                "No se pudieron descargar las muestras. Revisa tu conexion o usa los enlaces manuales.\n\n"
                f"Detalle: {exc}",
            )

    def open_dataset_links(self) -> None:
        for url in [
            "https://tc11.cvc.uab.es/datasets/",
            "https://www.unifr.ch/inf/diva/en/research/software-data/diva-hisdb.html",
            "https://ruiyangju.github.io/DKDS/",
            "https://zenodo.org/records/2567398",
        ]:
            webbrowser.open_new_tab(url)

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
        self.base_view.set_image(self.base_image)

    def _show_result(self, result: hdoc.TechniqueResult) -> None:
        self.result_view.set_image(result.image)


def main() -> None:
    root = Tk()
    app = HistoricalDocumentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
