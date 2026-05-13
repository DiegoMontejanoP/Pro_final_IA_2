# Sistema Inteligente de Restauracion y Analisis Forense

Aplicacion en Python para procesar imagenes de documentos historicos danados sin
modificar el archivo base. La interfaz permite cargar una imagen y aplicar cada
tecnica con un boton independiente.

## Tecnicas integradas

- **Reconstruccion:** elimina ruido y completa grietas/manchas con denoising e
  inpainting, como aproximacion practica a un autoencoder convolucional.
- **Segmentacion:** separa texto/fondo y detecta posibles sellos o firmas usando
  mascaras adaptativas, equivalente conceptual a una arquitectura U-Net.
- **Agrupamiento:** categoriza estilos de tinta o caligrafia sin etiquetas con
  K-Means/SOM conceptual.
- **Reconocimiento de patrones:** detecta componentes tipo caracter y anomalias
  por diferencia estadistica.
- **Interpolacion/Extrapolacion:** aumenta resolucion con bicubica x2 y
  refinamiento residual inspirado en SRResNet.

## Instalacion

```bash
pip install -r requirements.txt
```

En Windows, si `python` no esta en el PATH, usa:

```bash
py -m pip install -r requirements.txt
```

### PyCharm y error `externally-managed-environment`

Si PyCharm muestra que el Python esta administrado por `uv` y no permite
instalar `pillow` u `opencv-python`, crea un entorno virtual dentro del proyecto:

```bash
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Luego en PyCharm selecciona:

```text
Settings > Project > Python Interpreter > Add Interpreter > Existing
```

y usa:

```text
.\.venv\Scripts\python.exe
```

No uses `--break-system-packages` salvo que sea estrictamente necesario; es
mejor mantener intacto el Python administrado por `uv`.

## Ejecucion

```bash
python app.py
```

O en Windows:

```bash
py app.py
```

## Uso

1. Presiona **Cargar imagen** y selecciona un documento en PNG, JPG, BMP o TIFF.
2. Usa un boton por tecnica para ver el resultado derivado.
3. Usa **Flujo completo** para ejecutar todas las etapas sobre una copia.
4. Usa **Guardar resultado** para exportar una nueva imagen procesada.

El archivo original solo se lee; no se sobrescribe ni se modifica.

## Datasets para pruebas

Consulta [DATASETS.md](DATASETS.md) para descargar datasets historicos como
DIBCO/H-DIBCO, DIVA-HisDB, DKDS, ICDAR cBAD y DIDA.

Tambien puedes ejecutar pruebas en lote con:

```bash
.\.venv\Scripts\python.exe tools\batch_test_dataset.py datasets\dibco outputs\dibco
```
