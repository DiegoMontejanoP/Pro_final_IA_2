# Datasets recomendados para probar el sistema

Estos datasets sirven para evaluar documentos historicos degradados, texto
manuscrito, sellos, segmentacion, binarizacion, estilos de tinta y deteccion de
patrones.

## Recomendacion rapida

Para una prueba balanceada del proyecto:

1. **DIBCO / H-DIBCO**: mejor para reconstruccion, segmentacion texto/fondo y
   degradacion real.
2. **DIVA-HisDB**: mejor para manuscritos medievales con anotaciones de layout,
   lineas de texto y binarizacion.
3. **DKDS**: mejor si quieres probar sellos rojos sobre documentos antiguos.
4. **ICDAR 2019 cBAD / READ-BAD**: mejor para paginas historicas reales a gran
   escala.
5. **DIDA**: mejor para reconocimiento de patrones en digitos manuscritos
   historicos.

## Datasets

### 1. DIBCO / H-DIBCO

Uso ideal:

- Reconstruccion de documentos degradados.
- Segmentacion texto/fondo.
- Pruebas de ruido, manchas, tinta corrida, contraste bajo y papel envejecido.

Notas:

- DIBCO es un benchmark clasico de binarizacion de documentos.
- H-DIBCO se enfoca mas en documentos historicos manuscritos.
- Suele incluir imagen original y ground truth binario.

Enlace de referencia:

```text
https://tc11.cvc.uab.es/datasets/
```

### 2. DIVA-HisDB

Uso ideal:

- Manuscritos medievales.
- Segmentacion de lineas.
- Binarizacion.
- Identificacion de escritor/calificador de estilo.

Datos:

- 150 paginas.
- 50 paginas por manuscrito.
- Anotaciones precisas para tareas de Document Image Analysis.

Enlace:

```text
https://www.unifr.ch/inf/diva/en/research/software-data/diva-hisdb.html
```

### 3. DKDS

Uso ideal:

- Documentos japoneses premodernos degradados.
- Deteccion de sellos.
- Separacion entre caracteres y sellos.
- Binarizacion con sellos superpuestos.

Enlace:

```text
https://ruiyangju.github.io/DKDS/
```

### 4. ICDAR 2019 cBAD / READ

Uso ideal:

- Pruebas a escala con documentos historicos reales.
- Deteccion de lineas base.
- Segmentacion de escritura manuscrita en archivos historicos.

Datos:

- 3021 imagenes de paginas.
- Documentos recolectados de siete archivos europeos.
- Incluye anotaciones PAGE XML.
- Descarga grande: aproximadamente 4.7 GB.

Enlace:

```text
https://zenodo.org/records/2567398
```

Version en Hugging Face:

```text
https://huggingface.co/datasets/SSamDav/icdar-2019-competition-cbad
```

### 5. DIDA

Uso ideal:

- Reconocimiento de patrones.
- Clasificacion de caracteres/digitos.
- Agrupamiento de estilos de escritura historica.

Datos:

- Digitos historicos manuscritos de documentos suecos.
- Documentos entre 1800 y 1940.
- Incluye degradacion por papel, tinta y distorsiones historicas.

Enlace:

```text
https://www.kaggle.com/datasets/ayavariabdi/didadataset
```

### 6. PHIBD 2012

Uso ideal:

- Manuscritos persas historicos.
- Binarizacion.
- Segmentacion texto/fondo.

Datos:

- 15 imagenes de manuscritos historicos.
- Dataset pequeno, util para pruebas rapidas.

Enlace:

```text
https://tc11.cvc.uab.es/datasets/PHIBD%202012_1
```

## Como probarlos con este proyecto

1. Descarga uno de los datasets.
2. Extrae las imagenes en una carpeta, por ejemplo:

```text
datasets/dibco/
```

3. Ejecuta el script de prueba en lote:

```bash
.\.venv\Scripts\python.exe tools\batch_test_dataset.py datasets\dibco outputs\dibco
```

El script generara resultados derivados en `outputs/dibco` sin modificar las
imagenes originales.

## Recomendacion para la presentacion

Para mostrar las cinco tecnicas de forma clara:

- Usa **DIBCO/H-DIBCO** para reconstruccion y segmentacion.
- Usa **DKDS** para demostrar deteccion de sellos.
- Usa **DIDA** para reconocimiento de patrones.
- Usa **DIVA-HisDB** para manuscritos visualmente atractivos y segmentacion.

Si necesitas una sola fuente para empezar, usa **DIBCO/H-DIBCO** porque se
alinea mejor con documentos degradados y permite ver mejoras visuales rapido.
