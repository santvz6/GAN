Título
GANs para síntesis de cuerpo humano en 3 modalidades: tabular (joints), imagen y 3D

Objetivo del proyecto
Entrenar modelos generativos (GAN) que produzcan muestras plausibles de cuerpo humano en tres representaciones:
A) Tabular: coordenadas 2D/3D de articulaciones (joints/keypoints).
B) Imagen: imágenes “human-like” (p. ej., render de esqueleto, heatmaps o silhouettes).
C) 3D: representación 3D (nube de puntos / joints 3D / mesh simplificado).

Cada modalidad debe incluir:

Dataset grande (o derivado del tabular si no hay suficiente imagen/3D).

Entrenamiento de un generador y un discriminador adecuados.

Evaluación cuantitativa con métricas estándar y un análisis cualitativo (muestras).

Dataset recomendado y por qué
Opción principal: AMASS
AMASS unifica múltiples datasets de motion capture en un marco común, y se distribuye como parámetros SMPL/SMPL+H y secuencias. Permite obtener joints 3D consistentes, y también generar datos derivados (imágenes y 3D).

Extracción de joints desde AMASS
Una práctica habitual es usar una “SMPL layer” para convertir parámetros SMPL a joints y/o vértices; repositorios que preparan datasets derivados desde AMASS (ej. HumanML3D) explican este flujo (extraer joints con SMPL, normalizar ejes, recortar secuencias, etc.).

Nota sobre TN(T)15 / NOMO3D
NOMO3D existe como dataset de escaneos 3D para antropometría, pero es mucho más pequeño que AMASS y no está pensado como gran corpus generativo.
(Para un trabajo de GAN con muchas muestras, AMASS suele ser más viable.)

Qué vamos a generar exactamente (definiciones operativas)
Para evitar ambigüedad, fijamos “targets” concretos por modalidad:

3.1 Tabular (joints)
Formato recomendado por frame:

J = 22 o 24 joints (según SMPL/SMPL-X; elegir uno y mantenerlo fijo)

3D: (J, 3). Opcional 2D: proyectar después.

Normalización: centrar pelvis en (0,0,0), alinear el eje vertical, escalar por altura aproximada o longitud promedio de huesos.

Salida del generador:

Un frame (pose estática) o una secuencia corta T frames.
Sugerencia: empezar por pose estática (simplifica mucho).

3.2 Imagen
Dos alternativas válidas; elegid una y mantenedla:
A) “Skeleton render”: renderizar el esqueleto como líneas/puntos en un canvas (p. ej. 256×256). Se deriva del tabular y por tanto es “dataset grande”.
B) Heatmaps por joint (multi-canal), luego convertir a RGB para la GAN (menos estándar).

Recomendación: A) skeleton render. Es fácil, reproducible y está alineado con “si no hay dataset grande de imágenes, derivarlo del tabular”.

3.3 3D
Tres opciones, de más fácil a más compleja:
A) Joints 3D como “point cloud” pequeña (J puntos). (Más fácil; similar a tabular pero tratado como set).
B) Submuestreo de vértices SMPL a N puntos (point cloud de superficie). (Más realista; requiere SMPL layer).
C) Mesh completo (más pesado; no recomendado para un primer proyecto).

Recomendación: B si el tiempo da, si no A. Para point clouds se evalúa con Chamfer/EMD/F-score típicos; para joints, además plausibilidad cinemática.

Modelos (arquitecturas mínimas)
4.1 Tabular GAN (MLP-GAN)

Generator G(z): MLP que mapea z∈R^d → vector joints (J*3).

Discriminator D(x): MLP que clasifica real/fake.

Loss: WGAN-GP o hinge loss (más estable que GAN vanilla).

4.2 Image GAN (CNN-GAN)

DCGAN / ResNet GAN (según nivel).

Entrada: ruido z; salida: imagen 256×256 (o 128×128).

Si usáis skeleton render, las imágenes son binarias o casi binarias; considerad:

salida con tanh y luego umbral para visualizar, pero entrenar con valores continuos.

4.3 3D GAN (Point Cloud GAN)

Generator: MLP → (N,3) con reshape, o FoldingNet-style; simplificado.

Discriminator: PointNet-like (MLP por punto + maxpool global) para clasificar real/fake.
Esta familia (PointNet para sets) es estándar para tratar nubes de puntos. En surveys de pose/point-cloud se discuten pipelines basados en PointNet.

Métricas de evaluación (mínimo viable + recomendadas)
La evaluación “de plausibilidad” en generativo no se limita a error contra GT (porque no hay correspondencia 1:1), así que proponemos métricas de distribución + métricas biomecánicas.

5.1 Tabular (joints)
Distribución:

MMD (Maximum Mean Discrepancy) entre distribución real y generada (sobre vectores joints normalizados).

k-NN two-sample test (opcional).

Plausibilidad geométrica/cinemática:

Bone Length Consistency: error relativo de longitudes de huesos respecto a estadísticos del dataset (media/desviación).

Joint Angle Limits: penalizar ángulos fuera de rangos (si definís cinemática).

KCS (Kinematic Chain Space) / medidas de consistencia de cadena cinemática (si queréis algo “de paper”).

Métricas estándar en pose (para referenciar en memoria del trabajo):
MPJPE y PA-MPJPE son métricas estándar en evaluación de pose 3D (aunque son más naturales cuando hay correspondencia con GT).

Si queréis citar “métricas de plausibilidad física” más modernas:
Hay trabajos que proponen métricas específicas de plausibilidad/estabilidad física para poses 3D.

5.2 Imagen

FID (Fréchet Inception Distance) entre imágenes reales (render del dataset) y generadas.

Precision/Recall para GANs (opcional, si tenéis implementaciones).

5.3 3D (point clouds)

Chamfer Distance (CD) entre sets reales y generados (por matching aproximado).

Earth Mover’s Distance (EMD) si N es pequeño y el cómputo lo permite.

F-score a umbral τ (para proximidad entre nubes).

Estructura del repositorio
/README.md
/data/
/raw/ (AMASS u otros; NO versionar en git)
/processed/
tabular_joints.npz
images_skeleton/
pointclouds/
/src/
/data/
download_instructions.md
preprocess_amass_to_joints.py
render_skeleton_images.py
build_pointclouds.py
/models/
tabular_gan.py
image_gan.py
pointcloud_gan.py
/eval/
eval_tabular.py
eval_image_fid.py
eval_3d_cd_emd.py
/utils/
metrics.py
viz.py
/notebooks/ (opcionales)
/reports/
figures/
results.json
final_report.md

Pasos globales (checklist)
Paso 0. Entorno

Python 3.10+

PyTorch (o TF, pero que sea consistente)

Librerías: numpy, scipy, matplotlib, tqdm, opencv (para renders), torchmetrics (si aplica), etc.

Si usáis SMPL/SMPL-X: instalar dependencia y colocar modelos (según licencia).

Paso 1. Dataset

Descargar AMASS (aceptar licencias) y organizar en /data/raw/amass/.

Documentar exactamente qué subconjuntos usáis (nombres y tamaño).

Paso 2. Preprocesado común

Convertir parámetros SMPL → joints 3D (Jx3) por frame.

Normalizar poses (root-centric, escala, orientación).

Guardar en /data/processed/tabular_joints.npz.

Paso 3. Derivados

Renderizar skeleton images desde joints → /data/processed/images_skeleton/.

Crear point clouds (N puntos) desde vértices SMPL o desde joints → /data/processed/pointclouds/.

Paso 4. Entrenar 3 GANs

Tabular GAN (MLP-GAN)

Image GAN (DCGAN/ResNetGAN)

3D GAN (PointNet discriminator + generator)

Paso 5. Evaluación

Tabular: MMD + plausibilidad (huesos/ángulos) + visualizaciones

Imagen: FID + grid de samples

3D: Chamfer/EMD/F-score + visualización 3D (matplotlib o open3d)

Paso 6. Entrega

README + scripts reproducibles

Resultados (json/csv) + figuras

Conclusiones: qué funciona, qué no, y por qué.

División en 4 participantes (pasos exactos por persona)

Persona 1 — “Data Lead + Tabular GAN”
Objetivo: dejar listo AMASS→joints y entrenar Tabular GAN con evaluación de plausibilidad.

Descargar AMASS y documentar licencias + subconjuntos usados.

Implementar preprocess_amass_to_joints.py:

Cargar secuencias

Extraer joints 3D con SMPL layer

Normalizar (pelvis al origen, orientación, escala)

Guardar npz/hdf5 con splits train/val/test

Implementar tabular_gan.py:

G y D MLP

WGAN-GP (o hinge)

logging de pérdidas + samples

Implementar eval_tabular.py:

MMD real vs fake

Bone length statistics (mean/std) y “bone length error” en generados

Visualización simple (esqueleto 3D por matplotlib)

Entregables de Persona 1:

/src/data/preprocess_amass_to_joints.py

/src/models/tabular_gan.py

/src/eval/eval_tabular.py

/reports/figures/tabular_samples.png + results_tabular.json

Persona 2 — “Image Lead (dataset derivado + CNN GAN + FID)”
Objetivo: construir dataset de imágenes a partir de joints y entrenar GAN de imágenes.

Implementar render_skeleton_images.py:

Entrada: joints normalizados

Salida: PNG 128×128 o 256×256 con skeleton dibujado (líneas + puntos)

Guardar train/val/test en carpetas

Implementar image_gan.py (DCGAN/ResNet GAN):

Entrenamiento estable

Guardar checkpoints

Implementar eval_image_fid.py:

Calcular FID entre reales (renders) y generadas

Generar grids de muestras

Entregables de Persona 2:

/src/data/render_skeleton_images.py

/src/models/image_gan.py

/src/eval/eval_image_fid.py

/reports/figures/image_samples.png + results_image.json

Persona 3 — “3D Lead (point clouds + GAN + Chamfer/EMD)”
Objetivo: construir representación 3D y entrenar GAN 3D.

Elegir representación:

Opción recomendada: samplear N puntos de superficie SMPL (si SMPL layer disponible)

Alternativa: usar joints como point cloud de tamaño J

Implementar build_pointclouds.py:

Generar point clouds normalizadas (centradas, escala consistente)

Guardar en formato npz

Implementar pointcloud_gan.py:

Discriminador tipo PointNet (per-point MLP + maxpool)

Generador MLP que produce Nx3

Implementar eval_3d_cd_emd.py:

Chamfer Distance (mínimo)

EMD (opcional) y F-score (opcional)

Visualización de nubes (antes/después)

Entregables de Persona 3:

/src/data/build_pointclouds.py

/src/models/pointcloud_gan.py

/src/eval/eval_3d_cd_emd.py

/reports/figures/pc_samples.png + results_3d.json

Persona 4 — “Integración + Reproducibilidad + Reporte”
Objetivo: que todo sea ejecutable, comparable y presentable.

Definir estándar de splits, seeds y rutas (config.py o YAML).

Crear scripts de ejecución:

train_tabular.sh / train_image.sh / train_3d.sh

eval_all.sh que genere /reports/results.json

Unificar logging (tensorboard o CSV).

Redactar final_report.md:

Motivación y related work corto

Descripción de datos y preprocesado

Arquitecturas y pérdidas

Resultados cuantitativos y cualitativos

Ablaciones simples (por ejemplo: WGAN-GP vs GAN vanilla en tabular)

Revisar que README refleje exactamente cómo reproducir.

Entregables de Persona 4:

/src/configs/*.yaml + scripts de run

/reports/final_report.md + results.json consolidado

Checklist de reproducibilidad

Cronograma sugerido (compacto)
Semana 1: dataset + preprocesado tabular listo (Persona 1) y renderer de skeleton (Persona 2).
Semana 2: entrenos base tabular + imagen; preparar 3D pipeline.
Semana 3: 3D GAN + métricas; estabilización y comparación.
Semana 4: reporte, figuras, limpieza del repo y reproducibilidad.

Preguntas que necesito que confirmes (solo para cerrar ambigüedades del enunciado)

¿Vuestro profesor exige explícitamente que sea “GAN” en las 3 partes, o acepta “generativo” en general (VAE/diffusion) mientras haya generación?

¿La parte tabular debe generar “poses estáticas” (un frame) o “secuencias” (varios frames)?

En 3D, ¿os aceptan “joints 3D” como 3D, o tiene que ser “nube de puntos/mesh” de superficie?

Si me respondes esas 3, te ajusto el README a lo que os puntúe mejor (sin cambiar el enfoque general).







# 3D Body Generation & Mass Estimation using GANs 🏃‍♂️📊

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![SMPL](https://img.shields.io/badge/Model-SMPL--X-orange)

Este repositorio contiene la implementación de una **Red Generativa Antagónica (GAN)** diseñada para la reconstrucción de cuerpos humanos en 3D y la estimación de masa corporal a partir de datos antropométricos mínimos.

## 🎯 Objetivo del Proyecto

El reto principal consiste en generar representaciones corporales realistas (nubes de puntos de 6890 vértices) utilizando únicamente un **vector de 10 parámetros tabulares**. 

Utilizamos el dataset **AMASS** para entrenar un modelo que aprenda la distribución real de las formas humanas, permitiendo que el generador "sintetice" cuerpos que pasen el Test de Turing frente a un discriminador experto.

## 🛠️ Arquitectura Propuesta

El sistema se divide en tres bloques principales:

1.  **Generador (MLP):** Recibe parámetros latentes y devuelve el vector de masa/forma (10 puntos clave).
2.  **Pipeline Tabular-to-Body:** Mapea el vector generado a una malla SMPL de 6890 puntos.
3.  **Discriminador:** Una dos tres
