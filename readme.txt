README — Proyecto GAN: Cuerpo humano (Tabular → Imagen → 3D)
0) Qué se entiende del enunciado (y cómo lo vamos a aterrizar)

Vuestra interpretación es básicamente correcta, pero hay una ambigüedad clave: si el “tabular” son puntos 3D (x,y,z) de articulaciones, entonces ya estáis muy cerca de “3D”. Para que el trabajo tenga 3 partes claramente distintas, lo más limpio es:

Tabular: generar poses/esqueletos como vector numérico (puntos/joints).

Imagen: generar imágenes 2D (render del esqueleto o silueta) con una GAN de imagen.

3D: generar nubes de puntos densas o mallas (no solo joints), y evaluar con métricas 3D.

Si no hay dataset grande de imagen/3D: se crea a partir del tabular renderizando esqueletos a imágenes y/o “engordando” joints a malla/nube de puntos.

1) Objetivo del proyecto

Construir un pipeline de generación de cuerpo humano en 3 modalidades:

Tabular (poses): entrenar un generador (MLP-based GAN) que produzca conjuntos de puntos de articulaciones con forma humana plausible.

Imagen (2D): entrenar una GAN de imágenes (CNN) para generar imágenes plausibles (esqueleto 2D o silueta).

3D (forma): entrenar una GAN 3D (point cloud o voxel/mesh) para generar geometría humana plausible.

Cada parte debe incluir:

dataset / preprocessing,

modelo,

entrenamiento reproducible,

métricas de “realismo”,

resultados (figuras + tablas) y discusión.

2) Datasets recomendados (elige 1 principal y 1 plan B)
Opción A (recomendada): AMASS / SMPL

Pros: enorme, estandarizado, permite sacar joints 3D y también malla SMPL.

Contras: licencias/descargas por partes; requiere preprocessing.

Opción B: Human3.6M / 3DPW / MPI-INF-3DHP

Pros: más directo para joints.

Contras: menos “masivo” que AMASS, accesos/restricciones.

Opción C: cualquier dataset “TN15 / nomo3D…”

Si lo usáis, estandarizadlo a “joints en 3D”.

Decisión práctica: trabajad con una representación base común:

J = 22 o 24 joints (pelvis, cadera, rodillas, etc.)

cada pose: vector J*3 (3D) o J*2 (2D)

3) Estructura del repositorio (propuesta)
project-gan-human-body/
  README.md
  requirements.txt
  configs/
    tabular.yaml
    image.yaml
    shape3d.yaml
  data/
    raw/           # no subir a git
    processed/     # no subir a git (o subir subset pequeño)
    samples/       # outputs y ejemplos
  src/
    common/
      seed.py
      io.py
      metrics_pose.py
      render_skeleton.py
    tabular/
      preprocess.py
      model.py
      train.py
      sample.py
      eval.py
    image/
      make_dataset_from_pose.py
      model.py
      train.py
      sample.py
      eval.py
    shape3d/
      make_pointcloud_from_smpl.py
      model.py
      train.py
      sample.py
      eval.py
  reports/
    figures/
    tables/
    paper.md (o latex/)
  scripts/
    download_data.md
    run_all.sh
4) Setup reproducible (todos)
Entorno

Python 3.10/3.11

PyTorch + CUDA (si hay GPU)

pip install -r requirements.txt

Reglas de oro

Fijar semilla.

Guardar configs + checkpoints.

Guardar muestras por epoch.

Guardar métricas en CSV.

5) Representación “tabular” (la base de todo)
5.1 Preprocesado estándar (obligatorio)

Para cada pose (joints 3D):

Centrar: restar pelvis/hip a todos los joints.

Normalizar escala: dividir por una medida estable (ej. longitud cadera-hombro o altura aproximada).

Alinear orientación (opcional pero recomendable): rotar para que “miren al frente” usando vector hombros/caderas.

Vector final: x ∈ R^(J*3).

Guardar:

data/processed/poses_train.npy

data/processed/poses_val.npy

meta.json con orden de joints, escala, etc.

6) Parte 1 — TABULAR (MLP-GAN para joints)
Modelo recomendado (simple y defendible)

WGAN-GP con MLP

Generador: MLP(z→J*3)

Crítico: MLP(J*3→score)

Pérdida WGAN + gradient penalty

Métricas “tienen sentido humano”

Mínimo 3 (una estadística, una geométrica, una “aprendida”):

Bone-length consistency

Definir huesos (pelvis→rodilla, etc.)

Comparar distribución de longitudes en real vs generado (MSE / KL / Wasserstein-1 por hueso)

Joint-limit violations

Estimar ángulos (o proxies) y penalizar poses imposibles (rodillas invertidas, etc.)

% de muestras inválidas

Distancia en embedding (FID-like para poses)

Entrenar un clasificador simple o usar un encoder MLP/TCN sobre poses.

Calcular Fréchet distance entre embeddings reales vs generados (análogo a FID).

Outputs exigidos

Tabla con métricas.

Grid de esqueletos generados (render 2D/3D simple).

Curvas de loss + métricas vs epoch.

7) Parte 2 — IMAGEN (CNN-GAN)
Dataset de imágenes (si no hay uno grande)

Generadlo desde tabular:

Proyectar joints 3D a 2D (cámara fija).

Renderizar skeleton 2D (líneas y puntos) en 128×128 o 256×256.

Guardar PNG + split train/val.

Script: src/image/make_dataset_from_pose.py

Modelo recomendado

DCGAN (baseline) o StyleGAN2-ADA (si queréis nota alta y tenéis GPU/tiempo).

Para trabajo universitario “limpio”: DCGAN es suficiente.

Métricas de imagen

FID (mínimo)

Precision/Recall generativo (opcional)

“Human sanity check”: un pose-estimator simple sobre el skeleton renderizado debería detectar joints razonables (si es skeleton, esto es trivial; si es silueta, más difícil).

Outputs

Muestras por epoch.

FID vs epoch.

Comparativa real vs generado.

8) Parte 3 — 3D (forma real: nube de puntos o malla)

Aquí NO vale “solo joints”, porque eso ya fue tabular.

Opción 3D-A (recomendada): Point Cloud GAN

Convertir cada cuerpo a nube de puntos (ej. 2048 puntos) desde malla SMPL (AMASS) o desde un template deformado.

Modelo:

Generador: MLP que produzca 2048×3 (o folding-based).

Discriminador: PointNet (muy defendible).

Métricas 3D estándar:

Chamfer Distance (CD) real↔gen (comparando a vecinos o usando set matching)

Earth Mover’s Distance (EMD) (si se puede; es más pesado)

MMD / Coverage / 1-NN accuracy en features de PointNet (muy típico en papers)

Opción 3D-B: Voxels (3D CNN GAN)

Más simple conceptualmente, más pesado en memoria.

Outputs

Visualización de nubes 3D (capturas).

CD/EMD/MMD en tabla.

Ablation: 1024 vs 2048 puntos (si da tiempo).

9) Plan de experimentos (mínimo viable)

Tabular: WGAN-GP, 50k–200k steps (según GPU), batch 256 si cabe.

Imagen: DCGAN 128×128, 50–100 epochs.

3D: PointGAN con 2048 puntos, 100–200 epochs.

Guardad:

3 seeds (o 2 si vais justos).

Media ± std en métricas.

10) División del trabajo (4 personas) — pasos exactos
Persona A — Data + Tabular (líder de “base”)

Objetivo: dataset joints limpio + GAN tabular + métricas geométricas.
Pasos:

Descargar dataset principal (AMASS o el que elijáis) y documentar licencias en scripts/download_data.md.

Implementar src/tabular/preprocess.py:

extraer joints,

centrar/normalizar/alinear,

guardar .npy.

Implementar src/tabular/model.py (WGAN-GP MLP) + train.py.

Implementar métricas:

bone-length consistency,

% violaciones (ángulos/proxies),

generar renders de esqueletos.

Entregar:

poses_train.npy/poses_val.npy,

checkpoints,

tabla de métricas + figuras.

Persona B — Imagen (dataset + GAN + FID)

Objetivo: generar dataset de imágenes desde poses + GAN CNN.
Pasos:

Implementar src/image/make_dataset_from_pose.py:

leer poses_train.npy,

proyectar a 2D,

render skeleton en PNG (train/val).

Implementar DCGAN (model.py, train.py, sample.py).

Implementar FID (o usar implementación estándar) en eval.py.

Entregar:

carpeta data/processed/images_* (o script que la regenere),

muestras por epoch,

FID plot y comparativa.

Persona C — 3D (point cloud / mesh GAN)

Objetivo: generar forma 3D (no joints) y evaluarla.
Pasos:

Si hay SMPL/malla: implementar src/shape3d/make_pointcloud_from_smpl.py:

samplear 2048 puntos por cuerpo.

Implementar GAN 3D:

G: MLP → (N,3),

D: PointNet → score.

Implementar métricas:

Chamfer,

(opcional) EMD,

MMD/Coverage/1-NN en features.

Entregar:

visualizaciones 3D,

tabla de métricas,

checkpoints.

Persona D — Integración + Reporte + Calidad (el que os salva la nota)

Objetivo: que todo se ejecute, esté documentado y defendible.
Pasos:

Definir el estándar de config + logging:

configs/*.yaml, seeds, paths.

Escribir scripts/run_all.sh o instrucciones exactas reproducibles.

Unificar formato de resultados en reports/tables/*.csv.

Redactar memoria:

Introducción (GANs, motivación),

Metodología (3 modalidades),

Experimentos,

Resultados,

Limitaciones y trabajo futuro.

Preparar “defensa”:

10 preguntas típicas y respuestas (sobre WGAN-GP, FID, Chamfer, etc.)

11) Checklist final de entrega

 Un comando por modalidad para entrenar y otro para evaluar.

 Métricas sólidas (no solo “se ve bien”).

 Figuras: muestras + curvas.

 Tabla resumen con 3 modalidades.

 Reproducible (seed + configs).

 Limitaciones claras (sesgos dataset, orientación, escala, colapsos, etc.)

12) Comandos ejemplo (plantilla)

(adaptad rutas y configs)

# TABULAR
python -m src.tabular.preprocess --cfg configs/tabular.yaml
python -m src.tabular.train       --cfg configs/tabular.yaml
python -m src.tabular.eval        --cfg configs/tabular.yaml

# IMAGEN
python -m src.image.make_dataset_from_pose --cfg configs/image.yaml
python -m src.image.train                 --cfg configs/image.yaml
python -m src.image.eval                  --cfg configs/image.yaml

# 3D
python -m src.shape3d.make_pointcloud_from_smpl --cfg configs/shape3d.yaml
python -m src.shape3d.train                     --cfg configs/shape3d.yaml
python -m src.shape3d.eval                      --cfg configs/shape3d.yaml
Suposiciones que he hecho (para no bloquearos)

“Tabular” = joints 3D (o 2D) en forma vectorial.

“3D” = geometría (point cloud/mesh), no solo joints.

Entrenáis GANs “clásicas” (WGAN-GP / DCGAN / PointNet-GAN).

Si alguna de estas 3 no encaja con vuestro enunciado literal, me lo dices y lo ajusto en 2 minutos (sin drama, con precisión quirúrgica).
