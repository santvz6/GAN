# CLAUDE.md — Plan de Acción: GAN para Síntesis de Cuerpo Humano

## Regla de Workflow — Git

**Cada cambio realizado debe subirse a GitHub en la rama `dev_taron`.**

```bash
# Flujo obligatorio tras cada cambio
git add <archivos modificados>
git commit -m "descripción del cambio"
git push origin dev_taron
```

- Nunca hacer push a `main` directamente.
- Si la rama no existe localmente: `git checkout -b dev_taron`
- Si ya existe: `git checkout dev_taron` antes de empezar a trabajar.

---

## Regla de Notebook — Documentación del Proyecto

**El proyecto debe tener un notebook principal `notebooks/proyecto_gan.ipynb` que documente TODO lo realizado.**

### Requisitos del notebook:
- Escrito para ser entendido por alguien **sin conocimientos de programación**.
- Cada sección debe tener una explicación en texto antes del código.
- El código debe ir acompañado de comentarios que expliquen qué hace cada parte.
- Incluir visualizaciones (gráficas, imágenes generadas, métricas) con su interpretación.
- Lenguaje claro, sin jerga técnica innecesaria. Cuando se use un término técnico, explicarlo.

### Estructura del notebook:
1. **Introducción** — qué es una GAN y por qué se usa para cuerpos humanos
2. **Dataset** — qué es AMASS, cómo se descargó, qué contiene
3. **Preprocesado** — cómo se extraen los joints y por qué se normalizan
4. **GAN Tabular** — arquitectura, entrenamiento y resultados con visualizaciones
5. **GAN Imagen** — cómo se generan las imágenes de esqueleto, entrenamiento y resultados
6. **GAN 3D** — qué es una nube de puntos, entrenamiento y resultados
7. **Métricas** — qué mide cada métrica (MMD, FID, Chamfer) explicado en lenguaje simple
8. **Conclusiones** — qué funcionó, qué no, y próximos pasos

### Regla:
Cada vez que se implemente o modifique algo en el proyecto, el notebook debe actualizarse para reflejar el cambio con su explicación correspondiente.

---

## Descripción del Proyecto

Entrenamiento de GANs para generar representaciones plausibles del cuerpo humano en tres modalidades:
1. **Tabular** — coordenadas 3D de joints (24 joints × 3 = 72 dimensiones)
2. **Imagen** — renders de esqueleto 128×128 RGB
3. **3D** — malla SMPL completa (6890 vértices)

---

## Datasets (estructura real en disco)

```
src/dataset/
├── AMASS_profesor.zip            (117 MB — copia comprimida del módulo)
├── TNT15_V1_0.zip                (405 MB — copia comprimida)
├── TNT15_documentation.pdf       (referencia técnica del dataset)
├── AMASS/
│   └── smpl_module_project/      ← MÓDULO DEL PROFESOR (con modelos SMPL)
│       ├── measure.py, visualize.py, joint_definitions.py, ...
│       └── data/smpl/
│           ├── SMPL_FEMALE.pkl   (38 MB)
│           ├── SMPL_MALE.pkl     (38 MB)
│           ├── SMPL_NEUTRAL.pkl  (38 MB)
│           └── *.json
├── TNT15_V1_0/
│   ├── Documentation/
│   │   └── TNT15_documentation.pdf
│   └── Images/
│       ├── mr/   ← usada para Image GAN (~24.920 PNGs segmentados)
│       │   ├── 00/  (PNGs segmentados + MP4)
│       │   ├── 01/  (PNGs segmentados + MP4)
│       │   ├── 02/  (PNGs segmentados + MP4)
│       │   ├── 03/  (PNGs segmentados + MP4)
│       │   └── 04/  (PNGs segmentados + MP4)
│       └── pz/   ← también existe (~26.574 PNGs segmentados, no se usa)
│           ├── 00/  01/  02/  03/  04/
└── NOMO3D/
    └── nomo-scans(repetitions-removed)/
        ├── female/              (177 OBJ — female_0000.obj … female_0176.obj)
        ├── male/                (179 OBJ — male_0000.obj … male_0178.obj)
        ├── female_meas_txt/     (177 TXT — medidas antropométricas)
        └── male_meas_txt/       (179 TXT — medidas antropométricas)
```

**IMPORTANTE:**
- Los ficheros `.npz` de AMASS (poses reales) aún **no están** en `src/dataset/AMASS/` — hay que añadirlos ahí.
- TNT15: imágenes **segmentadas** (siluetas de personas). Existen **dos subcarpetas**: `pz/` (~26.574 PNGs) y `mr/` (~24.920 PNGs), cada una con subdirectorios `00/`–`04/`. Se usa `Images/mr/`.
- NOMO3D: 356 OBJ únicos (~57k verts c/u) + medidas en .txt. Se submuestrean a 6890 pts con `trimesh`.

| Dataset | Rol | Ubicación real | Estado |
|---------|-----|----------------|--------|
| **AMASS** | Tabular — joints via SMPL forward pass | `src/dataset/AMASS/` | Módulo OK — añadir .npz |
| **TNT15** | Imagen — ~26.581 PNGs segmentados | `src/dataset/TNT15_V1_0/Images/mr/` | Disponible |
| **NOMO3D** | 3D — 356 OBJ + medidas .txt | `src/dataset/NOMO3D/nomo-scans(repetitions-removed)/` | Disponible |

---

## Pipeline General

```
src/dataset/AMASS/ (.npz a añadir)
        │
        ▼
amass_loader.py ──► preprocess_joints.py ──► joints.npz (N×72)
                                                  │
                                             Tabular GAN
                                                  │
                                        generar N cuerpos z~N(0,1)
                                                  │
                                    discriminator_filter.py
                                    (mantener score_D > 0.8)
                                                  │
                                       "cuerpos perfectos"
                                 ┌────────────────┴────────────────┐
                                 ▼                                 ▼
                       smpl_mesh_generator.py              render_skeleton.py
                       malla 6890 verts                    imagen 128×128 PNG
                       (SMPL forward pass)                 (Pillow, no opencv)
                                 │                                 │
                    src/dataset/NOMO3D/               src/dataset/TNT15_V1_0/
                    OBJ subsampled a 6890 pts         Images/mr/ (~24.9k PNGs)
                    + medidas csv/txt                 + renders generados
                                 │                                 │
                          3D Mesh GAN                        Image GAN
                    (NOMO3D + sintético)               (TNT15 + renders)
```

**Criterio de aceptación:** `score_D > 0.8` — el discriminador del Tabular GAN puntúa cada cuerpo generado entre 0 y 1; solo se conservan los que superan 0.8 (los más parecidos a un cuerpo humano real).

---

## Estructura del Proyecto

```
GAN/
├── CLAUDE.md                         (este archivo)
├── README.md                         (especificación del proyecto)
├── TODO.md                           (checklist de tareas)
├── main.py                           (punto de entrada)
├── windows_environment.yaml          (entorno conda Windows)
├── ubuntu_environment.yaml           (entorno conda Ubuntu)
├── src/
│   ├── config/
│   │   ├── paths.py                  (EXISTENTE — paths del proyecto)
│   │   └── utils.py                  (EXISTENTE — utils SMPL)
│   ├── __init__.py                   (CREAR — hace src un paquete Python)
│   ├── config/
│   │   ├── __init__.py               (CREAR)
│   │   ├── paths.py                  (EXISTENTE — paths del proyecto)
│   │   └── utils.py                  (EXISTENTE — utils SMPL)
│   ├── dataset/                      (EXISTENTE — datasets reales)
│   │   ├── AMASS/
│   │   │   └── smpl_module_project/  (MÓDULO DEL PROFESOR — no modificar)
│   │   │       ├── *.py
│   │   │       └── data/smpl/        (modelos SMPL .pkl)
│   │   ├── TNT15_V1_0/
│   │   │   ├── Documentation/
│   │   │   └── Images/mr/            (~26.581 PNGs en 5 subcarpetas 00-04)
│   │   └── NOMO3D/
│   │       └── nomo-scans(repetitions-removed)/
│   │           ├── female/           (177 OBJ)
│   │           ├── male/             (179 OBJ)
│   │           ├── female_meas_txt/  (177 TXT)
│   │           └── male_meas_txt/    (179 TXT)
│   ├── data/                         (CREAR)
│   │   ├── __init__.py               (CREAR)
│   │   ├── amass_loader.py           (carga .npz de AMASS)
│   │   ├── preprocess_joints.py      (extrae joints, normaliza → joints.npz)
│   │   ├── tn15_loader.py            (carga PNGs de Images/mr/, resize 128×128)
│   │   ├── nomo3d_loader.py          (carga OBJ gender subfolders, subsamplea a 6890 pts)
│   │   ├── render_skeleton.py        (joints → imagen 128×128 con Pillow)
│   │   ├── smpl_mesh_generator.py    (joints/betas → malla 6890 verts via smplx)
│   │   └── discriminator_filter.py   (genera N cuerpos, filtra score_D > 0.8)
│   ├── models/
│   │   ├── __init__.py               (CREAR)
│   │   ├── tabular_gan.py            (CREAR — MLP Generator + Discriminator, WGAN-GP)
│   │   ├── image_gan.py              (CREAR — DCGAN)
│   │   └── mesh_gan.py               (CREAR — MLP Gen + PointNet Disc, 6890 verts)
│   ├── training/
│   │   ├── __init__.py               (CREAR)
│   │   ├── train_tabular.py          (CREAR — bucle WGAN-GP tabular)
│   │   ├── train_image.py            (CREAR — bucle imagen: TNT15 + renders)
│   │   └── train_mesh.py             (CREAR — bucle 3D: NOMO3D + sintético)
│   └── evaluation/
│       ├── __init__.py               (CREAR)
│       ├── eval_tabular.py           (CREAR — MMD + bone length error)
│       ├── eval_image.py             (CREAR — FID con pytorch-fid)
│       └── eval_3d.py                (CREAR — Chamfer Distance + F-score)
└── internal/                         (generado por Paths.init_project())
    ├── data/
    │   ├── joints.npz                (joints normalizados de AMASS)
    │   ├── generated_joints.npz      (cuerpos filtrados por discriminador)
    │   ├── skeleton_images/          (renders 128×128 generados)
    │   └── meshes/                   (mallas 6890 verts generadas)
    ├── experiments/
    ├── logs/
    └── temp/
```

---

## Reutilización del Módulo del Profesor

**Ruta real:** `src/dataset/AMASS/smpl_module_project/`
**NO modificar** nada dentro de esta carpeta. Solo importar desde ella.

| Módulo | Qué usar | Para qué |
|--------|----------|----------|
| `joint_definitions.py` | `SMPL_IND2JOINT`, `SMPL_JOINT2IND`, `get_joint_regressor()` | Extraer 24 joints de AMASS |
| `landmark_definitions.py` | `SMPL_LANDMARK_INDICES` | Landmarks de normalización (pelvis, head top) |
| `visualize.py` | `Visualizer.create_joint_plot()` | Visualización 3D de resultados |
| `utils.py` | `convex_hull_from_3D_points()` | Circumferencias si se necesitan |
| `evaluate.py` | `evaluate_mae()` | MAE entre medidas reales y generadas |
| `data/smpl/SMPL_NEUTRAL.pkl` | modelo SMPL | Forward pass para joints y meshes |

```python
# Importación correcta (ruta actualizada)
import sys
sys.path.insert(0, 'src/dataset/AMASS/smpl_module_project')
from joint_definitions import SMPL_IND2JOINT, get_joint_regressor
from landmark_definitions import SMPL_LANDMARK_INDICES
from visualize import Visualizer

SMPL_MODEL_PATH = 'src/dataset/AMASS/smpl_module_project/data/smpl/SMPL_NEUTRAL.pkl'
```

---

## Fases de Implementación

### Fase 1 — Preprocesado de Datos (Persona 1)

**Archivos:** `src/data/amass_loader.py`, `src/data/preprocess_joints.py`

**Datasets ya disponibles en disco:**
- AMASS: colocar ficheros `.npz` en `src/dataset/AMASS/` (el módulo ya está, faltan los .npz)
- TNT15: ya en `src/dataset/TNT15_V1_0/Images/mr/` (~24.920 PNGs segmentados en 5 subcarpetas)
- NOMO3D: no disponible — datos 3D serán 100% sintéticos

**Preprocesado AMASS → joints.npz:**
1. Los ficheros AMASS contienen: `poses` (N, 156), `betas` (10,), `gender`, `mocap_framerate`
2. Extraer `poses[:, :72]` (body pose, 24 joints × 3 axis-angle) + `betas`
3. Aplicar SMPL forward pass via `smplx` para obtener joints 3D (24, 3)
4. **Normalización:**
   - Pelvis (joint 0) al origen: `joints -= joints[0]`
   - Escala uniforme: dividir por altura (pelvis→head)
   - Aplanar a vector de 72 dimensiones
5. Guardar en `internal/data/joints.npz`: array `(N, 72)`

---

### Fase 2 — Tabular GAN + Filtrado (Persona 1)

**Archivos:** `src/models/tabular_gan.py`, `src/training/train_tabular.py`, `src/data/discriminator_filter.py`

**Arquitectura:**
```
Generator:  z(128) → Linear(128,256) → BN → LeakyReLU
                   → Linear(256,512) → BN → LeakyReLU
                   → Linear(512,72)  → Tanh

Discriminator: x(72) → Linear(72,512)  → LeakyReLU → Dropout(0.3)
                      → Linear(512,256) → LeakyReLU → Dropout(0.3)
                      → Linear(256,1)   (sin activación — WGAN)
```

**Loss: WGAN-GP** — `L_D = E[D(fake)] - E[D(real)] + 10 * GP`, `L_G = -E[D(fake)]`

**Filtrado (discriminator_filter.py):**
```python
# Generar hasta tener N cuerpos aceptados
accepted = []
while len(accepted) < N_target:
    z = torch.randn(batch, 128)
    fake = G(z)              # (batch, 72)
    score = D(fake).sigmoid()  # normalizar a [0,1]
    accepted.extend(fake[score > 0.8])
# Guardar en internal/data/generated_joints.npz
```

---

### Fase 3 — Datos Imagen: TNT15 + Renders (Persona 2)

**Archivos:** `src/data/tn15_loader.py`, `src/data/render_skeleton.py`, `src/models/image_gan.py`, `src/training/train_image.py`

**TNT15 loader:** Cargar los PNGs segmentados desde `src/dataset/TNT15_V1_0/Images/mr/` (subcarpetas `00/` a `04/`). Son siluetas humanas (~26.581 PNGs totales). Resize a 128×128.

**Render skeleton:** joints filtrados → imagen 128×128 con Pillow (opencv NO instalado)
- Proyectar joints 3D → 2D (plano XY)
- Dibujar huesos (líneas) y joints (círculos) sobre fondo negro

**Conexiones de huesos SMPL:**
```python
SMPL_BONES = [
    (0,1),(0,2),(1,4),(2,5),(4,7),(5,8),   # piernas
    (0,3),(3,6),(6,9),(9,12),(12,15),       # columna + cabeza
    (9,13),(9,14),(13,16),(14,17),(16,18),(17,19)  # brazos
]
```

**DCGAN:**
```
Generator: z(128) → FC → reshape(512,4,4)
  → 5× ConvTranspose2d(stride=2, BN, ReLU) → [128×128] → Tanh

Discriminator: img(3,128,128)
  → 4× Conv2d(stride=2, BN, LeakyReLU) → Flatten → Linear → 1
```

**Dataset imagen = TNT15 PNGs segmentados (~26.581) + renders de joints filtrados** (render complementa variedad de poses)

---

### Fase 4 — Datos 3D: NOMO3D + Meshes Sintéticas (Persona 3)

**Archivos:** `src/data/nomo3d_loader.py`, `src/data/smpl_mesh_generator.py`, `src/models/mesh_gan.py`, `src/training/train_mesh.py`

**NOMO3D loader (`nomo3d_loader.py`):**
- Los OBJ están separados por género:
  - `female/` → 177 OBJ (`female_0000.obj` … `female_0176.obj`)
  - `male/` → 179 OBJ (`male_0000.obj` … `male_0178.obj`)
- Medidas en `female_meas_txt/` y `male_meas_txt/` (usar en eval como ground truth)
- Cargar con `trimesh.load()`, submuestrear con `trimesh.sample.sample_surface(mesh, 6890)`
- Submuestrear superficie a 6890 puntos: `trimesh.sample.sample_surface(mesh, 6890)`
- Los ficheros de medidas (csv/txt) se pueden usar como ground truth en `eval_3d.py`

```python
import trimesh
mesh = trimesh.load('scan.obj')
points, _ = trimesh.sample.sample_surface(mesh, 6890)
# points.shape == (6890, 3)
```

**SMPL mesh generator (`smpl_mesh_generator.py`):**
```python
import smplx
model = smplx.create(SMPL_MODEL_PATH, model_type='smpl')
output = model(betas=betas, body_pose=poses)
vertices = output.vertices  # (1, 6890, 3) — topología SMPL exacta
```

**Mesh GAN (PointNet-style):**
```
Generator: z(128) → MLP → (6890×3)
  Linear(128,1024) → ReLU → Linear(1024,4096) → ReLU
  → Linear(4096, 6890*3) → reshape(6890,3)

Discriminator (PointNet):
  Per-point: (6890,3) → (6890,64) → (6890,128) → (6890,256)
  → MaxPool global → (256,) → Linear(256,128) → Linear(128,1)
```

**Dataset 3D = NOMO3D scans subsampled (~400) + meshes SMPL generadas y filtradas**

---

### Fase 5 — Evaluación (Personas 1, 2, 3)

- **`eval_tabular.py`**: MMD (kernel RBF, scipy) + bone length error
- **`eval_image.py`**: FID con `pytorch-fid` (instalar: `pip install pytorch-fid`)
- **`eval_3d.py`**: Chamfer Distance (`chamfer-distance` ya instalado) + F-score (τ=0.01)

---

### Fase 6 — Integración y Reporte (Persona 4)

- `main.py`: `Paths.init_project()` + orquestar todo el pipeline
- `laTex/main.tex`: Abstract, Intro, Metodología, Resultados (tablas métricas, grids de muestras), Conclusiones
- Notebook `notebooks/proyecto_gan.ipynb`: documentación completa (ver sección Regla de Notebook)

---

## Especificaciones Técnicas

| Parámetro | Valor |
|-----------|-------|
| Joints (tabular) | 24 × 3 = **72 dim** |
| 3D mesh | **6890 vértices** (topología SMPL) |
| Imagen | **128×128 RGB** |
| Latent space `z` | dim=128, N(0,1) |
| Loss | **WGAN-GP**, λ=10 |
| Ratio D/G | 5:1 (5 pasos D por 1 G) |
| Batch size | 64 |
| Learning rate | 1e-4 (Adam, β1=0.0, β2=0.9) |
| Epochs | 200+ |
| Filtro discriminador | score_D > **0.8** |
| Semilla | `torch.manual_seed(42)`, `np.random.seed(42)` |
| Checkpoints | cada 50 épocas en `internal/experiments/` |

---

## Dependencias Disponibles (entorno conda `smpl-env`)

Ya instaladas:
- `torch==1.13.1` + CUDA 11.8
- `smplx==0.1.28` — SMPL forward pass
- `trimesh==3.9.35` — cargar OBJ (NOMO3D), operaciones 3D
- `open3d==0.17.0` — visualización point clouds
- `chamfer-distance==0.1` — métrica 3D
- `geomloss==0.2.6` — EMD/Sinkhorn
- `plotly==5.18.0` — visualización 3D interactiva
- `pillow==9.5.0` — render skeleton (opencv NO instalado)
- `scipy`, `numpy`, `matplotlib`, `tqdm`, `pandas`

**Instalar antes de implementar:**
```bash
pip install pytorch-fid   # para FID en eval_image.py
```

**Nota:** `opencv-python` NO está en el entorno. Usar `Pillow` (PIL) para dibujar imágenes de skeleton.

---

## Comandos de Verificación

```bash
# 1. Preprocessing AMASS → joints
python src/data/preprocess_joints.py
# Esperado: internal/data/joints.npz shape=(N, 72)

# 2. Entrenar Tabular GAN
python src/training/train_tabular.py
# Esperado: internal/experiments/tabular/checkpoint_ep200.pt

# 3. Filtrar cuerpos generados (score_D > 0.8)
python src/data/discriminator_filter.py --n 1000
# Esperado: internal/data/generated_joints.npz

# 4. Generar meshes SMPL desde filtrados
python src/data/smpl_mesh_generator.py
# Esperado: internal/data/meshes/*.npz (6890 verts cada una)

# 5. Render imágenes skeleton
python src/data/render_skeleton.py
# Esperado: internal/data/skeleton_images/*.png

# 6. Training Image GAN
python src/training/train_image.py
# Esperado: internal/experiments/image/checkpoint_ep200.pt

# 7. Training Mesh GAN
python src/training/train_mesh.py
# Esperado: internal/experiments/mesh/checkpoint_ep200.pt

# 8. Evaluación
python src/evaluation/eval_tabular.py  # → MMD, bone error
python src/evaluation/eval_image.py    # → FID
python src/evaluation/eval_3d.py       # → Chamfer Distance, F-score
```

---

## División de Tareas

| Persona | Responsabilidad | Entregables |
|---------|----------------|-------------|
| **1** | Data Lead + Tabular GAN | `amass_loader.py`, `preprocess_joints.py`, `tabular_gan.py`, `train_tabular.py`, `discriminator_filter.py`, `eval_tabular.py` |
| **2** | Image GAN | `tn15_loader.py`, `render_skeleton.py`, `image_gan.py`, `train_image.py`, `eval_image.py` |
| **3** | 3D Mesh GAN | `nomo3d_loader.py`, `smpl_mesh_generator.py`, `mesh_gan.py`, `train_mesh.py`, `eval_3d.py` |
| **4** | Integración + Reporte | `main.py` completo, `laTex/main.tex`, `notebooks/proyecto_gan.ipynb`, consolidación de resultados |

---

## Correcciones Previas a la Implementación

Estas correcciones deben hacerse **antes** de empezar a codificar:

### 1. Crear `__init__.py` vacíos (necesarios para imports Python)
```bash
touch src/__init__.py
touch src/config/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
```

### 2. Corregir `main.py` — import roto
```python
# ACTUAL (no funciona — Config no existe):
from src.config import Config

# CORRECTO:
from src.config.paths import Paths
from src.config.utils import Utils
```

### 3. Añadir ficheros `.npz` de AMASS en `src/dataset/AMASS/`
- Formato: `poses` (N,156), `betas` (10,), `gender`, `mocap_framerate`

### 4. Instalar pytorch-fid
```bash
pip install pytorch-fid
```

### 5. Notebook
- `notebooks/n1.ipynb` existe pero está **vacío** (0 bytes)
- Crear `notebooks/proyecto_gan.ipynb` como notebook principal (ver Regla de Notebook)

---

## Notas Importantes

- Usar `Paths` de `src/config/paths.py` para todas las rutas — no hardcodear paths
- Los modelos SMPL están en `src/dataset/AMASS/smpl_module_project/data/smpl/` — no mover
- Los `.npz` de AMASS van en `src/dataset/AMASS/` para que `amass_loader.py` los encuentre
- TNT15: usar los PNGs de `Images/mr/` (NO `pz/`) directamente — no extraer los MP4
- NOMO3D: OBJs en subcarpetas `female/` y `male/` dentro de `nomo-scans(repetitions-removed)/`
- `opencv-python` NO está instalado — usar `Pillow` para renderizar imágenes
- `smpl_module_project` existe en 3 rutas — usar siempre `src/dataset/AMASS/smpl_module_project/` (la única con los .pkl)
- Priorizar estabilidad del entrenamiento antes que complejidad arquitectónica
- Documentar resultados en `internal/experiments/`
