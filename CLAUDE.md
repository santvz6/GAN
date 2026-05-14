# CLAUDE.md — Plan de Acción: GAN para Síntesis de Cuerpo Humano

## Estrategia Actual

**Base:** código de Santi (`origin/dev_santi`) — WGAN-GP condicional que aprende `medidas antropométricas → betas SMPL`.
**Complementos a añadir:** las piezas que faltan respecto a la spec original (ver "Piezas Faltantes" más abajo).

---

## Regla de Workflow — Git

**Cada cambio se sube a GitHub en la rama `dev_taron`.**

```bash
git add <archivos modificados>
git commit -m "descripción del cambio"
git push origin dev_taron
```

- Nunca hacer push a `main` directamente.
- Tag de respaldo del estado anterior: `dev_taron_pre_santi` (por si hay que volver atrás).

---

## Regla de Notebook

**Notebook principal:** `notebooks/proyecto_gan.ipynb` — debe documentar TODO lo realizado en lenguaje claro para alguien sin conocimientos de programación.

Estructura:
1. Introducción — qué es una GAN y por qué se usa para cuerpos humanos
2. Dataset — NOMO3D (medidas + escaneos), TNT15 (imágenes), SMPL (modelo paramétrico)
3. Pipeline — medidas → betas → malla
4. Tabular GAN (medidas→betas) — arquitectura, entrenamiento, resultados
5. Image GAN — generación de imágenes de cuerpo, entrenamiento, resultados
6. Mesh GAN — nubes de puntos directas, entrenamiento, resultados
7. Métricas — MAE, MMD, FID, Chamfer Distance, explicadas en lenguaje simple
8. Conclusiones

Cada cambio en código se refleja con su explicación en el notebook.

---

## Arquitectura del Proyecto (Base de Santi)

### Pipeline principal — medidas → betas → malla

```
medidas (10 dims: altura, busto, cintura, cadera, cuello, hombro, inseam, outseam, muslo, bicep)
        │
        ▼
G(z, medidas)  ──►  betas SMPL (10 dims)
        ▲
        │
        D(betas, medidas)  (WGAN-GP)
        │
NOMO3D female_meas_txt/ + male_meas_txt/  (medidas reales)
NOMO3D scans  ──►  fit_betas()  ──►  betas ground-truth
        │
        ▼
SMPL forward pass  ──►  malla 6890 verts
```

### Estructura de directorios

```
GAN/
├── CLAUDE.md                       (este archivo)
├── README.md
├── TODO.md
├── main.py                         (CLI: fit | train | eval | infer)
├── convert_smpl.py                 (chumpy .pkl → numpy .pkl)
├── requirements.txt
├── docs/
│   ├── CAVEMAN.md
│   └── commits/                    (notas por commit)
├── notebooks/
│   ├── n1.ipynb                    (notebook de Santi)
│   ├── n2_eda.py
│   └── proyecto_gan.ipynb          (notebook principal del proyecto)
├── laTex/                          (memoria LaTeX)
├── src/
│   ├── config/
│   │   ├── paths.py                (rutas — usar siempre Paths.init_project())
│   │   ├── hparams.py              (hiperparámetros centralizados)
│   │   └── utils.py
│   ├── data/
│   │   ├── dataset.py              (NOMODataset — parser de _meas.txt)
│   │   └── beta_fitter.py          (TODO: implementar optimización real)
│   ├── models/
│   │   ├── generator.py            (Conditional G con bloque residual)
│   │   └── discriminator.py        (Conditional D, sin BatchNorm)
│   ├── smpl_module_project/        (módulo del profesor — NO modificar)
│   ├── train.py                    (WGAN-GP trainer)
│   ├── eval.py                     (MAE de medidas extraídas)
│   └── inference.py                (medidas → betas → mesh OBJ)
└── internal/                       (generado por Paths.init_project)
    ├── data/
    │   ├── smpl/                   (modelos SMPL .pkl — fuera de git)
    │   ├── nomo3d/                 (link/copia de NOMO3D — fuera de git)
    │   ├── tnt15/                  (link/copia de TNT15 — fuera de git)
    │   └── betas_cache/            (betas ground-truth fitteadas)
    ├── experiments/                (checkpoints)
    ├── logs/                       (logs de entrenamiento)
    └── temp/
```

---

## Datasets

| Dataset | Rol | Estado |
|---------|-----|--------|
| **NOMO3D** | Tabular GAN (medidas→betas) + Mesh GAN | Disponible (`src/dataset/NOMO3D/`) |
| **TNT15** | Image GAN (siluetas 128×128) | Disponible (`src/dataset/TNT15_V1_0/Images/mr/`) |
| **SMPL** | Modelo paramétrico de cuerpo | `src/dataset/AMASS/smpl_module_project/data/smpl/*.pkl` |
| **SMPL+H** | Variante con manos articuladas | `src/dataset/AMASS/body_models_smplh/` (descargado, no usado por ahora) |
| **AMASS motion** | NO necesario en este enfoque | — |

**Nota:** el enfoque de Santi NO requiere AMASS motion capture. La condición del GAN son medidas, no poses.

---

## Hiperparámetros (de `src/config/hparams.py`)

| Parámetro | Valor |
|-----------|-------|
| `noise_dim` | 64 |
| `cond_dim` | 10 (medidas) |
| `num_betas` | 10 |
| `g_hidden_dims` | [256, 512, 1024, 512, 256] (con residual en peak) |
| `d_hidden_dims` | [256, 128, 64] |
| Batch size | 128 |
| Epochs | 2000 |
| `lr_g`, `lr_d` | 1e-4 |
| `b1`, `b2` | 0.5, 0.9 |
| `n_critic` (WGAN-GP) | 5 |
| `lambda_gp` | 10.0 |
| `checkpoint_interval` | 500 epochs |
| `sample_interval` | 100 epochs |

---

## Comandos

```bash
# 1. Fit pseudo-GT betas (TODO: actualmente placeholder)
python main.py fit

# 2. Entrenar el WGAN-GP condicional
python main.py train

# 3. Evaluar MAE de medidas extraídas
python main.py eval

# 4. Inferencia: medidas → mesh OBJ
python main.py infer --gender FEMALE --height 170 --bust 90 --waist 70 --hip 95 \
                     --neck 34 --shoulder 40 --inseam 80 --outseam 100 \
                     --thigh 55 --bicep 28 --show
```

---

## Componentes Implementados

### 1. Tabular GAN (medidas → betas) — base de Santi + `fit_betas()` real

- `src/data/beta_fitter.py` — `fit_betas()` con scipy Nelder-Mead optimiza los 10 betas SMPL para cada sample NOMO3D minimizando MSE ponderado contra las medidas extraídas con `MeasureSMPL`. Usa `CachedMeasureSMPL` (subclase que cachea el modelo smplx por género) y un simplex inicial con paso 0.5 para evitar estancamiento en el origen.
- `src/train.py`, `src/eval.py`, `src/inference.py`, `src/models/generator.py`, `src/models/discriminator.py` — heredados de Santi sin cambios.

### 2. Image GAN (TNT15 + skeleton renders) — DCGAN 128×128 WGAN-GP

- `src/data/render_skeleton.py` — `SkeletonRenderer` proyecta joints SMPL al plano XY y los dibuja con Pillow (huesos + joints). Función `render_batch_from_betas_cache()` produce un PNG por cada beta cacheado.
- `src/data/tnt15_loader.py` — `TNT15Dataset` carga PNGs de `Images/mr/00..04`, resize 128×128, grayscale, normaliza a [-1,1]. Opcionalmente mezcla con renders sintéticos.
- `src/models/image_gan.py` — `ImageGenerator` (FC → reshape 4×4 → 5 ConvTranspose 2× + LayerNorm) y `ImageDiscriminator` (4 Conv stride-2 + LayerNorm, sin BatchNorm).
- `src/train_image.py` — `ImageWGANGPTrainer`, mismo patrón WGAN-GP que el tabular (gradient penalty, n_critic=5). Guarda grid de 16 muestras en `internal/temp/image_samples/` cada `sample_interval` epochs.

### 3. Mesh GAN directo (NOMO3D point clouds) — PointNet WGAN-GP

- `src/data/nomo3d_obj_loader.py` — `NOMO3DPointCloudDataset` carga los 356 OBJ, subsamplea a 6890 puntos con `trimesh.sample.sample_surface`, centra y escala a max-norm unitario. Cachea cada nube en `internal/data/nomo3d_pointclouds/`.
- `src/models/mesh_gan.py` — `MeshGenerator` (MLP z → 6890×3 con LayerNorm) y `PointNetDiscriminator` (Conv1d per-punto + max-pool global + cabeza MLP, sin sigmoid).
- `src/train_mesh.py` — `MeshWGANGPTrainer`, gradient penalty con interpolación lineal entre nubes reales y falsas. Guarda muestras como `.npy` (cargables con `np.load` para visualizar con trimesh/plotly).

### 4. Métricas

- `src/eval_tabular.py` — `evaluate_mmd()`: MMD² con suma de kernels RBF multi-bandwidth, estimador U insesgado.
- `src/eval_image.py` — `evaluate_fid()`: dump N reales + N generados, llama a `python -m pytorch_fid` y parsea el score.
- `src/eval_3d.py` — `chamfer_distance()` simétrico + `fscore(tau=0.01)`, promedio sobre N muestras.

### 5. CLI consolidado

`main.py` ahora expone:

```
fit                  — fit_betas() (preprocesa pseudo-GT)
train                — Tabular WGAN-GP
eval                 — MAE de medidas (Santi)
infer ...            — medidas → mesh OBJ
render-skeletons     — render PNGs desde betas_cache/
train-image          — Image GAN (TNT15)
eval-image           — FID
train-mesh           — Mesh GAN (NOMO3D point clouds)
eval-mesh            — Chamfer + F-score
eval-mmd             — MMD del tabular GAN
```

---

## Reutilización del Módulo del Profesor

**Ruta:** `src/smpl_module_project/` (la versión de Santi).
**NO modificar.** Solo importar.

```python
from src.smpl_module_project.measure import MeasureSMPL, MeasureBody
from src.smpl_module_project.evaluate import evaluate_mae
from src.smpl_module_project.joint_definitions import SMPL_IND2JOINT
from src.smpl_module_project.landmark_definitions import SMPL_LANDMARK_INDICES
from src.smpl_module_project.visualize import Visualizer
```

Los modelos SMPL `.pkl` deben estar en `internal/data/smpl/`. Si no, se pueden copiar desde `src/dataset/AMASS/smpl_module_project/data/smpl/`.

---

## Dependencias (de `requirements.txt`)

```
torch==2.5.1+cu121
torchvision==0.20.1+cu121
scipy==1.17.1
tqdm==4.67.3
pandas==3.0.2
seaborn==0.13.2
plotly==6.7.0
trimesh==4.12.1
smplx==0.1.28
pyglet==1.5.31
```

**A instalar:**
- `pip install pytorch-fid` (para FID en Image GAN eval)

**Nota chumpy:** los `.pkl` originales de SMPL contienen objetos `chumpy.Ch`. Usar `convert_smpl.py` para convertirlos a numpy puro antes de cargarlos con PyTorch moderno.

---

## Tareas Activas

Ver lista de tareas vía `/tasks` o TaskList. Tareas iniciales tras rebase:
1. ✓ Preservar estado de dev_taron (tag `dev_taron_pre_santi`)
2. ✓ Rebasar dev_taron sobre origin/dev_santi
3. Restaurar notebook y CLAUDE.md actualizado
4. Implementar `fit_betas()` real (CRÍTICO)
5. Añadir Image GAN (TNT15 + skeleton renders)
6. Añadir Mesh GAN directo (PointNet)
7. Añadir métricas complementarias (MMD, FID, Chamfer)

---

## Notas Importantes

- Usar siempre `Paths` de `src/config/paths.py` — no hardcodear rutas.
- Los `.pkl` y `.npz` están en `.gitignore` — los modelos SMPL viven fuera de git.
- TNT15 y NOMO3D están en `src/dataset/`, también fuera de git.
- El módulo del profesor (`smpl_module_project`) no se modifica.
- Priorizar estabilidad de entrenamiento sobre complejidad arquitectónica.
- Documentar resultados en `internal/experiments/` y reflejarlos en el notebook.
