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

## Piezas Faltantes (a implementar)

### 1. `fit_betas()` real — CRÍTICO

**Fichero:** `src/data/beta_fitter.py`

**Problema actual:** devuelve `np.random.normal(0, 0.5, num_betas)`. El GAN aprende ruido.

**Solución:** optimización inversa con PyTorch:
```python
for each sample in NOMO3D:
    betas = torch.zeros(1, 10, requires_grad=True)
    optimizer = torch.optim.Adam([betas], lr=0.01)
    target_meas = sample['measurements']  # del .txt

    for step in range(500):
        measurer.from_body_model(gender, shape=betas)
        measurer.measure(measurer.all_possible_measurements)
        pred_meas = [measurer.measurements[name_map[k]] for k in target_meas_keys]
        loss = F.mse_loss(torch.tensor(pred_meas), target_meas)
        loss.backward(); optimizer.step()

    np.save(cache_file, betas.detach().numpy())
```

### 2. Image GAN

**Ficheros a crear:**
- `src/data/tnt15_loader.py` — carga PNGs de `Images/mr/`, resize 128×128, normaliza
- `src/data/render_skeleton.py` — proyecta joints SMPL a 2D con Pillow (sin opencv)
- `src/models/image_gan.py` — DCGAN 128×128 (5 ConvTranspose en G, 4 Conv en D)
- `src/train_image.py` — bucle de entrenamiento WGAN-GP imagen

### 3. Mesh GAN directo

**Ficheros a crear:**
- `src/data/nomo3d_obj_loader.py` — `trimesh.sample.sample_surface(mesh, 6890)`
- `src/models/mesh_gan.py` — MLP gen (z → 6890×3) + PointNet disc
- `src/train_mesh.py` — bucle WGAN-GP point cloud

### 4. Métricas complementarias

- `src/eval_tabular.py` — MMD (kernel RBF)
- `src/eval_image.py` — FID (pytorch-fid, requiere `pip install pytorch-fid`)
- `src/eval_3d.py` — Chamfer Distance + F-score (chamfer-distance ya instalado)

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
