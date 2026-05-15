# Proyecto GAN: Estado del Proyecto

## Fase 1 — Configuración y Datos
- [x] Estructura de proyecto (`src/config/paths.py`, `__init__.py`)
- [x] Loader AMASS (`src/data/amass_loader.py`)
- [x] Preprocesado joints (`src/data/preprocess_joints.py`)
- [x] Loader TNT15 (`src/data/tn15_loader.py`)
- [x] Loader NOMO3D (`src/data/nomo3d_loader.py`)
- [x] Render esqueleto con Pillow (`src/data/render_skeleton.py`)
- [x] Generador mallas SMPL (`src/data/smpl_mesh_generator.py`)
- [ ] **Descargar `.npz` de AMASS** y colocar en `src/dataset/AMASS/` — pendiente del usuario.

## Fase 2 — Modelos GAN (WGAN-GP)
- [x] Tabular GAN (`src/models/tabular_gan.py`) — MLP + gradient penalty
- [x] Image GAN (`src/models/image_gan.py`) — DCGAN
- [x] Mesh GAN (`src/models/mesh_gan.py`) — MLP + PointNet discriminator

## Fase 3 — Entrenamiento
- [x] `src/training/train_tabular.py`
- [x] `src/training/train_image.py`
- [x] `src/training/train_mesh.py`
- [x] Filtrado por discriminador (`src/data/discriminator_filter.py`)
- [ ] **Lanzar entrenamientos** — pendiente de GPU del usuario (~horas cada uno)

## Fase 4 — Evaluación
- [x] MMD + Bone Length (`src/evaluation/eval_tabular.py`)
- [x] FID (`src/evaluation/eval_image.py`)
- [x] Chamfer + F-Score (`src/evaluation/eval_3d.py`)
- [ ] **Instalar `pytorch-fid`** (`pip install pytorch-fid`) si aún no está
- [ ] Ejecutar `python main.py --eval` tras los entrenamientos

## Fase 5 — Integración y Documentación
- [x] `main.py` orquestador con flags por fase
- [x] Notebook principal (`notebooks/proyecto_gan.ipynb`)
- [x] LaTeX inicial con secciones (`laTex/main.tex`)
- [ ] Completar sección Resultados del LaTeX con números reales tras evaluación

## Comandos

```bash
# Pipeline completo (tras haber puesto los .npz de AMASS)
python main.py --all

# Paso a paso
python main.py --preprocess
python main.py --train-tabular
python main.py --filter --n 1000
python main.py --gen-meshes --n 500
python main.py --render --n 500
python main.py --train-image
python main.py --train-mesh
python main.py --eval
```

## Git workflow
- Todos los cambios van a la rama `dev_taron`, nunca a `main`.
