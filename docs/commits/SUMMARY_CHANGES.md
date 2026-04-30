# Resumen de Cambios: Pipeline GAN 3D

Se ha implementado un pipeline completo para la generación de cuerpos 3D (SMPL) condicionado por datos antropométricos tabulares.

## Nuevos Archivos

| Archivo | Propósito |
| :--- | :--- |
| `src/config/hparams.py` | Configuración de hiperparámetros (arquitectura, entrenamiento, columnas). |
| `src/data/dataset.py` | Cargador de datos NOMO3D. Procesa archivos `.txt` de medidas. |
| `src/data/beta_fitter.py` | Generación de caché de parámetros `betas` (pseudo-ground-truth). |
| `src/models/generator.py` | Generador MLP: (Ruido + Medidas) -> 10 Betas SMPL. |
| `src/models/discriminator.py` | Discriminador MLP: (Betas + Medidas) -> Score Real/Fake (WGAN). |
| `src/train.py` | Bucle de entrenamiento WGAN-GP con penalización de gradiente. |
| `src/eval.py` | Evaluación de Error Absoluto Medio (MAE) entre medidas reales y generadas. |
| `src/inference.py` | CLI para generar y exportar mallas `.obj` desde parámetros. |
| `notebooks/n2_eda.py` | Script de análisis exploratorio de datos (distribuciones por género). |

## Archivos Modificados

| Archivo | Cambios |
| :--- | :--- |
| `src/config/paths.py` | Añadidas rutas para NOMO3D (female/male/meas) y TNT15. |
| `main.py` | Refactorizado como punto de entrada CLI (`fit`, `train`, `eval`, `infer`). |

## Instrucciones de Uso

1. **Pre-procesado**: `python main.py fit`
2. **Entrenamiento**: `python main.py train`
3. **Evaluación**: `python main.py eval`
4. **Inferencia**: `python src/inference.py --height 175 --bust 95 --gender MALE --show`
