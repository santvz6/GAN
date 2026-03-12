# GANs para Síntesis de Cuerpo Humano (Tabular, Imagen y 3D)

## 1. Descripción General

Este proyecto tiene como objetivo entrenar modelos generativos basados en **GANs (Generative Adversarial Networks)** para producir representaciones plausibles del cuerpo humano en tres modalidades diferentes:

1. **Tabular** → Coordenadas 2D/3D de articulaciones (joints).
2. **Imagen** → Representaciones visuales (renderizado de esqueleto).
3. **3D** → Nube de puntos o representación geométrica tridimensional.

Cada modalidad debe incluir:
- Dataset suficientemente grande.
- Entrenamiento de generador y discriminador.
- Evaluación cuantitativa con métricas estándar.
- Evaluación cualitativa mediante visualización de muestras.

---

## 2. Dataset Recomendado

### Dataset principal sugerido: AMASS

AMASS es un meta-dataset que unifica múltiples datasets de motion capture bajo el modelo SMPL. Permite extraer:

- Joints 3D consistentes.
- Secuencias temporales.
- Vértices de superficie (mesh).

Ventajas:
- Gran tamaño.
- Estándar en investigación.
- Permite derivar imágenes y representaciones 3D.

Si no se dispone de un dataset grande de imágenes o 3D, se generarán a partir del dataset tabular (joints).

---

## 3. Definición Exacta de Representaciones

### 3.1 Modalidad Tabular (Joints)

Formato por muestra (pose estática recomendada):

- Número de joints: 22 o 24 (según SMPL).
- Dimensión: `(J, 3)` para 3D.
- Vector final: `(J * 3)`.

Normalización obligatoria:
- Pelvis en el origen.
- Escala uniforme.
- Orientación consistente.
- Normalización estadística opcional.

Salida del generador:
- Una pose estática (recomendado).
- Alternativamente, secuencia corta si el tiempo lo permite.

---

### 3.2 Modalidad Imagen

Representación recomendada:

Renderizado de esqueleto en imagen RGB:
- Tamaño: 128×128 o 256×256.
- Fondo negro.
- Líneas conectando joints.
- Puntos marcando articulaciones.

Las imágenes se generan automáticamente desde los joints tabulares.

---

### 3.3 Modalidad 3D

Opciones posibles:

Opción A (más simple):
- Usar joints como nube de puntos pequeña.

Opción B (más completa):
- Muestrear N puntos de la superficie SMPL.

Recomendación:
- Si el tiempo es limitado → Opción A.
- Si se quiere mayor nivel técnico → Opción B.

---

## 4. Arquitecturas

### 4.1 Tabular GAN (MLP-GAN)

Generador:
- MLP que mapea `z ∈ R^d` → vector `(J*3)`.

Discriminador:
- MLP binario real/fake.

Loss recomendada:
- WGAN-GP o Hinge Loss (más estable que GAN clásica).

---

### 4.2 Image GAN (CNN)

Arquitectura sugerida:
- DCGAN o ResNet GAN.

Entrada:
- Ruido z.

Salida:
- Imagen 128×128 o 256×256.

Activación final:
- `tanh`.

---

### 4.3 3D GAN (Point Cloud GAN)

Generador:
- MLP que produce `(N, 3)`.

Discriminador:
- Estilo PointNet:
  - MLP por punto.
  - Max pooling global.
  - Clasificador binario.

---

## 5. Métricas de Evaluación

### 5.1 Tabular

Distribución:
- MMD (Maximum Mean Discrepancy).

Plausibilidad geométrica:
- Error de longitud de huesos.
- Consistencia de proporciones.
- Límites angulares (si se implementa).

Visual:
- Render 3D de esqueletos generados.

---

### 5.2 Imagen

- FID (Fréchet Inception Distance).
- Visualización de grids de muestras.

---

### 5.3 3D

- Chamfer Distance.
- Earth Mover's Distance (opcional).
- F-score con umbral τ.

---


---

## 7. Pipeline General

### Paso 0 — Entorno
- Python 3.10+
- PyTorch
- numpy
- scipy
- matplotlib
- opencv
- torchmetrics
- open3d (opcional)

---

### Paso 1 — Dataset
- Descargar AMASS.
- Documentar subconjuntos usados.

---

### Paso 2 — Preprocesado
- Extraer joints desde SMPL.
- Normalizar.
- Guardar en formato `.npz`.

---

### Paso 3 — Derivados
- Generar imágenes skeleton.
- Generar point clouds.

---

### Paso 4 — Entrenamiento
- Entrenar Tabular GAN.
- Entrenar Image GAN.
- Entrenar 3D GAN.

---

### Paso 5 — Evaluación
- Ejecutar scripts de métricas.
- Generar visualizaciones.
- Guardar resultados.

---

### Paso 6 — Reporte
- Comparación cuantitativa.
- Discusión técnica.
- Conclusiones.

---

## 8. División del Trabajo (4 Personas)

---

### Persona 1 — Data Lead + Tabular GAN

Responsabilidades:
1. Descargar y organizar AMASS.
2. Implementar extracción de joints.
3. Normalización consistente.
4. Implementar MLP-GAN.
5. Implementar MMD y métricas geométricas.
6. Generar visualizaciones.

Entregables:
- `preprocess_amass_to_joints.py`
- `tabular_gan.py`
- `eval_tabular.py`
- Resultados tabular.

---

### Persona 2 — Image Lead

Responsabilidades:
1. Generar imágenes desde joints.
2. Implementar DCGAN.
3. Calcular FID.
4. Generar grids de imágenes.

Entregables:
- `render_skeleton_images.py`
- `image_gan.py`
- `eval_image_fid.py`

---

### Persona 3 — 3D Lead

Responsabilidades:
1. Construir point clouds.
2. Implementar GAN 3D.
3. Implementar Chamfer Distance.
4. Visualizar resultados 3D.

Entregables:
- `build_pointclouds.py`
- `pointcloud_gan.py`
- `eval_3d_cd_emd.py`

---

### Persona 4 — Integración y Reporte

Responsabilidades:
1. Configuración unificada.
2. Scripts reproducibles.
3. Consolidar resultados.
4. Redactar informe final.
5. Generar figuras comparativas.

Entregables:
- Scripts de entrenamiento.
- `final_report.md`
- Resultados consolidados.

---

## 9. Cronograma Sugerido

Semana 1:
- Dataset y preprocesado listos.

Semana 2:
- Entrenamiento tabular e imagen.

Semana 3:
- Implementación 3D.

Semana 4:
- Evaluación final y reporte.

---

## 10. Consideraciones Finales

- Usar semillas fijas para reproducibilidad.
- Documentar versiones de librerías.
- Guardar checkpoints.
- Comparar estabilidad entre distintas losses si hay tiempo.
- Priorizar estabilidad antes que complejidad arquitectónica.

---

## Resultado Esperado

Un pipeline reproducible que:

- Genere poses humanas plausibles en formato tabular.
- Genere imágenes coherentes de esqueletos.
- Genere representaciones 3D con estructura humana.
- Evalúe cuantitativamente cada modalidad.
- Compare rendimiento entre modalidades.
