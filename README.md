
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
