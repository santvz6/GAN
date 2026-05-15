# 🚀 Proyecto GAN: Estimación de Masa y Cuerpo 3D (AMASS/SMPL)

## 📌 Fase 1: Configuración de Datos y Entorno
- [x] **Acceso a Datos:** Descargar el dataset [AMASS](https://amass.is.tue.mpg.de/).
- [ ] **Pre-procesamiento:**
    - [ ] Convertir nubes de puntos al formato **MSPN** (6890 puntos).
    - [ ] Normalizar los **10 datos tabulares** (inputs clave para la masa del usuario).
- [ ] **Integración SMPL/SMPL-X:** Configurar el pipeline para transformar los parámetros tabulares en mallas 3D.

## 🏗️ Fase 2: Arquitectura del Modelo (GAN)
- [ ] **Generador:**
    - [ ] Diseñar MLP (Multi-Layer Perceptron) para procesar datos tabulares.
    - [ ] Objetivo: Generar el vector de 10 puntos que represente la masa/forma real.
- [ ] **Discriminador:**
    - [ ] Implementar arquitectura para distinguir entre datos reales (AMASS) y generados.
    - [ ] *Nota:* Evaluar si el discriminador procesa el vector tabular o la reconstrucción 3D.
- [ ] **Codificadores (Opcional según arquitectura):**
    - [ ] CNN para procesamiento de imágenes (si se usa info visual).
    - [ ] MLP para datos tabulares.
    - [ ] PointNet/CNN 3D si se procesa la nube de puntos directamente.

## 🧪 Fase 3: Experimentación
- [ ] Definir **Hiperparámetros** (Learning rate, batch size, épocas).
- [ ] Entrenamiento del bucle GAN (Loss de Minimax).
- [ ] Realizar pruebas de "Test de Turing" del modelo para validar realismo.
- [ ] **Métricas:** Definir KPIs cuantitativos (Error medio en puntos, precisión en masa).

## 📝 Fase 4: Redacción de la Memoria (LaTeX)
### 📘 Preliminares
- [ ] Título y Nombres de autores.
- [ ] **Abstract:** Resumen ejecutivo del proyecto.
- [ ] **Keywords:** (ej. GAN, SMPL, Point Cloud, Anthropometry).

### 📖 Cuerpo del Trabajo
- [ ] **Introducción & Estado del Arte:** 
- [ ] Revisar [Goodfellow (2014)](https://arxiv.org/pdf/1406.2661) y [CGAN (2014)](https://arxiv.org/pdf/1411.1784).
- [ ] **Motivación y Objetivos:**
    - [ ] **Objetivo Principal:** Desarrollar una red GAN capaz de generar representaciones corporales realistas a partir de datos antropométricos mínimos.
    - [ ] **Objetivo Secundario 1:** Implementar un pipeline de transformación "Tabular a Cuerpo" que mapee el vector generado a la malla SMPL de 6890 puntos.
    - [ ] **Objetivo Secundario 2:** Evaluar la precisión del modelo en la reconstrucción de la masa del usuario utilizando solo 10 parámetros.
    - [ ] **Objetivo Secundario 3:** Comparar el rendimiento de distintas arquitecturas (MLP vs CNN) para el procesamiento de información condicional.
- [ ] **Metodología:**
    - [ ] Búsqueda en bases de datos (PubMed, Arxiv, IEEE).
    - [ ] **Pseudocódigo** del entrenamiento.
    - [ ] Diagramas de arquitectura (sin código fuente).
    - [ ] Gestión de riesgos y división de tareas.

### 📊 Análisis y Cierre
- [ ] **Resultados:** 
    - [ ] Tablas comparativas.
    - [ ] Imágenes de las nubes de puntos generadas vs. reales.
- [ ] **Conclusión:** Evaluación de cumplimiento de objetivos.
- [ ] **Referencias:** Formatear con Mendeley (exportar a `.bib`).
