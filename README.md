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
