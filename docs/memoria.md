# Notas para la memoria del proyecto

Este documento recoge explicaciones en prosa de los distintos bloques del proyecto, escritas con el detalle suficiente para servir de base directa a la memoria final en LaTeX. Cada sección puede leerse de forma independiente.

---

## Rama de imagen — GAN sobre TNT15

### Punto de partida y decisión arquitectónica

El proyecto ya contaba con un GAN tabular que aprende a generar 10 betas SMPL condicionadas por 10 medidas antropométricas. La rama de imagen es una **segunda red completamente separada**, que vive en paralelo a la tabular y reutiliza la infraestructura común (rutas, configuración, esquema de checkpoints). La razón de mantenerlas independientes es que las preguntas que responden son distintas: la tabular hace `medidas → forma SMPL`, mientras que la de imagen hace `ruido → imagen humana realista`. Mezclar ambas en un único modelo habría obligado a renderizar SMPL → imagen en cada paso de entrenamiento, lo que multiplica por veinte o treinta el coste de cómputo sin aportar nada para el objetivo concreto planteado ("producir otra imagen que se parezca a un humano").

La pérdida elegida es **WGAN-GP**, que es la misma que ya usa el trainer tabular existente. Esta consistencia tiene dos ventajas prácticas: el código del bucle de entrenamiento se parece mucho al tabular (lo que facilita lectura y revisión cruzada en la memoria), y WGAN-GP es notablemente más estable que el minimax clásico cuando la dimensión de salida es alta — algo que importa aquí, porque pasamos de un vector de 10 betas a una imagen de 4 096 píxeles. La arquitectura sigue el patrón **DCGAN**: capas de `ConvTranspose2d` en el generador para subir de un ruido latente a una imagen, y capas de `Conv2d` en el discriminador para bajar de imagen a un escalar.

### El dataset: TNT15

TNT15 contiene cuatro sujetos (`mr`, `pz`, `sg`, `sp`) grabados desde cinco cámaras (`00`–`04`), con vídeos descompuestos en PNGs ya segmentados (el sujeto en gris sobre fondo negro). Cada cámara aporta entre 3 000 y 8 000 frames, lo que da un total bruto de unos 100 000. Como muchos de esos frames son prácticamente idénticos (vídeo a 30 fps), entrenar con todos satura disco e introduce redundancia que ralentiza la convergencia sin mejorar la diversidad. El dataset (`src/data/img_dataset.py`) aplica por eso un **submuestreo por stride** (`frame_stride=5` por defecto, configurable), quedándose con uno de cada cinco frames de cada cámara. Tras ese filtro queda un corpus efectivo de ~20 600 imágenes, suficientemente diverso para WGAN-GP pero manejable.

El pipeline de carga es deliberadamente simple. Cada PNG se lee con PIL, se convierte a un canal (grayscale, que es lo nativo de TNT15), se redimensiona y se recorta al tamaño objetivo (64×64 por defecto), se pasa a tensor en `[0, 1]` y finalmente se normaliza a `[-1, 1]`. Esa normalización es crucial: el generador termina en una activación `tanh` que produce salidas en `[-1, 1]`, así que los datos reales tienen que vivir en el mismo rango para que las dos distribuciones (real y falsa) sean comparables por el discriminador. El split train/test se hace por orden alfabético tras el sort estable, con un 90/10 — no es un split estricto en el sentido estadístico (frames consecutivos están correlacionados), pero para una primera evaluación cualitativa basta.

### El generador (`src/models/img_generator.py`)

El generador parte de un vector de ruido `z` de 128 dimensiones, lo expande a un tensor `(B, 128, 1, 1)`, y lo va subiendo de resolución mediante cuatro `ConvTranspose2d` con stride 2. La progresión típica de DCGAN multiplica por dos el lado espacial y divide por dos el número de canales en cada bloque: `1×1 → 4×4 → 8×8 → 16×16 → 32×32 → 64×64`. Cada bloque intermedio lleva `BatchNorm2d` + `ReLU`, que es la receta clásica probada en el paper original de Radford y suele convergir bien. La capa final no tiene BatchNorm (introduce ruido innecesario) y usa `tanh` para dejar la imagen en `[-1, 1]`. Hay un pequeño detalle de cómputo: el número de bloques de upsample se deriva por `log2(image_size) − 2`, lo que hace que la arquitectura funcione automáticamente para 32×32, 64×64, 128×128, etc., sin tocar el código. La inicialización de pesos sigue también la receta DCGAN (normal 0/0.02 en convoluciones, normal 1/0.02 en BatchNorms).

### El discriminador (`src/models/img_discriminator.py`)

El discriminador hace el camino inverso: parte de la imagen `(B, 1, 64, 64)` y aplica cinco `Conv2d` con stride 2 que la reducen progresivamente hasta un escalar. La salida es un único número por imagen, **sin sigmoid**, porque WGAN-GP interpreta el discriminador como un "crítico" que estima la distancia de Wasserstein, no como un clasificador binario probabilístico.

Aquí hay un detalle técnico importante: en lugar de usar `BatchNorm2d` se usa **`GroupNorm` con un solo grupo**, que es equivalente a `LayerNorm` sobre `(C, H, W)`. La razón es que el gradient penalty de WGAN-GP requiere derivar la salida del discriminador respecto a cada muestra individualmente, y BatchNorm acopla las muestras del batch entre sí (la media y varianza se calculan sobre todo el batch), lo que rompe ese supuesto. LayerNorm/GroupNorm normalizan por muestra, así que respetan la independencia que GP necesita. Es el mismo motivo por el que el discriminador tabular original tampoco lleva BatchNorm.

### El bucle de entrenamiento (`src/train_img.py`)

El trainer está modelado sobre `WGANGPTrainer` del pipeline tabular, con tres adaptaciones. La primera es que el **gradient penalty se calcula sobre imágenes**, no sobre vectores: el tensor `alpha` que interpola entre real y fake tiene forma `(B, 1, 1, 1)` en lugar de `(B, 1)`, y los gradientes se aplanan antes de tomar la norma. La segunda es que el discriminador se actualiza `n_critic=5` veces por cada actualización del generador, una práctica estándar en WGAN-GP que da al crítico tiempo de adaptarse antes de que el generador se mueva. La tercera es que **cada epoch se guarda un grid de muestras** generadas con el *mismo* ruido fijo (`self._fixed_z`), de modo que al revisar `internal/logs/img_samples/sample_epoch_*.png` se ve la evolución cualitativa del mismo conjunto de "puntos latentes" a lo largo del entrenamiento — esto es mucho más informativo que ver muestras aleatorias distintas en cada epoch.

Los checkpoints se guardan con prefijo `wgangp_img_ckpt_*.pt` (diferente del prefijo tabular `wgangp_ckpt_*.pt`) para que ambos pipelines puedan coexistir en la misma carpeta `internal/experiments/` sin pisarse. El optimizador es Adam con `β₁ = 0.5`, `β₂ = 0.9` (los valores recomendados por el paper de WGAN-GP, no los de Adam por defecto, porque las betas estándar introducen demasiada inercia para esta pérdida).

### Inferencia y orquestación (`src/inference_img.py`, `main.py`)

La inferencia busca el último checkpoint de imagen por mtime, restaura el generador, genera N muestras desde ruido independiente y las guarda en `internal/temp/`, ya sea como ficheros sueltos (`generated_0000.png`, ...) o como un grid único con `--grid`. Lo único que el script hace antes de guardar es deshacer la normalización (`(x + 1) / 2`) para que `torchvision.utils.save_image` reciba valores en `[0, 1]`.

Todo está cableado a `main.py` mediante dos subcomandos nuevos, `train_img` e `infer_img`, que se suman a los cuatro tabulares (`fit`, `train`, `eval`, `infer`) sin tocarlos. El flujo completo desde la línea de comandos queda:

```bash
python main.py train_img                # entrena el GAN sobre TNT15
python main.py infer_img -n 64 --grid   # genera un grid 8×8 con el último checkpoint
```

### Portabilidad entre máquinas

La ruta al dataset TNT15 no está hardcoded a la máquina de ningún colaborador. La resolución sigue este orden:

1. La variable de entorno `TNT15_ROOT` (si está definida y apunta a una carpeta que contiene `Images/`).
2. La ubicación canónica del proyecto: `internal/data/tnt15/` (relativa a la raíz del repo).

Si TNT15 está dentro del repo, no hay que hacer nada. Si está en otro disco, basta con `export TNT15_ROOT=/ruta/al/TNT15_V1_0`. El mensaje de error del dataset indica explícitamente ambas opciones para que cualquiera del equipo pueda configurarlo sin leer el código.

### Lo que deliberadamente no hace este diseño

Conviene dejar claro qué decisiones se tomaron y por qué, para evitar que parezcan omisiones. **No es condicional**: aunque el TODO mencionaba comparar MLP vs CNN como condicionador, TNT15 sólo tiene cuatro sujetos distintos, así que condicionar por identidad o por medidas degenera a un problema con cuatro clases y deja de ser una comparación útil. **No usa color**: TNT15 ya viene en grayscale segmentado, y mantener un solo canal triplica la velocidad de entrenamiento. **No usa Models_31par**: esos fits paramétricos están a nivel sujeto, no a nivel frame, así que no aportan supervisión adicional a una arquitectura como ésta. Si en algún momento se quisiera pasar a un GAN condicional por pose, el `dataset.py` y los modelos se prestan a esa extensión añadiendo un argumento de condición en `forward`, pero el modelo actual aprende la distribución marginal de "imagen humana segmentada" sin más.
