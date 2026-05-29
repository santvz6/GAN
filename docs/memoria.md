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

---

## Rama Tab Transformer — GAN con nubes de puntos SMPL

### Motivación y diseño general

Además del GAN tabular clásico (MLP) que toma medidas antropométricas como condición, se implementó una **segunda arquitectura condicional basada en Transformer** que recibe como entrada una **nube de puntos 3D** extraída del cuerpo SMPL. La idea es explorar si un mecanismo de atención sobre tokens geométricos puede capturar mejor la correlación espacial entre la forma del cuerpo y los parámetros betas, frente al enfoque MLP que recibe las medidas como un vector plano.

La decisión de usar un Tab Transformer (originalmente diseñado para datos tabulares, donde cada feature es un token independiente procesado por self-attention) sobre nubes de puntos es **deliberadamente un "misuse"**: se trata cada punto 3D como si fuera una columna categórica/numérica de una tabla, sin codificación posicional geométrica ni invarianza a permutaciones. El objetivo del experimento es precisamente medir qué tan mal funciona esta inadecuación frente al MLP condicionado por medidas, para justificar en la memoria que la representación del condicionamiento importa.

### El dataset de nubes de puntos (`src/data/pc_dataset.py`)

El dataset reutiliza los betas pseudo-ground-truth que ya se ajustaron con el comando `fit` (guardados en `internal/data/betas_cache/*.npy`). Para cada muestra:

1. Se carga el fichero de betas cacheado (`.npy`).
2. Se usa el modelo SMPL (cargado sin dependencia de chumpy, mediante un stub que convierte los arrays `chumpy.Ch` a numpy) para computar los vértices del cuerpo: `V = v_template + shapedirs @ betas`, que produce una malla de 6 890 vértices × 3 coordenadas.
3. Se **submuestrean 256 vértices** (configurable vía `TabHParams.n_pc_points`) usando una semilla fija (`np.random.seed(0)`), lo que garantiza que los mismos índices de vértice se usan en entrenamiento, evaluación e inferencia. Los índices se ordenan tras el muestreo para estabilidad.
4. La nube de puntos resultante `(256, 3)` se cachea en `internal/data/pc_cache/{id}.npy` para evitar recomputarla en cada epoch.

El cargador SMPL (`_load_smpl_pkl`) inyecta un módulo stub de chumpy en `sys.modules` para que `pickle.load` pueda reconstruir los objetos serializados sin tener chumpy instalado. Tras la carga, limpia todos los stubs. Esto evita instalar chumpy (que tiene incompatibilidades con NumPy moderno y Python 3.11+) manteniendo la compatibilidad con los `.pkl` originales del proyecto SMPL.

El split train/test usa la misma semilla y proporción 80/20 que el `NOMODataset` tabular, de modo que las mismas muestras van a test en ambas ramas y la comparación de métricas es justa.

### Normalización de nubes de puntos

Antes de entrar al Transformer, las coordenadas de la nube de puntos se normalizan a media cero y varianza unitaria por eje. Las estadísticas `(mean, std)` se calculan sobre **todo el split de entrenamiento** al inicio del training y se guardan en `internal/experiments/pc_scaler.npz`. Tanto la evaluación como la inferencia cargan este fichero para aplicar la misma normalización. Esto es análogo al `scaler.npz` que el pipeline MLP usa para normalizar las medidas antropométricas.

### El generador Tab Transformer (`src/models/tab_generator.py`)

El generador toma dos entradas: un vector de ruido `z` de 64 dimensiones y una nube de puntos `(B, 256, 3)`. El flujo es:

1. **Embedding de puntos**: cada punto 3D `(x, y, z)` se proyecta a un token de dimensión `d_model = 64` mediante `Linear(3, 64) → LayerNorm → LeakyReLU(0.2)`. Esto produce una secuencia de 256 tokens de dimensión 64.
2. **Inyección de ruido**: el vector `z` se proyecta con `Linear(64, 256 × 64) → LayerNorm → LeakyReLU`, se remodela a `(B, 256, 64)` y se **suma** a los tokens de punto. La suma (en lugar de concatenación) mantiene la dimensión constante y actúa como una perturbación estocástica sobre la representación geométrica.
3. **Encoding posicional**: se suma un tensor aprendible `pos_embed` de forma `(1, 256, 64)`, inicializado con `N(0, 0.02)`.
4. **Transformer Encoder**: 3 capas de `TransformerEncoderLayer` con 4 cabezas de atención, feedforward de 128 unidades, activación GELU, dropout 0.1, y normalización pre-layer (`norm_first=True`). Cada capa aplica self-attention sobre los 256 tokens, permitiendo que cada punto "vea" la geometría global del cuerpo.
5. **Cabeza de salida**: mean-pooling sobre los 256 tokens → `LayerNorm(64) → Linear(64, 10)`, produciendo los 10 betas SMPL.

La inicialización de pesos usa Kaiming normal con pendiente 0.2 (consistente con LeakyReLU). El total de parámetros es del orden de ~150K, mucho menor que el generador MLP debido a la reutilización de pesos en las capas de atención.

### El discriminador Tab Transformer (`src/models/tab_discriminator.py`)

El discriminador recibe los betas generados (o reales) junto con la nube de puntos condicionante, y produce un escalar (sin sigmoid, por ser WGAN):

1. **Embedding de betas**: cada uno de los 10 betas se trata como un token independiente, con su propio `Linear(1, 64) → LayerNorm → LeakyReLU`. Usar embeddings **separados por posición** (un `ModuleList` de 10 capas) es el rasgo distintivo del Tab Transformer frente a un embedding compartido: cada "columna" (beta₀, beta₁, …, beta₉) tiene su propia proyección.
2. **Embedding de puntos**: igual que en el generador, `Linear(3, 64) → LayerNorm → LeakyReLU` compartido para los 256 puntos.
3. **Concatenación**: los 10 tokens de beta + los 256 tokens de punto se concatenan en una secuencia de 266 tokens.
4. **Encoding posicional + Transformer**: mismo esquema que el generador (3 capas, 4 cabezas, GELU, pre-norm).
5. **Cabeza de salida**: mean-pool → `LayerNorm(64) → Linear(64, 1)`.

El discriminador no usa BatchNorm en ningún punto — por las mismas razones que en la rama de imagen: el gradient penalty de WGAN-GP requiere independencia entre muestras del batch, y BatchNorm acopla las estadísticas. LayerNorm (incorporada en los bloques Transformer y en los embeddings) normaliza por muestra individual.

### El bucle de entrenamiento (`src/train_tab.py`)

El trainer (`TabWGANGPTrainer`) sigue el patrón WGAN-GP ya establecido en el pipeline tabular y de imagen, con las siguientes particularidades:

- **Gradient penalty condicionado**: la interpolación `alpha` se aplica sólo sobre los betas (no sobre la nube de puntos, que es la condición fija). El discriminador recibe `(interpolated_betas, pc)` y los gradientes se toman respecto a `interpolated_betas` solamente. Esto es coherente con la formulación WGAN-GP: el crítico debe ser Lipschitz respecto a los datos, no respecto a la condición.
- **Normalización on-the-fly**: cada batch de nubes de puntos se normaliza usando `pc_mean` y `pc_std` calculados al inicio del training.
- **Log de muestras**: cada `sample_interval` epochs (100 por defecto), se genera un vector de betas con la primera nube de puntos del training set (`_probe_pc`) y ruido aleatorio, y se escribe en un CSV (`internal/logs/tab_beta_samples.csv`). Esto permite trazar la evolución de los betas generados a lo largo del entrenamiento.
- **Hiperparámetros**: batch size 32 (más pequeño que el MLP, 128, porque el Transformer tiene mayor coste por muestra), 3 000 epochs, `n_critic = 5`, `λ_GP = 10`, Adam con `β₁ = 0.5`, `β₂ = 0.9`.
- **Checkpoints**: prefijo `wgangp_tab_ckpt_*.pt`, separado del MLP (`wgangp_ckpt_*`) y de imagen (`wgangp_img_ckpt_*`), con guardado cada 500 epochs.

### Evaluación (`src/eval_tab.py`)

El script de evaluación compara cuantitativamente los modelos MLP y Tab Transformer usando dos métricas adaptadas al dominio tabular:

1. **FID (Fréchet Inception Distance) tabular**: en lugar de usar features de una red InceptionV3 (que es para imágenes), se calcula la distancia de Fréchet directamente sobre los vectores de 10 betas. Se computan la media y la covarianza de los betas reales (test set) y de los betas generados, y se aplica la fórmula:

   `FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^½)`

   Un FID bajo indica que la distribución generada se parece a la real. Para datos de 10 dimensiones, valores por debajo de ~10 son buenos; por encima de 50 indican una discrepancia significativa.

2. **Inception Score (IS) tabular**: como no hay un clasificador pre-entrenado para betas, se entrena uno ad hoc. Se agrupan los betas reales en K clusters con K-Means (K adaptativo: `min(10, max(2, N_test / 5))`), se entrena un MLP pequeño (`64 → 32 → K`) para clasificar cada beta en su cluster, y se aplica la fórmula estándar de IS:

   `IS = exp( E_x[ KL( p(y|x) ‖ p(y) ) ] )`

   Un IS alto indica que los betas generados son diversos (se reparten entre muchos clusters) y confiados (cada muestra se asigna claramente a un cluster). El K-Means se implementa a mano con numpy (sin sklearn) para evitar segfaults observados con la versión de scikit-learn en el entorno de desarrollo.

El script carga los checkpoints de ambos modelos (MLP y Tab Transformer) si existen, genera betas para todo el test set, y presenta una tabla comparativa.

### Inferencia (`src/infer_tab.py`)

La inferencia del Tab Transformer tiene un flujo distinto al del MLP porque necesita fabricar una nube de puntos de entrada. El proceso es:

1. Se genera un **cuerpo SMPL plantilla** con `betas = 0` (la forma media) usando el modelo SMPL del género especificado.
2. Se submuestrean los mismos 256 vértices (misma semilla y mismos índices que en entrenamiento).
3. Se normaliza la nube con el `pc_scaler.npz` guardado durante el training.
4. Se generan N vectores de ruido (32 por defecto), se pasa cada uno junto con la nube a través del generador, y se **promedian los betas resultantes**. Este promedio sobre múltiples z reduce la varianza del ruido latente y produce una estimación más estable.
5. Con los betas generados se recomputan los vértices SMPL y se exporta el resultado como imagen PNG (render ortográfico 2D) o como malla OBJ 3D, o ambos.

El renderizado 2D (`src/render_2d.py`) proyecta los vértices ortográficamente, pinta las caras de la malla ordenadas por profundidad (painter's algorithm) con supersampleo ×2, y soporta vistas frontal, lateral y trasera.

### Integración en `main.py`

El Tab Transformer añade tres subcomandos a la CLI:

```bash
python main.py train_tab                        # Entrenar WGAN-GP Tab Transformer
python main.py eval_tab                         # Evaluar FID e IS de todos los modelos
python main.py infer_tab --gender FEMALE --show  # Generar cuerpo y visualizar
```

Los subcomandos coexisten con los tabulares (`fit`, `train`, `eval`, `infer`) y los de imagen (`train_img`, `infer_img`) sin interferencia, ya que cada pipeline usa su propio prefijo de checkpoint y sus propios directorios de caché.

### Resultados obtenidos

Tras entrenar el Tab Transformer durante 3 000 epochs con WGAN-GP (checkpoint `wgangp_tab_ckpt_2999.pt`), los resultados de evaluación son:

| Modelo          |    FID    |    IS     |
|:----------------|:---------:|:---------:|
| TabTransformer  |  54.3054  |  1.5367   |

El **FID de 54.31** indica que la distribución de betas generada por el Tab Transformer difiere significativamente de la distribución real. Para un espacio de 10 dimensiones, este valor es alto — un generador que reprodujera fielmente la distribución esperaría un FID por debajo de 10. El **IS de 1.54** (sobre un máximo teórico de K = número de clusters) confirma que las muestras generadas tienen poca diversidad y/o se concentran en unos pocos clusters, lo que sugiere colapso de modos parcial.

Estos resultados son coherentes con la hipótesis de partida: el Tab Transformer **no es la arquitectura adecuada para datos geométricos 3D**. Tratar cada punto como un token tabular ignora la estructura espacial del cuerpo (no hay noción de vecindad, no hay invarianza a rotación ni escala), y el mecanismo de atención debe "redescubrir" relaciones geométricas que una red diseñada para point clouds (como PointNet o DGCNN) codificaría de forma nativa. El resultado sirve como **línea base negativa** para justificar que la representación de la condición importa: el MLP condicionado por medidas antropométricas, a pesar de usar una arquitectura más simple, trabaja sobre una representación más informativa y condensada del cuerpo.

### Lo que deliberadamente no hace este diseño

- **No usa PointNet ni DGCNN**: el objetivo del experimento era mostrar la limitación del Tab Transformer sobre datos geométricos, no competir con arquitecturas especializadas en point clouds.
- **No condiciona por medidas**: a diferencia del MLP, el Tab Transformer recibe la nube de puntos directamente. Si se quisiera combinar ambas condiciones, bastaría con añadir los tokens de medidas a la secuencia de entrada del discriminador y del generador.
- **No aplica data augmentation geométrica** (rotaciones, jitter): podría mejorar la robustez, pero oscurecería la comparación con el MLP que tampoco la aplica.
