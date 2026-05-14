"""
Builder for notebooks/proyecto_gan.ipynb.

Generates a Colab-ready notebook that:
  - Mounts Google Drive and finds the project root automatically
  - Installs dependencies (CUDA torch + smplx + trimesh + chumpy + pytorch-fid)
  - Converts SMPL .pkl (chumpy -> numpy)
  - Initializes Paths and runs the full pipeline:
    fit -> train -> eval -> infer -> eval-mmd -> render-skeletons
    -> train-image -> eval-image -> train-mesh -> eval-mesh
  - Visualizes intermediate outputs

Run from project root:
    python notebooks/build_notebook.py
"""

import json
from pathlib import Path


def md(*lines: str) -> dict:
    src = "\n".join(lines)
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(*lines: str) -> dict:
    src = "\n".join(lines)
    return {
        "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
        "source": src.splitlines(keepends=True),
    }


cells = []

# ====================================================================
# Title
# ====================================================================
cells.append(md(
    "# Proyecto GAN — Síntesis de Cuerpo Humano",
    "",
    "**Autores:** Santiago Álvarez Geanta, Alejandro García Belmonte, Guillermo García Blázquez, Taron Sargsyan  ",
    "**Asignatura:** Redes Neuronales · Universidad de Alicante",
    "",
    "---",
    "",
    "## Índice",
    "",
    "0. Preparación del entorno (Colab + Drive + GPU + dependencias)",
    "1. Introducción — ¿qué es una GAN y qué hace este proyecto?",
    "2. Datasets — NOMO3D, TNT15, SMPL",
    "3. Pipeline tabular: medidas → betas → malla",
    "   - 3.1 `fit_betas`: preparación de targets pseudo-GT",
    "   - 3.2 Entrenar el WGAN-GP condicional",
    "   - 3.3 Evaluar (MAE + MMD)",
    "   - 3.4 Inferencia: generar un cuerpo a partir de medidas",
    "4. Image GAN — DCGAN 128×128 sobre TNT15",
    "5. Mesh GAN — PointNet WGAN-GP sobre nubes 6890 pts",
    "6. Métricas consolidadas",
    "7. Conclusiones",
    "",
    "> Este notebook está pensado para que alguien sin conocimientos profundos pueda seguirlo. Cada sección explica primero **qué** estamos haciendo y **por qué**, y después muestra el código que lo ejecuta.",
))

# ====================================================================
# 0. Setup
# ====================================================================
cells.append(md(
    "---",
    "## 0. Preparación del entorno",
    "",
    "Si estás en **Google Colab**, esta sección monta tu Drive, encuentra la carpeta del proyecto, instala dependencias y prepara los modelos SMPL. Si estás en local, las celdas detectan que no hay Drive y se saltan ese paso.",
))

cells.append(code(
    "# 0.1 Montar Google Drive (solo si estamos en Colab)",
    "import os, sys",
    "IS_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')",
    "if IS_COLAB:",
    "    from google.colab import drive",
    "    drive.mount('/content/drive')",
    "    print('Drive montado en /content/drive')",
    "else:",
    "    print('Entorno local — sin Drive')",
))

cells.append(code(
    "# 0.2 Localizar la raíz del proyecto",
    "from pathlib import Path",
    "",
    "candidates = [",
    "    Path('/content/drive/MyDrive/Colab Notebooks/GAN/GAN'),",
    "    Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd(),",
    "    Path.cwd(),",
    "]",
    "PROJECT_ROOT = next((p for p in candidates if (p / 'main.py').exists() and (p / 'src').exists()), None)",
    "",
    "if PROJECT_ROOT is None:",
    "    import subprocess",
    "    if IS_COLAB:",
    "        out = subprocess.check_output(",
    "            ['find', '/content/drive/MyDrive', '-name', 'main.py', '-path', '*/GAN/*'],",
    "            text=True,",
    "        ).strip().splitlines()",
    "        PROJECT_ROOT = Path(out[0]).parent if out else None",
    "",
    "assert PROJECT_ROOT is not None, 'No se encontró la raíz del proyecto'",
    "os.chdir(PROJECT_ROOT)",
    "if str(PROJECT_ROOT) not in sys.path:",
    "    sys.path.insert(0, str(PROJECT_ROOT))",
    "print(f'PROJECT_ROOT = {PROJECT_ROOT}')",
))

cells.append(code(
    "# 0.3 Instalar dependencias (~3-5 min en Colab la primera vez)",
    "import subprocess, sys",
    "",
    "def pip(args):",
    "    return subprocess.run([sys.executable, '-m', 'pip', 'install', *args], check=False)",
    "",
    "# Torch ya viene en Colab con CUDA; en local se asume que está instalado",
    "pip(['--quiet', 'smplx==0.1.28', 'trimesh==4.12.1', 'scipy', 'tqdm',",
    "     'pandas', 'plotly', 'seaborn', 'matplotlib', 'pyglet==1.5.31', 'pytorch-fid'])",
    "# chumpy necesita --no-build-isolation por su setup.py legacy",
    "pip(['--quiet', '--no-build-isolation', 'chumpy'])",
    "print('Deps instaladas')",
))

cells.append(code(
    "# 0.4 Verificar GPU",
    "import torch",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "if device.type == 'cuda':",
    "    print(f'GPU disponible: {torch.cuda.get_device_name(0)}')",
    "    print(f'Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')",
    "else:",
    "    print('Sin GPU — entrenamiento será MUY lento. En Colab: Runtime → Change runtime type → GPU')",
))

cells.append(code(
    "# 0.5 Inicializar Paths (crea internal/data, internal/experiments, copia SMPL .pkl, etc.)",
    "from src.config.paths import Paths",
    "Paths.init_project()",
    "print(f'SMPL_DIR:           {Paths.SMPL_DIR}')",
    "print(f'NOMO3D_FEMALE_OBJ:  {Paths.NOMO3D_FEMALE_OBJ}')",
    "print(f'NOMO3D_FEMALE_MEAS: {Paths.NOMO3D_FEMALE_MEAS}')",
    "print(f'TNT15_IMAGES_DIR:   {Paths.TNT15_IMAGES_DIR}')",
    "print(f'EXPERIMENTS_DIR:    {Paths.EXPERIMENTS_DIR}')",
))

cells.append(code(
    "# 0.6 Convertir los .pkl de SMPL (chumpy → numpy puro)",
    "# Los .pkl originales contienen objetos chumpy.Ch incompatibles con numpy 2.x.",
    "# convert_smpl.py los re-empaqueta como arrays numpy normales.",
    "import subprocess",
    "for g in ['FEMALE', 'MALE', 'NEUTRAL']:",
    "    pkl = Paths.SMPL_DIR / f'SMPL_{g}.pkl'",
    "    if pkl.exists():",
    "        # Solo convertimos si aún tiene Ch objects (heurística rápida)",
    "        try:",
    "            import pickle",
    "            with open(pkl, 'rb') as f:",
    "                d = pickle.load(f, encoding='latin1')",
    "            needs_conv = any('chumpy' in str(type(v)) for v in d.values())",
    "        except Exception:",
    "            needs_conv = True",
    "        if needs_conv:",
    "            subprocess.run([sys.executable, 'convert_smpl.py', str(pkl), str(pkl)], check=True)",
    "            print(f'Convertido: {pkl.name}')",
    "        else:",
    "            print(f'Ya limpio:  {pkl.name}')",
))

# ====================================================================
# 1. Introducción
# ====================================================================
cells.append(md(
    "---",
    "## 1. Introducción — ¿qué es una GAN?",
    "",
    "Una **GAN** (*Generative Adversarial Network*) es una pareja de redes neuronales que aprenden compitiendo entre sí:",
    "",
    "- El **Generador (G)** parte de ruido aleatorio e intenta crear datos que parezcan reales.",
    "- El **Discriminador (D)** recibe datos reales y generados, e intenta distinguirlos.",
    "",
    "Tras muchas iteraciones, G aprende a generar ejemplos tan realistas que D ya no puede distinguirlos.",
    "",
    "### ¿Qué hace este proyecto?",
    "",
    "Entrenamos **tres GANs** que abordan el problema de generar cuerpos humanos plausibles desde tres ángulos:",
    "",
    "| Componente | Pregunta que responde | Entrada → Salida |",
    "|---|---|---|",
    "| **Tabular** | ¿Qué forma tiene un cuerpo con estas medidas? | 10 medidas (cm) → 10 betas SMPL |",
    "| **Imagen** | ¿Cómo se ve una silueta humana plausible? | ruido → imagen 128×128 |",
    "| **Mesh 3D** | ¿Qué forma tridimensional tiene un cuerpo? | ruido → 6890 puntos 3D |",
    "",
    "Las tres usan la variante **WGAN-GP** (Wasserstein GAN con Gradient Penalty) — mucho más estable que la GAN clásica.",
))

# ====================================================================
# 2. Datasets
# ====================================================================
cells.append(md(
    "---",
    "## 2. Datasets",
    "",
    "### 2.1 NOMO3D — la base de todo",
    "356 escaneos 3D de personas reales (177 mujeres + 179 hombres). De cada persona tenemos:",
    "- Un OBJ con la malla del escaneo (~57k vértices)",
    "- Un .txt con sus medidas antropométricas (altura, busto, cintura, etc.)",
    "",
    "Lo usamos para el Tabular GAN (medidas → betas) y para el Mesh GAN (nubes de puntos).",
    "",
    "### 2.2 TNT15 — siluetas para el Image GAN",
    "~25.000 PNGs de personas segmentadas (silueta blanca sobre negro), 128×128 tras resize.",
    "",
    "### 2.3 SMPL — el modelo paramétrico de cuerpo",
    "Un modelo matemático que, dados 10 betas (forma) y 24 poses (articulaciones), produce una malla 3D de 6890 vértices. Lo usamos como **puente** entre las modalidades: betas → malla → joints → render.",
))

cells.append(code(
    "# Inventario rápido de los datasets",
    "print(f'NOMO3D female OBJs: {len(list(Paths.NOMO3D_FEMALE_OBJ.glob(\"*.obj\")))}')",
    "print(f'NOMO3D male   OBJs: {len(list(Paths.NOMO3D_MALE_OBJ.glob(\"*.obj\")))}')",
    "print(f'NOMO3D female meas: {len(list(Paths.NOMO3D_FEMALE_MEAS.glob(\"*.txt\")))}')",
    "print(f'NOMO3D male   meas: {len(list(Paths.NOMO3D_MALE_MEAS.glob(\"*.txt\")))}')",
    "print(f'TNT15 PNGs (recursivo): {sum(1 for _ in Paths.TNT15_IMAGES_DIR.rglob(\"*.png\"))}')",
    "print(f'SMPL .pkl: {[p.name for p in Paths.SMPL_DIR.glob(\"*.pkl\")]}')",
))

# ====================================================================
# 3. Tabular GAN
# ====================================================================
cells.append(md(
    "---",
    "## 3. Pipeline tabular: medidas → betas → malla",
    "",
    "Esta es la **pieza central** del proyecto. Aprende a mapear medidas antropométricas (las que te toman en una sastrería: altura, busto, cintura, cadera, cuello, hombro, inseam, outseam, muslo, bíceps) a los **10 parámetros de forma** (`betas`) del modelo SMPL.",
    "",
    "Una vez tenemos los betas, hacer un *forward pass* de SMPL nos da la **malla 3D completa**.",
    "",
    "### Arquitectura (condicional WGAN-GP)",
    "",
    "```",
    "Generador:    G(z, medidas) → 10 betas",
    "Discriminador: D(betas, medidas) → score WGAN",
    "```",
    "",
    "Tanto G como D son MLPs con LayerNorm y LeakyReLU. G incluye un bloque residual en el dim pico (1024). D no usa BatchNorm (incompatible con gradient penalty).",
    "",
    "### Hiperparámetros (de `src/config/hparams.py`)",
    "- `noise_dim=64`, `cond_dim=10` (medidas), `num_betas=10`",
    "- Generador: `[256, 512, 1024, 512, 256]`",
    "- Discriminador: `[256, 128, 64]`",
    "- Batch=128, lr=1e-4, Adam β=(0.5, 0.9), n_critic=5, λ_gp=10",
    "- Epochs=2000 (recomendado en GPU; reducir si solo CPU)",
))

cells.append(md(
    "### 3.1 `fit_betas` — preparar los targets",
    "",
    "Aquí está el truco crítico: NOMO3D nos da las medidas, pero **no** los betas SMPL que las producen. Tenemos que invertir esa relación.",
    "",
    "Para cada sample NOMO3D resolvemos el problema:",
    "$$",
    "\\beta^* = \\arg\\min_\\beta \\sum_i w_i \\, (\\text{medida}_i^{SMPL}(\\beta) - \\text{medida}_i^{NOMO3D})^2",
    "$$",
    "",
    "Usamos **scipy Nelder-Mead** porque la extracción de medidas de SMPL (cortes de malla + convex hull para las circunferencias) no es diferenciable.",
    "",
    "Los betas resultantes se cachean en `internal/data/betas_cache/<id>.npy` y se usan como targets de entrenamiento del WGAN-GP. Si el cache ya existe, esta celda es instantánea.",
    "",
    "> **Tiempo:** ~3 horas para los 356 samples (mismo en CPU y GPU — el cuello es scipy y el mesh slicing).",
))

cells.append(code(
    "# 3.1 Ejecutar fit_betas (idempotente — saltea samples ya cacheados)",
    "from src.data.beta_fitter import fit_betas",
    "fit_betas()",
))

cells.append(code(
    "# Inspección rápida de los betas fitteados",
    "import numpy as np",
    "cache_files = sorted(Paths.BETAS_CACHE_DIR.glob('*.npy'))",
    "print(f'Betas cacheados: {len(cache_files)} / 356')",
    "if cache_files:",
    "    betas = np.stack([np.load(f) for f in cache_files])",
    "    print(f'  shape: {betas.shape}')",
    "    print(f'  rango: [{betas.min():.2f}, {betas.max():.2f}]')",
    "    print(f'  media: {betas.mean(0).round(3)}')",
    "    print(f'  std:   {betas.std(0).round(3)}')",
))

cells.append(md(
    "### 3.2 Entrenar el WGAN-GP",
    "",
    "Bucle clásico WGAN-GP:",
    "1. Para cada batch real `(medidas, betas_GT)`:",
    "2. **n_critic=5 pasos de D:**",
    "   - Genera `betas_fake = G(z, medidas)`",
    "   - Calcula `L_D = E[D(fake)] - E[D(real)] + λ·GP`",
    "   - Backprop + paso de Adam",
    "3. **1 paso de G:**",
    "   - Calcula `L_G = -E[D(G(z, medidas))]`",
    "",
    "Cada 100 epochs se loguea una muestra de betas con una medida \"probe\" fija para ver la evolución.",
    "",
    "> **Tiempo:** 2000 epochs ≈ horas en GPU, días en CPU.",
))

cells.append(code(
    "# 3.2 Entrenar el WGAN-GP condicional",
    "from src.train import WGANGPTrainer",
    "trainer = WGANGPTrainer()",
    "# Si quieres un entrenamiento más corto para test, descomenta:",
    "# trainer.hp.epochs = 200",
    "trainer.train()",
))

cells.append(code(
    "# Visualizar la evolución del probe sample",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "csv_path = Paths.LOGS_DIR / 'beta_samples.csv'",
    "if csv_path.exists():",
    "    df = pd.read_csv(csv_path)",
    "    fig, ax = plt.subplots(figsize=(10, 4))",
    "    for col in [c for c in df.columns if c.startswith('beta_')]:",
    "        ax.plot(df['epoch'], df[col], label=col, alpha=0.6)",
    "    ax.set_xlabel('Epoch'); ax.set_ylabel('Probe beta value')",
    "    ax.set_title('Evolución del probe-sample (medida fija → betas a lo largo del entrenamiento)')",
    "    ax.legend(ncol=2, fontsize=8)",
    "    plt.tight_layout(); plt.show()",
    "else:",
    "    print('Aún no hay log de muestras (entrena al menos sample_interval epochs)')",
))

cells.append(md(
    "### 3.3 Evaluar — MAE + MMD",
    "",
    "**MAE de medidas:** para cada sample del test, genera betas → forward SMPL → extrae medidas → compara con las medidas reales. Reportamos error medio por medida.",
    "",
    "**MMD (Maximum Mean Discrepancy):** mide la distancia entre la distribución de betas reales y la de betas generados. Menor = mejor. Usamos kernel RBF multi-bandwidth.",
))

cells.append(code(
    "# 3.3a Evaluación MAE",
    "from src.eval import evaluate",
    "evaluate()",
))

cells.append(code(
    "# 3.3b Evaluación MMD",
    "from src.eval_tabular import evaluate_mmd",
    "evaluate_mmd(n_samples=500)",
))

cells.append(md(
    "### 3.4 Inferencia — generar un cuerpo concreto",
    "",
    "Damos medidas → obtenemos malla SMPL. La inferencia es instantánea (un único forward pass).",
))

cells.append(code(
    "# 3.4 Inferir y visualizar la malla generada",
    "import argparse, torch, trimesh",
    "from src.inference import infer",
    "from src.config.paths import Paths",
    "",
    "args = argparse.Namespace(",
    "    gender='FEMALE', height=170.0, bust=90.0, waist=70.0, hip=95.0,",
    "    neck=34.0, shoulder=40.0, inseam=80.0, outseam=100.0, thigh=55.0, bicep=28.0,",
    "    show=False,",
    ")",
    "infer(args)",
    "",
    "# Visualización del OBJ generado con trimesh + matplotlib",
    "obj_path = Paths.TEMP_DIR / f'generated_{args.gender}_{args.height}.obj'",
    "mesh = trimesh.load(str(obj_path), process=False)",
    "verts = mesh.vertices",
    "print(f'Malla generada: {len(verts)} verts, {len(mesh.faces)} caras')",
    "",
    "import matplotlib.pyplot as plt",
    "fig = plt.figure(figsize=(6, 8))",
    "ax = fig.add_subplot(111, projection='3d')",
    "idx = np.random.choice(len(verts), 2000, replace=False)",
    "ax.scatter(verts[idx, 0], verts[idx, 1], verts[idx, 2], s=1, c='steelblue')",
    "ax.set_title(f'Malla generada para {args.gender} h={args.height}cm')",
    "ax.set_box_aspect((1,1,2))",
    "plt.tight_layout(); plt.show()",
))

# ====================================================================
# 4. Image GAN
# ====================================================================
cells.append(md(
    "---",
    "## 4. Image GAN — DCGAN 128×128",
    "",
    "Generamos siluetas humanas 128×128 entrenando sobre TNT15. Opcionalmente complementamos el dataset con renders de esqueleto producidos desde los betas cacheados.",
    "",
    "### Arquitectura",
    "",
    "- **Generador:** FC(z) → reshape (8·base, 4, 4) → 5 ConvTranspose2d stride 2 + LayerNorm + ReLU → Tanh",
    "- **Discriminador:** 4 Conv2d stride 2 + LayerNorm + LeakyReLU → FC → 1",
    "",
    "Sin BatchNorm (incompatible con gradient penalty).",
))

cells.append(code(
    "# 4.1 Renderizar esqueletos desde los betas cacheados (opcional pero útil)",
    "from src.data.render_skeleton import render_batch_from_betas_cache",
    "n = render_batch_from_betas_cache()",
    "print(f'Renderizados {n} esqueletos en {Paths.SKELETON_RENDERS_DIR}')",
))

cells.append(code(
    "# 4.2 Ver algunas muestras del dataset y de los renders",
    "import matplotlib.pyplot as plt",
    "from PIL import Image",
    "import random",
    "",
    "fig, axes = plt.subplots(2, 6, figsize=(14, 5))",
    "tnt = list(Paths.TNT15_IMAGES_DIR.rglob('*.png'))",
    "rend = list(Paths.SKELETON_RENDERS_DIR.glob('*.png'))",
    "for ax, p in zip(axes[0], random.sample(tnt, k=min(6, len(tnt)))):",
    "    ax.imshow(Image.open(p).convert('L'), cmap='gray'); ax.axis('off')",
    "axes[0, 0].set_title('TNT15 (siluetas reales)', loc='left')",
    "for ax, p in zip(axes[1], random.sample(rend, k=min(6, len(rend)))):",
    "    ax.imshow(Image.open(p).convert('L'), cmap='gray'); ax.axis('off')",
    "axes[1, 0].set_title('Renders de esqueleto (sintéticos)', loc='left')",
    "plt.tight_layout(); plt.show()",
))

cells.append(code(
    "# 4.3 Entrenar el Image GAN",
    "from src.train_image import ImageWGANGPTrainer",
    "img_trainer = ImageWGANGPTrainer()",
    "# Para test rápido descomenta:",
    "# img_trainer.hp.epochs = 5",
    "img_trainer.train()",
))

cells.append(code(
    "# 4.4 Visualizar muestras generadas",
    "import torch",
    "from src.models.image_gan import ImageGenerator",
    "from src.config.hparams import ImageHParams",
    "",
    "hp = ImageHParams()",
    "G = ImageGenerator(hp).to(device)",
    "ckpts = sorted(Paths.EXPERIMENTS_IMAGE.glob('image_wgangp_*.pt'))",
    "if ckpts:",
    "    G.load_state_dict(torch.load(ckpts[-1], map_location=device)['G_state_dict'])",
    "    G.eval()",
    "    with torch.no_grad():",
    "        z = torch.randn(16, hp.noise_dim, device=device)",
    "        fake = ((G(z).cpu() + 1) / 2).clamp(0, 1)",
    "    fig, axes = plt.subplots(4, 4, figsize=(8, 8))",
    "    for ax, img in zip(axes.flat, fake):",
    "        ax.imshow(img.squeeze(), cmap='gray'); ax.axis('off')",
    "    plt.suptitle('Muestras generadas por el Image GAN'); plt.tight_layout(); plt.show()",
    "else:",
    "    print('Aún no hay checkpoint del Image GAN')",
))

cells.append(code(
    "# 4.5 Métrica FID",
    "from src.eval_image import evaluate_fid",
    "evaluate_fid(n_samples=1000)",
))

# ====================================================================
# 5. Mesh GAN
# ====================================================================
cells.append(md(
    "---",
    "## 5. Mesh GAN — PointNet WGAN-GP",
    "",
    "Genera **nubes de 6890 puntos 3D** que representan la superficie del cuerpo.",
    "",
    "### Arquitectura",
    "- **Generador:** MLP `z → 1024 → 4096 → 6890·3` con LayerNorm.",
    "- **Discriminador (PointNet):** Conv1d per-punto (`3 → 64 → 128 → 256`) → max-pool global → MLP → score.",
    "",
    "PointNet es **invariante al orden** de los puntos (la max-pool ignora el orden), lo que es la propiedad correcta para una nube de puntos.",
    "",
    "### Datos",
    "- 356 escaneos NOMO3D, cada uno submuestreado a 6890 pts vía `trimesh.sample.sample_surface`.",
    "- Se cachean en `internal/data/nomo3d_pointclouds/`. La primera ejecución tarda ~5 min, después es instantánea.",
))

cells.append(code(
    "# 5.1 Visualizar una nube real de NOMO3D",
    "from src.data.nomo3d_obj_loader import NOMO3DPointCloudDataset",
    "ds = NOMO3DPointCloudDataset(num_points=6890)",
    "print(f'Dataset: {len(ds)} nubes')",
    "pc = ds[0].numpy()",
    "fig = plt.figure(figsize=(6, 8))",
    "ax = fig.add_subplot(111, projection='3d')",
    "idx = np.random.choice(6890, 1500, replace=False)",
    "ax.scatter(pc[idx, 0], pc[idx, 1], pc[idx, 2], s=1, c='steelblue')",
    "ax.set_title('Nube real de NOMO3D (1500 pts mostrados)')",
    "ax.set_box_aspect((1,1,2)); plt.tight_layout(); plt.show()",
))

cells.append(code(
    "# 5.2 Entrenar el Mesh GAN",
    "from src.train_mesh import MeshWGANGPTrainer",
    "mesh_trainer = MeshWGANGPTrainer()",
    "# Test rápido: mesh_trainer.hp.epochs = 10",
    "mesh_trainer.train()",
))

cells.append(code(
    "# 5.3 Visualizar muestras generadas",
    "import torch",
    "from src.models.mesh_gan import MeshGenerator",
    "from src.config.hparams import MeshHParams",
    "",
    "hp = MeshHParams()",
    "G = MeshGenerator(hp).to(device)",
    "ckpts = sorted(Paths.EXPERIMENTS_MESH.glob('mesh_wgangp_*.pt'))",
    "if ckpts:",
    "    G.load_state_dict(torch.load(ckpts[-1], map_location=device)['G_state_dict'])",
    "    G.eval()",
    "    with torch.no_grad():",
    "        z = torch.randn(2, hp.noise_dim, device=device)",
    "        fake = G(z).cpu().numpy()",
    "    fig = plt.figure(figsize=(10, 5))",
    "    for i in range(2):",
    "        ax = fig.add_subplot(1, 2, i+1, projection='3d')",
    "        pc = fake[i]",
    "        idx = np.random.choice(len(pc), 1500, replace=False)",
    "        ax.scatter(pc[idx, 0], pc[idx, 1], pc[idx, 2], s=1, c='darkorange')",
    "        ax.set_title(f'Generada #{i}'); ax.set_box_aspect((1,1,2))",
    "    plt.tight_layout(); plt.show()",
    "else:",
    "    print('Aún no hay checkpoint del Mesh GAN')",
))

cells.append(code(
    "# 5.4 Métricas Chamfer + F-score",
    "from src.eval_3d import evaluate_3d",
    "evaluate_3d(n_samples=50, tau=0.01)",
))

# ====================================================================
# 6. Resultados consolidados
# ====================================================================
cells.append(md(
    "---",
    "## 6. Métricas consolidadas",
    "",
    "Reunimos todas las métricas en una tabla.",
))

cells.append(code(
    "# Tabla resumen (re-ejecuta para refrescar)",
    "import pandas as pd",
    "rows = []",
    "try:",
    "    from src.eval_tabular import evaluate_mmd",
    "    rows.append(('Tabular MMD', evaluate_mmd(n_samples=500)))",
    "except Exception as e:",
    "    rows.append(('Tabular MMD', f'ERROR: {e}'))",
    "try:",
    "    from src.eval_image import evaluate_fid",
    "    rows.append(('Image FID', evaluate_fid(n_samples=1000)))",
    "except Exception as e:",
    "    rows.append(('Image FID', f'ERROR: {e}'))",
    "try:",
    "    from src.eval_3d import evaluate_3d",
    "    res = evaluate_3d(n_samples=50)",
    "    rows.append(('Mesh Chamfer (mean)', res['chamfer_mean']))",
    "    rows.append(('Mesh F-score @ 0.01', res['fscore_mean']))",
    "except Exception as e:",
    "    rows.append(('Mesh', f'ERROR: {e}'))",
    "pd.DataFrame(rows, columns=['Métrica', 'Valor'])",
))

# ====================================================================
# 7. Conclusiones
# ====================================================================
cells.append(md(
    "---",
    "## 7. Conclusiones",
    "",
    "### Lo que hemos construido",
    "Un pipeline reproducible que entrena tres GANs distintas sobre datos reales de cuerpos humanos, con:",
    "- Un **WGAN-GP condicional** que aprende `medidas → betas SMPL` y permite generar la malla 3D correspondiente.",
    "- Un **DCGAN** sobre siluetas TNT15 (~25k imágenes) opcionalmente aumentado con renders de esqueleto sintéticos.",
    "- Un **PointNet WGAN-GP** que genera nubes de 6890 puntos comparables con escaneos reales de NOMO3D.",
    "- Métricas estándar: MAE de medidas, MMD para tabular, FID para imagen, Chamfer + F-score para 3D.",
    "",
    "### Limitaciones",
    "- `fit_betas` actualmente usa **scipy Nelder-Mead** porque la extracción de circunferencias SMPL no es diferenciable. Un sustituto autodiff (medidas aproximadas con rings de vértices fijos) aceleraría 10x.",
    "- El Image GAN genera siluetas pero no rostros ni textura — la diversidad viene del propio TNT15.",
    "- El Mesh GAN genera nubes desordenadas (no preservan la topología SMPL). Para usarlas como mallas habría que registrar contra SMPL.",
    "",
    "### Próximos pasos",
    "- Sustituir Nelder-Mead por un fitter autodiff parcial (longitudes diferenciables + circunferencias suaves).",
    "- Condicionar el Image GAN también con medidas, para alinearlo con el tabular.",
    "- Añadir cabeza/textura al Mesh GAN o forzar topología SMPL.",
))

# ====================================================================
# Build notebook
# ====================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "proyecto_gan.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out} ({len(cells)} cells)")
