"""One-shot patch: turn proyecto_gan_original.ipynb into a Colab-runnable notebook.

Replaces the single environment cell (os.chdir + Paths.init_project) with the full
Colab boilerplate (Drive mount, project-root discovery, deps install, GPU check,
Paths init, SMPL .pkl chumpy->numpy conversion). Also updates the section-0
markdown to mention Colab + Drive. Idempotent: re-running detects the patch.
"""
import json
from pathlib import Path

NB = Path(__file__).parent / 'proyecto_gan_original.ipynb'

MARKER = '# 0.1 Montar Google Drive'  # presence => already patched


def code_cell(src: str) -> dict:
    return {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': src.splitlines(keepends=True),
    }


SECTION0_MD = """---
## 0. Preparación del entorno

Si estás en **Google Colab**, esta sección monta tu Drive, encuentra la carpeta del proyecto, instala dependencias, comprueba que hay GPU e inicializa las rutas (`internal/data`, `internal/experiments`, modelos SMPL, etc.). Si estás en local, las celdas detectan que no hay Drive y se saltan ese paso.
"""

CELL_DRIVE = """# 0.1 Montar Google Drive (solo si estamos en Colab)
import os, sys
IS_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive montado en /content/drive')
else:
    print('Entorno local — sin Drive')
"""

CELL_ROOT = """# 0.2 Localizar la raíz del proyecto
from pathlib import Path

candidates = [
    Path('/content/drive/MyDrive/Colab Notebooks/GAN/GAN'),
    Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd(),
    Path.cwd(),
]
PROJECT_ROOT = next((p for p in candidates if (p / 'main.py').exists() and (p / 'src').exists()), None)

if PROJECT_ROOT is None and IS_COLAB:
    import subprocess
    out = subprocess.check_output(
        ['find', '/content/drive/MyDrive', '-name', 'main.py', '-path', '*/GAN/*'],
        text=True,
    ).strip().splitlines()
    PROJECT_ROOT = Path(out[0]).parent if out else None

assert PROJECT_ROOT is not None, 'No se encontró la raíz del proyecto'
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print(f'PROJECT_ROOT = {PROJECT_ROOT}')
"""

CELL_DEPS = """# 0.3 Instalar dependencias (~3-5 min en Colab la primera vez)
import subprocess, sys

def pip(args):
    return subprocess.run([sys.executable, '-m', 'pip', 'install', *args], check=False)

# Torch ya viene en Colab con CUDA; en local se asume que está instalado
pip(['--quiet', 'smplx==0.1.28', 'trimesh==4.12.1', 'scipy', 'tqdm',
     'pandas', 'plotly', 'seaborn', 'matplotlib', 'pyglet==1.5.31', 'pytorch-fid'])
# chumpy necesita --no-build-isolation por su setup.py legacy
pip(['--quiet', '--no-build-isolation', 'chumpy'])
print('Deps instaladas')
"""

CELL_GPU = """# 0.4 Verificar GPU
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f'GPU disponible: {torch.cuda.get_device_name(0)}')
    print(f'Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('Sin GPU — entrenamiento será MUY lento. En Colab: Runtime → Change runtime type → GPU')
"""

CELL_PATHS = """# 0.5 Inicializar Paths (crea internal/data, internal/experiments, etc.)
from src.config.paths import Paths
Paths.init_project()
print(f'PROJECT_ROOT:    {Paths.PROJECT_ROOT}')
print(f'DATA_DIR:        {Paths.DATA_DIR}')
print(f'EXPERIMENTS_DIR: {Paths.EXPERIMENTS_DIR}')
"""

CELL_SMPL = """# 0.6 Convertir los .pkl de SMPL (chumpy → numpy puro), si existen y aún tienen objetos Ch.
# Los .pkl originales son incompatibles con numpy 2.x. convert_smpl.py los re-empaqueta.
import subprocess, pickle
convert_script = Path('convert_smpl.py')
smpl_dir = getattr(Paths, 'SMPL_DIR', None)
if convert_script.exists() and smpl_dir is not None and Path(smpl_dir).exists():
    for g in ['FEMALE', 'MALE', 'NEUTRAL']:
        pkl = Path(smpl_dir) / f'SMPL_{g}.pkl'
        if not pkl.exists():
            continue
        try:
            with open(pkl, 'rb') as f:
                d = pickle.load(f, encoding='latin1')
            needs_conv = any('chumpy' in str(type(v)) for v in d.values())
        except Exception:
            needs_conv = True
        if needs_conv:
            subprocess.run([sys.executable, str(convert_script), str(pkl), str(pkl)], check=True)
            print(f'Convertido: {pkl.name}')
        else:
            print(f'Ya limpio:  {pkl.name}')
else:
    print('convert_smpl.py o SMPL_DIR no disponibles — saltando conversión')
"""


def main() -> None:
    nb = json.loads(NB.read_text(encoding='utf-8'))
    cells = nb['cells']

    # Idempotency check
    for c in cells:
        if MARKER in ''.join(c.get('source', [])):
            print('Notebook already patched — nothing to do.')
            return

    # Find the old environment cell: the code cell containing `os.chdir('..')`
    target_idx = None
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code' and "os.chdir('..')" in ''.join(c['source']):
            target_idx = i
            break

    if target_idx is None:
        raise SystemExit('Could not locate the original os.chdir cell; aborting.')

    # The markdown header right before it (section 0 intro). Replace it with our updated version.
    if target_idx > 0 and cells[target_idx - 1]['cell_type'] == 'markdown' \
            and '0. Preparación del entorno' in ''.join(cells[target_idx - 1]['source']):
        md_idx = target_idx - 1
    else:
        md_idx = None

    new_cells = [
        md_cell(SECTION0_MD),
        code_cell(CELL_DRIVE),
        code_cell(CELL_ROOT),
        code_cell(CELL_DEPS),
        code_cell(CELL_GPU),
        code_cell(CELL_PATHS),
        code_cell(CELL_SMPL),
    ]

    if md_idx is not None:
        # Replace markdown + old code cell with the new block
        cells[md_idx:target_idx + 1] = new_cells
    else:
        # Just replace the old code cell, leaving any prior markdown intact
        cells[target_idx:target_idx + 1] = new_cells

    # Ensure colab metadata so the notebook opens cleanly in Colab
    nb.setdefault('metadata', {})
    nb['metadata']['colab'] = {'provenance': [], 'toc_visible': True}
    nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'Patched: {NB} ({len(nb["cells"])} cells total)')


if __name__ == '__main__':
    main()
