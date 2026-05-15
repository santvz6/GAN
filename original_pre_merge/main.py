"""
Punto de entrada del proyecto GAN.

Orquesta el pipeline completo por fases. Cada fase puede ejecutarse de forma
independiente mediante flags de línea de comandos.

Uso:
    python main.py --all                      # pipeline completo
    python main.py --preprocess               # solo extraer joints de AMASS
    python main.py --train-tabular            # solo entrenar Tabular GAN
    python main.py --filter --n 1000          # filtrar cuerpos con score_D > 0.8
    python main.py --gen-meshes --n 500       # generar mallas SMPL sintéticas
    python main.py --render --n 500           # renderizar esqueletos a PNG
    python main.py --train-image              # solo entrenar Image GAN
    python main.py --train-mesh               # solo entrenar Mesh GAN
    python main.py --eval                     # correr las 3 evaluaciones
"""

import argparse
import sys

from src.config.paths import Paths


def _run_preprocess():
    from src.data.preprocess_joints import preprocess_and_save
    preprocess_and_save()


def _run_train_tabular():
    from src.training.train_tabular import train_tabular
    train_tabular()


def _run_filter(n: int, threshold: float):
    from src.data.discriminator_filter import filter_generated_bodies
    filter_generated_bodies(n_target=n, threshold=threshold)


def _run_gen_meshes(n: int):
    from src.data.smpl_mesh_generator import generate_and_save_meshes
    generate_and_save_meshes(n_meshes=n)


def _run_render(n: int):
    import numpy as np
    from src.data.render_skeleton import render_joints_batch

    if Paths.GENERATED_JOINTS_NPZ.exists():
        data = np.load(str(Paths.GENERATED_JOINTS_NPZ))
        joints = data['joints'][:n]
        print(f"[RENDER] Usando {len(joints)} cuerpos filtrados.")
    else:
        print("[RENDER] No hay cuerpos filtrados. Generando ruido como placeholder.")
        np.random.seed(42)
        joints = np.random.randn(n, 72).astype('float32')

    render_joints_batch(joints, prefix='skeleton')


def _run_train_image():
    from src.training.train_image import train_image
    train_image()


def _run_train_mesh():
    from src.training.train_mesh import train_mesh
    train_mesh()


def _run_eval():
    results = {}
    try:
        from src.evaluation.eval_tabular import evaluate_tabular
        results['tabular'] = evaluate_tabular()
    except Exception as e:
        print(f"[EVAL] Tabular falló: {e}")

    try:
        from src.evaluation.eval_image import evaluate_image
        results['image_fid'] = evaluate_image()
    except Exception as e:
        print(f"[EVAL] Image falló: {e}")

    try:
        from src.evaluation.eval_3d import evaluate_3d
        results['3d'] = evaluate_3d()
    except Exception as e:
        print(f"[EVAL] 3D falló: {e}")

    print("\n[EVAL] Resumen final:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline GAN cuerpos humanos")
    p.add_argument('--all', action='store_true', help='Ejecutar pipeline completo')
    p.add_argument('--preprocess', action='store_true')
    p.add_argument('--train-tabular', action='store_true')
    p.add_argument('--filter', action='store_true')
    p.add_argument('--gen-meshes', action='store_true')
    p.add_argument('--render', action='store_true')
    p.add_argument('--train-image', action='store_true')
    p.add_argument('--train-mesh', action='store_true')
    p.add_argument('--eval', action='store_true')
    p.add_argument('--n', type=int, default=1000, help='Cantidad para --filter/--render/--gen-meshes')
    p.add_argument('--threshold', type=float, default=0.8, help='Umbral score_D para --filter')
    return p.parse_args()


def main():
    args = parse_args()
    Paths.init_project()
    print(f"[MAIN] Proyecto inicializado. Raíz: {Paths.ROOT}")

    # Si no se pasa ningún flag, mostrar ayuda
    any_flag = any([
        args.all, args.preprocess, args.train_tabular, args.filter,
        args.gen_meshes, args.render, args.train_image, args.train_mesh, args.eval,
    ])
    if not any_flag:
        print("\nNo se pasó ningún flag. Usa --all o alguno de los --flags individuales.")
        print("Ejecuta:  python main.py --help  para ver las opciones.")
        return

    if args.all or args.preprocess:
        _run_preprocess()
    if args.all or args.train_tabular:
        _run_train_tabular()
    if args.all or args.filter:
        _run_filter(args.n, args.threshold)
    if args.all or args.gen_meshes:
        _run_gen_meshes(args.n)
    if args.all or args.render:
        _run_render(args.n)
    if args.all or args.train_image:
        _run_train_image()
    if args.all or args.train_mesh:
        _run_train_mesh()
    if args.all or args.eval:
        _run_eval()


if __name__ == "__main__":
    sys.exit(main() or 0)
