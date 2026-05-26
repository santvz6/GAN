import inspect
import numpy as np
import warnings
import builtins

# Guardamos la función print original
_original_print = builtins.print

# Creamos un print personalizado que filtra el aviso de SMPL
def _custom_print(*args, **kwargs):
    message = " ".join(map(str, args))
    if "10 shape coefficients" in message:
        return  # Ignora el print si contiene el texto del warning
    _original_print(*args, **kwargs)

# Sobrescribimos el print global
builtins.print = _custom_print

# Silenciar advertencias de compatibilidad y parches
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Parches de compatibilidad para chumpy en Python 3.11+ y NumPy moderno
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# chumpy busca estos atributos eliminados en numpy
for name in ['bool', 'int', 'float', 'complex', 'object', 'unicode', 'str']:
    if not hasattr(np, name):
        setattr(np, name, getattr(__builtins__, name) if hasattr(__builtins__, name) else None)

import argparse
from src.config.paths import Paths
from src.data.beta_fitter import fit_betas
from src.train import WGANGPTrainer
from src.eval import evaluate
from src.inference import infer
from src.train_img import ImgWGANGPTrainer
from src.inference_img import infer as infer_img
from src.train_tab import TabWGANGPTrainer
from src.infer_tab import infer as infer_tab
from src.eval_tab import evaluate_tab

def main():
    Paths.init_project()

    parser = argparse.ArgumentParser(description="GAN 3D Body Generation Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- Tabular pipeline ---
    subparsers.add_parser("fit", help="Fit pseudo-ground-truth betas for dataset")
    subparsers.add_parser("train", help="Train the tabular WGAN-GP model")
    subparsers.add_parser("eval", help="Evaluate the tabular model (MAE)")

    parser_infer = subparsers.add_parser(
        "infer",
        help="Run tabular inference and generate a 2D image or 3D mesh",
    )
    parser_infer.add_argument("--gender", type=str, default="FEMALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser_infer.add_argument("--height", type=float, default=170.0)
    parser_infer.add_argument("--bust", type=float, default=90.0)
    parser_infer.add_argument("--waist", type=float, default=70.0)
    parser_infer.add_argument("--hip", type=float, default=95.0)
    parser_infer.add_argument("--neck", type=float, default=34.0)
    parser_infer.add_argument("--shoulder", type=float, default=40.0)
    parser_infer.add_argument("--inseam", type=float, default=80.0)
    parser_infer.add_argument("--outseam", type=float, default=100.0)
    parser_infer.add_argument("--thigh", type=float, default=55.0)
    parser_infer.add_argument("--bicep",     type=float, default=28.0)
    parser_infer.add_argument("--n_samples", type=int,   default=32,
                              help="z samples to average at inference")
    parser_infer.add_argument("--output", type=str, default="image",
                              choices=["image", "mesh", "both"],
                              help="Output format: 2D PNG image, 3D OBJ mesh, or both")
    parser_infer.add_argument("--view", type=str, default="front",
                              choices=["front", "side", "back"],
                              help="Camera view used for the 2D PNG render")
    parser_infer.add_argument("--image_size", type=int, default=512,
                              help="Output PNG size in pixels")
    parser_infer.add_argument("--wireframe", action="store_true",
                              help="Draw triangle edges in the 2D PNG render")
    parser_infer.add_argument("--show", action="store_true", help="Show 3D viewer")

    # --- Image pipeline (TNT15 WGAN-GP) ---
    subparsers.add_parser("train_img", help="Train the image WGAN-GP on TNT15")

    parser_infer_img = subparsers.add_parser("infer_img", help="Generate images from the trained image GAN")
    parser_infer_img.add_argument("-n", type=int, default=16, help="Number of images to generate.")
    parser_infer_img.add_argument("--grid", action="store_true",
                                  help="Save as a single grid PNG instead of N separate files.")

    # --- Tab Transformer pipeline ---
    subparsers.add_parser("train_tab", help="Train the Tab Transformer WGAN-GP model")
    subparsers.add_parser("eval_tab", help="Evaluate tabular models (FID & Inception Score)")

    parser_infer_tab = subparsers.add_parser(
        "infer_tab",
        help="Run Tab Transformer inference and generate a 2D image or 3D mesh",
    )
    parser_infer_tab.add_argument("--gender", type=str, default="FEMALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser_infer_tab.add_argument("--height", type=float, default=170.0)
    parser_infer_tab.add_argument("--bust", type=float, default=90.0)
    parser_infer_tab.add_argument("--waist", type=float, default=70.0)
    parser_infer_tab.add_argument("--hip", type=float, default=95.0)
    parser_infer_tab.add_argument("--neck", type=float, default=34.0)
    parser_infer_tab.add_argument("--shoulder", type=float, default=40.0)
    parser_infer_tab.add_argument("--inseam", type=float, default=80.0)
    parser_infer_tab.add_argument("--outseam", type=float, default=100.0)
    parser_infer_tab.add_argument("--thigh", type=float, default=55.0)
    parser_infer_tab.add_argument("--bicep",     type=float, default=28.0)
    parser_infer_tab.add_argument("--n_samples", type=int,   default=32,
                                  help="z samples to average at inference")
    parser_infer_tab.add_argument("--output", type=str, default="image",
                                  choices=["image", "mesh", "both"],
                                  help="Output format: 2D PNG image, 3D OBJ mesh, or both")
    parser_infer_tab.add_argument("--view", type=str, default="front",
                                  choices=["front", "side", "back"],
                                  help="Camera view used for the 2D PNG render")
    parser_infer_tab.add_argument("--image_size", type=int, default=512,
                                  help="Output PNG size in pixels")
    parser_infer_tab.add_argument("--wireframe", action="store_true",
                                  help="Draw triangle edges in the 2D PNG render")
    parser_infer_tab.add_argument("--show", action="store_true", help="Show 3D viewer")

    args = parser.parse_args()

    if args.command == "fit":
        fit_betas()
    elif args.command == "train":
        trainer = WGANGPTrainer()
        trainer.train()
    elif args.command == "eval":
        evaluate()
    elif args.command == "infer":
        infer(args)
    elif args.command == "train_img":
        trainer = ImgWGANGPTrainer()
        trainer.train()
    elif args.command == "infer_img":
        infer_img(args)
    elif args.command == "train_tab":
        trainer = TabWGANGPTrainer()
        trainer.train()
    elif args.command == "eval_tab":
        evaluate_tab()
    elif args.command == "infer_tab":
        infer_tab(args)

if __name__ == "__main__":
    main()
