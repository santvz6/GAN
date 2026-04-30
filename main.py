import argparse
from src.config.paths import Paths
from src.data.beta_fitter import fit_betas
from src.train import WGANGPTrainer
from src.eval import evaluate
from src.inference import infer

def main():
    Paths.init_project()

    parser = argparse.ArgumentParser(description="GAN 3D Body Generation Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)
    
    # Fit command
    subparsers.add_parser("fit", help="Fit pseudo-ground-truth betas for dataset")
    
    # Train command
    subparsers.add_parser("train", help="Train the WGAN-GP model")
    
    # Eval command
    subparsers.add_parser("eval", help="Evaluate the model (MAE)")
    
    # Infer command
    parser_infer = subparsers.add_parser("infer", help="Run inference and generate 3D mesh")
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
    parser_infer.add_argument("--bicep", type=float, default=28.0)
    parser_infer.add_argument("--show", action="store_true", help="Show 3D viewer")
    
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

if __name__ == "__main__":
    main()