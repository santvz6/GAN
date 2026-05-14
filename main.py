import argparse
from src.config.paths import Paths


def main():
    Paths.init_project()

    parser = argparse.ArgumentParser(description="GAN 3D Body Generation Pipeline")
    sub = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # ----- Tabular (medidas -> betas) -----
    sub.add_parser("fit",   help="Fit pseudo-ground-truth betas for the dataset")
    sub.add_parser("train", help="Train the tabular WGAN-GP model (medidas -> betas)")
    sub.add_parser("eval",  help="Evaluate the tabular model (MAE of extracted measurements)")

    p_infer = sub.add_parser("infer", help="Run inference and generate 3D mesh from measurements")
    p_infer.add_argument("--gender", type=str, default="FEMALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    p_infer.add_argument("--height", type=float, default=170.0)
    p_infer.add_argument("--bust",    type=float, default=90.0)
    p_infer.add_argument("--waist",   type=float, default=70.0)
    p_infer.add_argument("--hip",     type=float, default=95.0)
    p_infer.add_argument("--neck",    type=float, default=34.0)
    p_infer.add_argument("--shoulder",type=float, default=40.0)
    p_infer.add_argument("--inseam",  type=float, default=80.0)
    p_infer.add_argument("--outseam", type=float, default=100.0)
    p_infer.add_argument("--thigh",   type=float, default=55.0)
    p_infer.add_argument("--bicep",   type=float, default=28.0)
    p_infer.add_argument("--show",    action="store_true", help="Show 3D viewer")

    # ----- Image GAN (TNT15 + skeleton renders) -----
    sub.add_parser("render-skeletons", help="Render SMPL skeletons from cached betas to PNG (data prep for Image GAN)")
    sub.add_parser("train-image",      help="Train the Image GAN (DCGAN on TNT15 silhouettes)")
    sub.add_parser("eval-image",       help="Evaluate the Image GAN (FID)")

    # ----- Mesh GAN (NOMO3D point clouds) -----
    sub.add_parser("train-mesh", help="Train the Mesh GAN (PointNet WGAN-GP on NOMO3D scans)")
    sub.add_parser("eval-mesh",  help="Evaluate the Mesh GAN (Chamfer Distance + F-score)")

    # ----- Tabular extra metric -----
    sub.add_parser("eval-mmd", help="Evaluate the tabular GAN with MMD between real and generated betas")

    args = parser.parse_args()

    if args.command == "fit":
        from src.data.beta_fitter import fit_betas
        fit_betas()

    elif args.command == "train":
        from src.train import WGANGPTrainer
        WGANGPTrainer().train()

    elif args.command == "eval":
        from src.eval import evaluate
        evaluate()

    elif args.command == "infer":
        from src.inference import infer
        infer(args)

    elif args.command == "render-skeletons":
        from src.data.render_skeleton import render_batch_from_betas_cache
        n = render_batch_from_betas_cache()
        print(f"Rendered {n} skeleton images.")

    elif args.command == "train-image":
        from src.train_image import ImageWGANGPTrainer
        ImageWGANGPTrainer().train()

    elif args.command == "eval-image":
        from src.eval_image import evaluate_fid
        evaluate_fid()

    elif args.command == "train-mesh":
        from src.train_mesh import MeshWGANGPTrainer
        MeshWGANGPTrainer().train()

    elif args.command == "eval-mesh":
        from src.eval_3d import evaluate_3d
        evaluate_3d()

    elif args.command == "eval-mmd":
        from src.eval_tabular import evaluate_mmd
        evaluate_mmd()


if __name__ == "__main__":
    main()
