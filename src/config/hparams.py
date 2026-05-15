from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class HParams:
    # Generator
    noise_dim: int = 64
    cond_dim: int = 10  # 10 measurements
    g_hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 1024, 512, 256])
    
    # Discriminator
    d_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    # SMPL
    num_betas: int = 10
    
    # Training
    batch_size: int = 128
    epochs: int = 2000
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    b1: float = 0.5
    b2: float = 0.9
    
    # WGAN-GP
    n_critic: int = 5
    lambda_gp: float = 10.0
    
    # Logging / checkpointing
    checkpoint_interval: int = 500   # save weights every N epochs
    sample_interval:     int = 100   # log sample betas every N epochs
    num_workers:         int = 4     # DataLoader parallel workers
    
    # Beta fitting (NOMO3D pseudo-GT — iterative Gauss-Newton)
    fit_max_iters: int = 8
    fit_tol: float = 1e-3        # ||Δβ||₂ stopping criterion
    fit_rel_tol: float = 1e-4    # relative residual improvement stopping criterion
    fit_eps: float = 0.15        # central-difference step for Jacobian
    fit_max_backtracks: int = 5  # backtracking line-search halvings

    # Dataset
    train_split: float = 0.8
    meas_cols: List[str] = field(default_factory=lambda: [
        "Head_Top_Height", # approx height
        "BUST_Circ",
        "NaturalWAIST_Circ",
        "HIP_Circ",
        "NeckBase_Circ",
        "Shoulder_to_Shoulder",
        "Inseam",
        "Outseam",
        "Thigh_Circ",
        "Bicep_Circ"
    ])


@dataclass
class ImgHParams:
    # Image
    image_size: int = 64
    channels: int = 1  # TNT15 segmented PNGs are grayscale

    # Generator (DCGAN-style)
    noise_dim: int = 128
    g_feature_maps: int = 64    # base channel multiplier for G

    # Discriminator
    d_feature_maps: int = 64

    # Training (WGAN-GP)
    batch_size: int = 64
    epochs: int = 100
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    b1: float = 0.5
    b2: float = 0.9
    n_critic: int = 5
    lambda_gp: float = 10.0

    # Logging / checkpoints
    checkpoint_interval: int = 5
    sample_interval: int = 1
    sample_grid: int = 64        # how many images per sample grid (8x8)
    num_workers: int = 4

    # Dataset
    frame_stride: int = 5         # subsample 1 of every N frames per camera
    subjects: Tuple[str, ...] = ("mr", "pz", "sg", "sp")
    cameras: Tuple[str, ...] = ("00", "01", "02", "03", "04")
    train_split: float = 0.9
