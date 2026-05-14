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
class ImageHParams:
    """Hiperparámetros del Image GAN (DCGAN 128x128 + WGAN-GP)."""
    noise_dim:   int = 128
    image_size:  int = 128
    channels:    int = 1                # silhouette grayscale; set 3 for RGB
    g_base_ch:   int = 64               # base channels for ConvTranspose tower
    d_base_ch:   int = 64               # base channels for Conv tower

    batch_size:  int = 64
    epochs:      int = 200
    lr_g:        float = 1e-4
    lr_d:        float = 1e-4
    b1:          float = 0.0
    b2:          float = 0.9

    n_critic:    int = 5
    lambda_gp:   float = 10.0

    checkpoint_interval: int = 25
    sample_interval:     int = 5
    num_workers:         int = 0        # 0 on Windows for safety

    tnt15_subdirs: List[str] = field(default_factory=lambda: ["00", "01", "02", "03", "04"])
    use_rendered_skeletons: bool = True  # mezcla renders sintéticos con los PNG reales


@dataclass
class MeshHParams:
    """Hiperparámetros del Mesh GAN (point cloud 6890 pts + PointNet disc)."""
    noise_dim:   int = 128
    num_points:  int = 6890

    g_hidden_dims: List[int] = field(default_factory=lambda: [1024, 4096])
    d_pn_dims:     List[int] = field(default_factory=lambda: [64, 128, 256])
    d_head_dims:   List[int] = field(default_factory=lambda: [256, 128])

    batch_size:  int = 16               # 6890*3 floats * batch — modest size
    epochs:      int = 500
    lr_g:        float = 1e-4
    lr_d:        float = 1e-4
    b1:          float = 0.0
    b2:          float = 0.9

    n_critic:    int = 5
    lambda_gp:   float = 10.0

    checkpoint_interval: int = 50
    sample_interval:     int = 25
    num_workers:         int = 0

    use_synthetic_smpl_meshes: bool = True   # complementa NOMO3D con mallas SMPL del tabular GAN
