from dataclasses import dataclass, field
from typing import List

@dataclass
class HParams:
    # Generator
    noise_dim: int = 64
    cond_dim: int = 10  # 10 measurements
    g_hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 256])
    
    # Discriminator
    d_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    # SMPL
    num_betas: int = 10
    
    # Training
    batch_size: int = 32
    epochs: int = 2000
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    b1: float = 0.5
    b2: float = 0.9
    
    # WGAN-GP
    n_critic: int = 5
    lambda_gp: float = 10.0
    
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
