import torch
import torch.nn as nn
from src.config.hparams import HParams

class Discriminator(nn.Module):
    def __init__(self, hparams=None):
        super().__init__()
        self.hp = hparams or HParams()
        
        # Discriminator takes betas (real or fake) + measurements (condition)
        input_dim = self.hp.num_betas + self.hp.cond_dim
        
        layers = []
        in_dim = input_dim
        for hidden_dim in self.hp.d_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            # No BatchNorm in WGAN-GP discriminator (LayerNorm is okay, but we'll stick to simple first)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, 1))
        # No sigmoid at the end for WGAN
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, betas, cond):
        x = torch.cat([betas, cond], dim=1)
        return self.model(x)

if __name__ == "__main__":
    d = Discriminator()
    betas = torch.randn(4, d.hp.num_betas)
    cond = torch.randn(4, d.hp.cond_dim)
    out = d(betas, cond)
    print(f"Discriminator output shape: {out.shape}")
