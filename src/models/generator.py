import torch
import torch.nn as nn
from src.config.hparams import HParams

class Generator(nn.Module):
    def __init__(self, hparams=None):
        super().__init__()
        self.hp = hparams or HParams()
        
        input_dim = self.hp.noise_dim + self.hp.cond_dim
        
        layers = []
        in_dim = input_dim
        for hidden_dim in self.hp.g_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, self.hp.num_betas))
        # Betas are usually in range [-3, 3], scaling tanh output
        # layers.append(nn.Tanh()) 
        # But we can just leave it linear and let it learn the scale,
        # or use tanh and multiply by 3. Let's stick to linear for now.
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        return self.model(x)

if __name__ == "__main__":
    g = Generator()
    z = torch.randn(4, g.hp.noise_dim)
    cond = torch.randn(4, g.hp.cond_dim)
    out = g(z, cond)
    print(f"Generator output shape: {out.shape}")
