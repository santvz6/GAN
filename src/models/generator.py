import torch
import torch.nn as nn
from src.config.hparams import HParams


class ResidualBlock(nn.Module):
    """Linear residual block with LayerNorm (safe for WGAN-GP)."""
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class Generator(nn.Module):
    """
    Conditional Generator: (z, measurements) -> betas (10 SMPL shape params).

    Architecture:
      - Input projection: [noise_dim + cond_dim] -> first hidden dim
      - Series of linear layers from g_hidden_dims
      - ResidualBlock on the widest (middle) layer for gradient flow
      - Output head: last hidden dim -> num_betas (linear, no activation)
    """
    def __init__(self, hparams=None):
        super().__init__()
        self.hp = hparams or HParams()

        input_dim = self.hp.noise_dim + self.hp.cond_dim
        dims = self.hp.g_hidden_dims  # e.g. [256, 512, 1024, 512, 256]

        # --- Input projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, dims[0]),
            nn.LayerNorm(dims[0]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # --- Hidden tower ---
        hidden_layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            hidden_layers.append(nn.Linear(in_d, out_d))
            hidden_layers.append(nn.LayerNorm(out_d))
            hidden_layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.hidden = nn.Sequential(*hidden_layers)

        # --- Residual block on the peak dimension ---
        peak_dim = max(dims)
        self.res_block = ResidualBlock(peak_dim)

        # --- Output head ---
        self.output_head = nn.Linear(dims[-1], self.hp.num_betas)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        x = self.input_proj(x)
        x = self.hidden(x)
        # Residual block only if current dim == peak_dim
        # (if hidden dims are symmetric the last layer == peak at midpoint,
        #  we apply it after hidden tower regardless — dims[-1] == dims[0])
        # To apply it correctly we hook it mid-tower via forward-pass dimension check:
        # But since we build it as nn.Sequential we apply it if dims[-1] == peak_dim.
        # For the default [256,512,1024,512,256] the final dim IS 256, not 1024.
        # We therefore insert the residual at the peak inside the hidden tower:
        # (see _build_hidden_with_residual for clarity — this is a safe no-op path)
        return self.output_head(x)

    # ------------------------------------------------------------------ #
    # Override: rebuild hidden tower with residual at peak dim             #
    # ------------------------------------------------------------------ #
    def _build(self):
        pass  # hook kept for subclassing


class _GeneratorWithResidual(Generator):
    """
    Proper residual-at-peak implementation.
    Replaces Generator; imported as `Generator` from __init__.
    """
    def __init__(self, hparams=None):
        # Skip parent __init__ and build from scratch
        nn.Module.__init__(self)
        self.hp = hparams or HParams()

        input_dim = self.hp.noise_dim + self.hp.cond_dim
        dims = self.hp.g_hidden_dims

        peak_dim = max(dims)

        # Build segments: pre-peak, residual, post-peak
        pre, post = [], []
        in_d = input_dim
        peak_reached = False

        for out_d in dims:
            segment = post if peak_reached else pre
            segment.append(nn.Linear(in_d, out_d))
            segment.append(nn.LayerNorm(out_d))
            segment.append(nn.LeakyReLU(0.2, inplace=True))
            if out_d == peak_dim and not peak_reached:
                peak_reached = True
            in_d = out_d

        self.pre_peak  = nn.Sequential(*pre)
        self.res_block = ResidualBlock(peak_dim)
        self.post_peak = nn.Sequential(*post)
        self.out       = nn.Linear(dims[-1], self.hp.num_betas)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        x = self.pre_peak(x)
        x = self.res_block(x)
        x = self.post_peak(x)
        return self.out(x)


# Export the residual-capable version as `Generator`
Generator = _GeneratorWithResidual


if __name__ == "__main__":
    g = Generator()
    z    = torch.randn(4, g.hp.noise_dim)
    cond = torch.randn(4, g.hp.cond_dim)
    out  = g(z, cond)
    print(f"Generator output shape: {out.shape}")   # (4, 10)
    total = sum(p.numel() for p in g.parameters())
    print(f"Generator params: {total:,}")
