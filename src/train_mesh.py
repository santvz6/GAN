"""
WGAN-GP trainer for the Mesh GAN.

Trains on subsampled NOMO3D point clouds (~356 scans, each 6890 points).
Optionally augments with synthetic SMPL meshes generated from the
tabular WGAN-GP's outputs (set MeshHParams.use_synthetic_smpl_meshes).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import grad
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from src.config.paths import Paths
from src.config.hparams import MeshHParams
from src.data.nomo3d_obj_loader import NOMO3DPointCloudDataset
from src.models.mesh_gan import MeshGenerator, PointNetDiscriminator


class MeshWGANGPTrainer:
    def __init__(self, hparams: MeshHParams = None):
        Paths.init_project()
        self.hp = hparams or MeshHParams()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.G = MeshGenerator(self.hp).to(self.device)
        self.D = PointNetDiscriminator(self.hp).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.hp.lr_g, betas=(self.hp.b1, self.hp.b2))
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.hp.lr_d, betas=(self.hp.b1, self.hp.b2))

        ds = NOMO3DPointCloudDataset(num_points=self.hp.num_points)
        self.dataloader = DataLoader(
            ds,
            batch_size=self.hp.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hp.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        self.exp_dir = Paths.EXPERIMENTS_MESH
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir = Paths.MESH_SAMPLES_DIR
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        self.fixed_z = torch.randn(4, self.hp.noise_dim, device=self.device)

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        bs = real.size(0)
        alpha = torch.rand(bs, 1, 1, device=self.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interp = self.D(interp)
        grads = grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads = grads.reshape(bs, -1)
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def _save_samples(self, epoch: int):
        self.G.eval()
        with torch.no_grad():
            fake = self.G(self.fixed_z).cpu().numpy()
        self.G.train()
        np.save(self.sample_dir / f"mesh_samples_ep{epoch:04d}.npy", fake)

    def _save_checkpoint(self, epoch: int):
        torch.save({
            "epoch":        epoch,
            "G_state_dict": self.G.state_dict(),
            "D_state_dict": self.D.state_dict(),
            "opt_G":        self.opt_G.state_dict(),
            "opt_D":        self.opt_D.state_dict(),
        }, self.exp_dir / f"mesh_wgangp_ep{epoch:04d}.pt")

    def train(self):
        print(f"[Mesh GAN] device={self.device} | dataset={len(self.dataloader.dataset)} | batches={len(self.dataloader)}")
        print(f"[Mesh GAN] G params: {sum(p.numel() for p in self.G.parameters()):,}")
        print(f"[Mesh GAN] D params: {sum(p.numel() for p in self.D.parameters()):,}")

        for epoch in range(self.hp.epochs):
            d_losses, g_losses = [], []

            for real in tqdm(self.dataloader, desc=f"ep {epoch}", leave=False):
                real = real.to(self.device, non_blocking=True)
                bs = real.size(0)

                for _ in range(self.hp.n_critic):
                    self.opt_D.zero_grad()
                    z = torch.randn(bs, self.hp.noise_dim, device=self.device)
                    fake = self.G(z).detach()
                    gp = self._gradient_penalty(real.data, fake.data)
                    d_loss = self.D(fake).mean() - self.D(real).mean() + self.hp.lambda_gp * gp
                    d_loss.backward()
                    self.opt_D.step()
                d_losses.append(d_loss.item())

                self.opt_G.zero_grad()
                z = torch.randn(bs, self.hp.noise_dim, device=self.device)
                g_loss = -self.D(self.G(z)).mean()
                g_loss.backward()
                self.opt_G.step()
                g_losses.append(g_loss.item())

            if epoch % 10 == 0 or epoch == self.hp.epochs - 1:
                print(f"[ep {epoch}] D={np.mean(d_losses):.4f}  G={np.mean(g_losses):.4f}")
            if epoch % self.hp.sample_interval == 0 or epoch == self.hp.epochs - 1:
                self._save_samples(epoch)
            if epoch % self.hp.checkpoint_interval == 0 or epoch == self.hp.epochs - 1:
                self._save_checkpoint(epoch)


if __name__ == "__main__":
    MeshWGANGPTrainer().train()
