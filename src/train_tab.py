"""WGAN-GP training loop for the Tab Transformer on point clouds."""
import os
import csv
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import grad
from tqdm import tqdm

from src.config.paths import Paths
from src.config.hparams import TabHParams
from src.data.pc_dataset import PointCloudDataset
from src.models import TabTransformerGenerator, TabTransformerDiscriminator


class TabWGANGPTrainer:
    def __init__(self):
        Paths.init_project()
        self.hp = TabHParams()
        self.device = torch.device("mps")

        self.G = TabTransformerGenerator(self.hp).to(self.device)
        self.D = TabTransformerDiscriminator(self.hp).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.hp.lr_g, betas=(self.hp.b1, self.hp.b2))
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.hp.lr_d, betas=(self.hp.b1, self.hp.b2))

        train_ds = PointCloudDataset(split='train', hparams=self.hp)
        self.dataloader = DataLoader(
            train_ds,
            batch_size=self.hp.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hp.num_workers,
            pin_memory=True,
            persistent_workers=self.hp.num_workers > 0,
        )

        # --- Point cloud normalisation stats ---
        self.pc_mean, self.pc_std = self._compute_pc_scaler(train_ds)
        np.savez(
            str(Paths.EXPERIMENTS_DIR / "pc_scaler.npz"),
            mean=self.pc_mean.numpy(),
            std=self.pc_std.numpy(),
        )
        print(f"  [PC SCALER] Saved -> {Paths.EXPERIMENTS_DIR / 'pc_scaler.npz'}")

        # --- Sample log CSV ---
        self.sample_log_path = Paths.LOGS_DIR / "tab_beta_samples.csv"
        self._init_sample_log()

        # --- Fixed probe point cloud (first training sample, reused for logging) ---
        probe_pc, _, _ = train_ds[0]
        self._probe_pc = self._normalise_pc(probe_pc.unsqueeze(0)).to(self.device)

    # ------------------------------------------------------------------
    def _compute_pc_scaler(self, dataset: PointCloudDataset):
        """Compute mean and std of point cloud coordinates across training split."""
        all_points = []
        for i in range(len(dataset)):
            pc, _, _ = dataset[i]
            all_points.append(pc)  # (n_pc_points, 3)
        all_points = torch.stack(all_points)  # (N, n_pc_points, 3)
        # Flatten to (N * n_pc_points, 3) for global stats
        flat = all_points.view(-1, 3)
        mean = flat.mean(dim=0)  # (3,)
        std = flat.std(dim=0).clamp(min=1e-6)  # (3,)
        return mean, std

    def _normalise_pc(self, pc: torch.Tensor) -> torch.Tensor:
        """Normalise point cloud coordinates to roughly N(0,1)."""
        return (pc - self.pc_mean) / self.pc_std

    # ------------------------------------------------------------------
    def _init_sample_log(self):
        """Create CSV with header if it doesn't exist yet."""
        if not self.sample_log_path.exists():
            with open(self.sample_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["epoch", "timestamp"] + [f"beta_{i}" for i in range(self.hp.num_betas)]
                writer.writerow(header)

    def _log_sample(self, epoch: int):
        """Run probe through G, append betas to CSV."""
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(1, self.hp.noise_dim, device=self.device)
            betas = self.G(z, self._probe_pc).cpu().numpy().flatten().tolist()
        self.G.train()

        with open(self.sample_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            writer.writerow([epoch, ts] + [f"{b:.6f}" for b in betas])

    # ------------------------------------------------------------------
    def compute_gradient_penalty(self, real_samples, fake_samples, pc):
        alpha = torch.rand(real_samples.size(0), 1, device=self.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

        d_interpolates = self.D(interpolates, pc)
        fake = torch.ones(real_samples.size(0), 1, device=self.device)

        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int):
        ckpt_path = Paths.EXPERIMENTS_DIR / f"wgangp_tab_ckpt_{epoch}.pt"
        torch.save({
            "epoch":        epoch,
            "G_state_dict": self.G.state_dict(),
            "D_state_dict": self.D.state_dict(),
            "opt_G":        self.opt_G.state_dict(),
            "opt_D":        self.opt_D.state_dict(),
        }, ckpt_path)
        print(f"  [CKPT] Saved -> {ckpt_path.name}")

    # ------------------------------------------------------------------
    def train(self):
        print(f"Training Tab Transformer (point cloud) on: {self.device}")
        print(f"Batch size  : {self.hp.batch_size}  |  Workers: {self.hp.num_workers}")
        print(f"PC points   : {self.hp.n_pc_points}")
        g_params = sum(p.numel() for p in self.G.parameters())
        d_params = sum(p.numel() for p in self.D.parameters())
        print(f"G params    : {g_params:,}  |  D params: {d_params:,}")
        print(f"Sample log  : {self.sample_log_path}")
        print()

        for epoch in range(self.hp.epochs):
            g_losses = []
            d_losses = []

            for i, (pc, betas, _) in enumerate(self.dataloader):
                pc         = self._normalise_pc(pc).to(self.device, non_blocking=True)
                real_betas = betas.to(self.device, non_blocking=True)
                batch_size = pc.size(0)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                for _ in range(self.hp.n_critic):
                    self.opt_D.zero_grad()

                    z          = torch.randn(batch_size, self.hp.noise_dim, device=self.device)
                    fake_betas = self.G(z, pc).detach()

                    real_validity = self.D(real_betas, pc)
                    fake_validity = self.D(fake_betas, pc)

                    gp = self.compute_gradient_penalty(real_betas.data, fake_betas.data, pc)
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.hp.lambda_gp * gp
                    d_loss.backward()
                    self.opt_D.step()

                d_losses.append(d_loss.item())

                # -----------------
                #  Train Generator
                # -----------------
                self.opt_G.zero_grad()

                z          = torch.randn(batch_size, self.hp.noise_dim, device=self.device)
                fake_betas = self.G(z, pc)

                fake_validity = self.D(fake_betas, pc)
                g_loss        = -torch.mean(fake_validity)
                g_loss.backward()
                self.opt_G.step()

                g_losses.append(g_loss.item())

            # ----- Logging -----
            if epoch % 10 == 0 or epoch == self.hp.epochs - 1:
                print(
                    f"[Epoch {epoch}/{self.hp.epochs}] "
                    f"[D loss: {np.mean(d_losses):.4f}] "
                    f"[G loss: {np.mean(g_losses):.4f}]"
                )

            # ----- Sample log every sample_interval -----
            if epoch % self.hp.sample_interval == 0 or epoch == self.hp.epochs - 1:
                self._log_sample(epoch)

            # ----- Checkpoint every checkpoint_interval -----
            if epoch % self.hp.checkpoint_interval == 0 or epoch == self.hp.epochs - 1:
                self._save_checkpoint(epoch)


if __name__ == "__main__":
    trainer = TabWGANGPTrainer()
    trainer.train()
