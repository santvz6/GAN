"""WGAN-GP training loop for the image GAN on TNT15."""
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import grad
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from src.config.paths import Paths
from src.config.hparams import ImgHParams
from src.data.img_dataset import TNT15ImageDataset
from src.models import ImgGenerator, ImgDiscriminator


class ImgWGANGPTrainer:
    def __init__(self):
        Paths.init_project()
        self.hp = ImgHParams()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.G = ImgGenerator(self.hp).to(self.device)
        self.D = ImgDiscriminator(self.hp).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.hp.lr_g, betas=(self.hp.b1, self.hp.b2))
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.hp.lr_d, betas=(self.hp.b1, self.hp.b2))

        train_ds = TNT15ImageDataset(split="train", hparams=self.hp)
        self.dataloader = DataLoader(
            train_ds,
            batch_size=self.hp.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hp.num_workers,
            pin_memory=True,
            persistent_workers=self.hp.num_workers > 0,
        )

        # Fixed noise to monitor visual progression across epochs
        self._fixed_z = torch.randn(self.hp.sample_grid, self.hp.noise_dim, device=self.device)

    def compute_gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        bsz = real.size(0)
        alpha = torch.rand(bsz, 1, 1, 1, device=self.device)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interpolates = self.D(interpolates)
        grad_outputs = torch.ones_like(d_interpolates)
        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(bsz, -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def _save_checkpoint(self, epoch: int):
        path = Paths.EXPERIMENTS_DIR / f"{Paths.IMG_CKPT_PREFIX}{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "G_state_dict": self.G.state_dict(),
            "D_state_dict": self.D.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
        }, path)
        print(f"  [CKPT] Saved -> {path.name}")

    def _save_sample_grid(self, epoch: int):
        self.G.eval()
        with torch.no_grad():
            fake = self.G(self._fixed_z).cpu()
        self.G.train()
        # de-normalise [-1,1] -> [0,1] for saving
        fake = (fake + 1.0) / 2.0
        grid = make_grid(fake, nrow=int(self.hp.sample_grid ** 0.5))
        out = Paths.IMG_SAMPLES_DIR / f"sample_epoch_{epoch:04d}.png"
        save_image(grid, out)
        print(f"  [SAMPLE] {out.name}")

    def train(self):
        print(f"Training image GAN on: {self.device}")
        print(f"Batch size  : {self.hp.batch_size}  |  Workers: {self.hp.num_workers}")
        print(f"Train frames: {len(self.dataloader.dataset)}")
        g_params = sum(p.numel() for p in self.G.parameters())
        d_params = sum(p.numel() for p in self.D.parameters())
        print(f"G params    : {g_params:,}  |  D params: {d_params:,}")
        print()

        for epoch in range(self.hp.epochs):
            g_losses, d_losses = [], []
            data_iter = iter(self.dataloader)
            n_batches = len(self.dataloader)

            pbar = tqdm(range(n_batches), desc=f"Epoch {epoch}/{self.hp.epochs}")
            for _ in pbar:
                # ---- D step (n_critic times) ----
                for _ in range(self.hp.n_critic):
                    try:
                        real = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.dataloader)
                        real = next(data_iter)

                    real = real.to(self.device, non_blocking=True)
                    bsz = real.size(0)

                    self.opt_D.zero_grad()
                    z = torch.randn(bsz, self.hp.noise_dim, device=self.device)
                    fake = self.G(z).detach()

                    d_real = self.D(real)
                    d_fake = self.D(fake)
                    gp = self.compute_gradient_penalty(real.data, fake.data)
                    d_loss = -d_real.mean() + d_fake.mean() + self.hp.lambda_gp * gp
                    d_loss.backward()
                    self.opt_D.step()

                d_losses.append(d_loss.item())

                # ---- G step ----
                self.opt_G.zero_grad()
                z = torch.randn(self.hp.batch_size, self.hp.noise_dim, device=self.device)
                fake = self.G(z)
                g_loss = -self.D(fake).mean()
                g_loss.backward()
                self.opt_G.step()
                g_losses.append(g_loss.item())

                pbar.set_postfix(d=f"{np.mean(d_losses[-50:]):.3f}",
                                 g=f"{np.mean(g_losses[-50:]):.3f}")

            print(f"[Epoch {epoch}/{self.hp.epochs}] "
                  f"D loss: {np.mean(d_losses):.4f}  |  G loss: {np.mean(g_losses):.4f}")

            if epoch % self.hp.sample_interval == 0 or epoch == self.hp.epochs - 1:
                self._save_sample_grid(epoch)
            if epoch % self.hp.checkpoint_interval == 0 or epoch == self.hp.epochs - 1:
                self._save_checkpoint(epoch)


if __name__ == "__main__":
    trainer = ImgWGANGPTrainer()
    trainer.train()
