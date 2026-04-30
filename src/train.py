import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import grad
from tqdm import tqdm

from src.config.paths import Paths
from src.config.hparams import HParams
from src.data.dataset import NOMODataset
from src.models import Generator, Discriminator

class WGANGPTrainer:
    def __init__(self):
        Paths.init_project()
        self.hp = HParams()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.G = Generator(self.hp).to(self.device)
        self.D = Discriminator(self.hp).to(self.device)
        
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.hp.lr_g, betas=(self.hp.b1, self.hp.b2))
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.hp.lr_d, betas=(self.hp.b1, self.hp.b2))
        
        train_ds = NOMODataset(split='train')
        self.dataloader = DataLoader(train_ds, batch_size=self.hp.batch_size, shuffle=True, drop_last=True)
        
    def compute_gradient_penalty(self, real_samples, fake_samples, cond):
        alpha = torch.rand(real_samples.size(0), 1).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates = self.D(interpolates, cond)
        
        fake = torch.ones(real_samples.size(0), 1).to(self.device)
        
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

    def train(self):
        print(f"Starting WGAN-GP training on device: {self.device}")
        
        for epoch in range(self.hp.epochs):
            g_losses = []
            d_losses = []
            
            for i, (meas, betas, _) in enumerate(self.dataloader):
                meas = meas.to(self.device)
                real_betas = betas.to(self.device)
                
                batch_size = meas.size(0)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                for _ in range(self.hp.n_critic):
                    self.opt_D.zero_grad()
                    
                    z = torch.randn(batch_size, self.hp.noise_dim).to(self.device)
                    fake_betas = self.G(z, meas)
                    
                    real_validity = self.D(real_betas, meas)
                    fake_validity = self.D(fake_betas, meas)
                    
                    gp = self.compute_gradient_penalty(real_betas.data, fake_betas.data, meas)
                    
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.hp.lambda_gp * gp
                    d_loss.backward()
                    self.opt_D.step()
                    
                d_losses.append(d_loss.item())
                
                # -----------------
                #  Train Generator
                # -----------------
                self.opt_G.zero_grad()
                
                z = torch.randn(batch_size, self.hp.noise_dim).to(self.device)
                fake_betas = self.G(z, meas)
                
                fake_validity = self.D(fake_betas, meas)
                g_loss = -torch.mean(fake_validity)
                
                g_loss.backward()
                self.opt_G.step()
                
                g_losses.append(g_loss.item())
                
            if epoch % 10 == 0 or epoch == self.hp.epochs - 1:
                print(f"[Epoch {epoch}/{self.hp.epochs}] [D loss: {np.mean(d_losses):.4f}] [G loss: {np.mean(g_losses):.4f}]")
                
            # Save checkpoint periodically
            if epoch % 500 == 0 or epoch == self.hp.epochs - 1:
                ckpt_path = Paths.EXPERIMENTS_DIR / f"wgangp_ckpt_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'G_state_dict': self.G.state_dict(),
                    'D_state_dict': self.D.state_dict(),
                }, ckpt_path)

if __name__ == "__main__":
    import numpy as np
    trainer = WGANGPTrainer()
    trainer.train()
