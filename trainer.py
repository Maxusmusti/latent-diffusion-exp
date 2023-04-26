from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from vqgan.interface import pretrained_vqgan
from data import load_data
from diffusion import Diffusion
from unet import UNet
import pytorch_lightning as pl
from torchvision.utils import make_grid
from argparse import Namespace

class DiffusionTrainer(pl.LightningModule):
    def __init__(self, args):
        super(DiffusionTrainer, self).__init__()
        self.args = args
        print(self.device)
        self.diffusion = Diffusion(image_size=args.image_size, device=args.device)
        self.vqgan = pretrained_vqgan().eval()
        for param in self.vqgan.parameters():
            param.requires_grad = False
        self.encode = self.vqgan.encode
        self.decode = self.vqgan.decode

        self.model = UNet()
   
   
   def forward(self, x_t, t):
        return self.model(x_t, t)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.lr)

    def log_images(self, noised_image, reconstructed_image, orig_image, predicted_noise, step, stage, image_type):
        noised_image, reconstructed_image, orig_image, predicted_noise = noised_image.mul(0.5).add(0.5), reconstructed_image.mul(0.5).add(0.5), orig_image.mul(0.5).add(0.5), predicted_noise.mul(0.5).add(0.5)
        grid = make_grid(torch.cat([noised_image, reconstructed_image, orig_image, predicted_noise]), nrow=2)
        
        self.logger.experiment.add_image(f"{stage}/{image_type}_Images", grid, global_step=step)

    def training_step(self, batch, batch_idx):
        images, _ = batch
        t = torch.randint(low=1, high=self.diffusion.num_steps, size=(images.shape[0],)).to(self.device)
        encoded_images = self.encode(images)
        x_t, noise = self.diffusion.forward(encoded_images, t)
        predicted_noise = self.model(x_t, t)
        loss = nn.MSELoss()(noise, predicted_noise)

        # Log images
        if batch_idx % self.args.log_image_interval == 0:
                predicted_images = x_t - predicted_noise
                pixel_equiv_noised_image = self.decode(x_t)
                pixel_reconstructed_images = self.decode(predicted_images)  # Get the reconstructed image
                pixel_predicted_noise = images - pixel_reconstructed_images
                self.log_images(pixel_equiv_noised_image, pixel_reconstructed_images, images, pixel_predicted_noise, self.global_step, "Train", "Pixel-Space")
                self.log_images(x_t, predicted_images, encoded_images, noise, self.global_step, "Train", "Latent-Space")
            
        self.log("Train/MSE", loss)
        self.log("mse", loss, prog_bar=True, logger=False)  # Add this line
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        t = torch.randint(low=1, high=self.diffusion.num_steps, size=(images.shape[0],)).to(self.device)
        encoded_images = self.encode(images)
        x_t, noise = self.diffusion.forward(encoded_images, t)
        predicted_noise = self.model(x_t, t)
        loss = nn.MSELoss()(noise, predicted_noise)

        # Log images
        if batch_idx % self.args.log_image_interval == 0:
            predicted_images = x_t - predicted_noise
            pixel_equiv_noised_image = self.decode(x_t)
            pixel_reconstructed_images = self.decode(predicted_images)
            pixel_predicted_noise = images - pixel_reconstructed_images
            self.log_images(pixel_equiv_noised_image, pixel_reconstructed_images, images, pixel_predicted_noise, self.global_step, "Val", "Pixel-Space")
            self.log_images(x_t, predicted_images, encoded_images, noise, self.global_step, "Val", "Latent-Space")

        self.log("Val/MSE", loss)
        self.log("mse", loss, prog_bar=True, logger=False)  # Add this line
        return {"val_loss": loss}

    def train_dataloader(self):
        return load_data(self.args.dataset_path, self.args.batch_size, self.args.orig_resolution, self.args.num_workers)

    def val_dataloader(self):
        return load_data(self.args.dataset_path)

default_args = Namespace(
    run_name='diff',
    orig_resolution = 256,
    epochs=30,
    batch_size=4,
    dataset_path='wikiart_images',
    device='cuda',
    lr=3e-4,
    save_interval=100,
    image_size=[32, 32],
    log_image_interval=10,
    num_workers = 4 
)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='diff', type=str)
    parser.add_argument('--orig_resolution', default=256, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--dataset_path', default='wikiart_images', type=str)
    parser.add_argument('--val_dataset_path', default='wikiart_images', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--val_batch_per_epoch', default=0, type=int)
    parser.add_argument('--save_interval', default=100, type=int)
    parser.add_argument('--image_size', default=[32, 32], type=int, nargs=2)
    parser.add_argument('--log_image_interval', default=50, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str)

    args = parser.parse_args()

    # Example of how to train the DiffusionTrainer using PyTorch Lightning
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator = 'gpu', limit_val_batches=args.val_batch_per_epoch, log_every_n_steps=1)
    diffusion_trainer = DiffusionTrainer(args)
    trainer.fit(diffusion_trainer, ckpt_path = args.checkpoint_path)
