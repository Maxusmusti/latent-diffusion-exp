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

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.train_loader = load_data(args.dataset_path, args.batch_size)

        self.model = UNet().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
        self.loss_fn = nn.MSELoss()
        self.diffusion = Diffusion(image_size=args.image_size, device=self.device)
        
        self.vqgan = pretrained_vqgan()
        self.encode = self.vqgan.encode # returns pre quantized latent vector
        self.decode = self.vqgan.decode # can either force quantization or not (specify with quantize_first)

        self.logger = SummaryWriter(os.path.join("runs", args.run_name))
        self.train_logs = []
        self.val_logs = []

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader)
        l = len(self.train_loader)
        epoch_loss = 0

        for i, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            t = torch.randint(low=1, high=self.diffusion.num_steps, size=(images.shape[0],)).to(self.device)
            encoded_images = self.encode(images)
            x_t, noise = self.diffusion.forward(encoded_images, t)
            predicted_noise = self.model(x_t, t)
            loss = self.loss_fn(noise, predicted_noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
            self.logger.add_scalar("Train/MSE", loss.item(), global_step=epoch * l + i)

        self.train_logs.append(epoch_loss / l)

    def val_epoch(self, epoch):
        # TODO - ADD VAL DATA FOR FUNCTIONING
        self.model.eval()
        pbar = tqdm(self.val_loader)
        l = len(self.val_loader)
        epoch_loss = 0

        with torch.no_grad():
            for i, (images, _) in enumerate(pbar):
                images = images.to(self.device)
                t = torch.randint(low=1, high=self.diffusion.num_steps, size=(images.shape[0],)).to(self.device)
                encoded_images = self.encoder(images).last_hidden_state
                x_t, noise = self.diffusion.forward(encoded_images, t)
                predicted_noise = self.model(x_t, t)
                loss = self.loss_fn(noise, predicted_noise)

                epoch_loss += loss.item()
                pbar.set_postfix(MSE=loss.item())
                self.logger.add_scalar("Val/MSE", loss.item(), global_step=epoch * l + i)

        self.val_logs.append(epoch_loss / l)

    def save_checkpoint(self, epoch):
        checkpoint_dir = os.path.join("models", self.args.run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"ckpt_{epoch}.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_logs': self.train_logs,
            'val_logs': self.val_logs
        }, checkpoint_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_logs = checkpoint['train_logs']
        self.val_logs = checkpoint['val_logs']

    def train(self):
        for epoch in range(self.args.epochs):
            print(f"Starting epoch {epoch}:")
            self.train_epoch(epoch)
            #self.val_epoch(epoch)  TO BE FIXED LATER

            # sampled_images = self.diffusion.sample_n_images(self.model, n=self.args.batch_size)
            # save_images(sampled_images, os.path.join("results", self.args.run_name, f"{epoch}.jpg"))

            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch)

        self.logger.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a DDPM model.")
    parser.add_argument("--run_name", type=str, default="run", help="Name for the run.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training.")
    parser.add_argument("--image_size", type=List[int], default=[32,32], help="Size of the input images.")
    parser.add_argument("--dataset_path", type=str, help="Path to the training dataset.")
    parser.add_argument("--val_dataset_path", type=str, help="Path to the validation dataset.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval to save model checkpoints.")
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
