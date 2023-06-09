import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
class Diffusion(nn.Module):
    def __init__(self, cosine = False, image_size = (32, 32), num_steps = 1000, beta_start = 1e-4, beta_end = 0.02, device = "cuda"):
        super().__init__()
        self.width = image_size[0]
        self.height = image_size[1]
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.num_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0).to(device)
        self.device = device
        self.cosine = cosine
      #  self.senoise_schedule()
    def noise_schedule(self):
        print("noise schedule INIT ", self.device)
        if self.cosine:
            print("Using Cosine Scheduler")
            # following the equation from Improved Denoising Diffusion Probabilistic Models
            s = 8e-3 
            timesteps = torch.arange(self.num_steps + 1, dtype = torch.float64, device = self.device) / self.num_steps + s
            alphas = torch.cos(timesteps / (1 + s) * np.pi / 2).pow(2)
            self.alpha_hat = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            self.betas = torch.clip(betas, min = 0, max = 0.999)
            self.alphas = 1. - self.betas 
            self.betas = self.betas.float()
            self.alphas = self.alphas.float()
            self.alpha_hat = self.alpha_hat.float()
            if self.device == torch.device('cuda:0'):
                print(self.betas, self.alphas, self.alpha_hat)
        else:
            self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, device = self.device)
            self.alphas = 1. - self.betas
            self.alpha_hat = torch.cumprod(self.alphas, dim = 0)

    def forward(self, x, t):
       # alpha_hat = torch.cumprod(self.alpha, dim = 0).to(x)
        alpha_hat_t = self.alpha_hat[t]
        alpha_hat_t = alpha_hat_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Add three new dimensions
        eps = torch.randn_like(x)
        return torch.sqrt(alpha_hat_t) * x + torch.sqrt(1 - alpha_hat_t) * eps, eps

    def sample_n_images(self, ae, model, n, x = None):
        model.eval().to(self.device)
        
        with torch.no_grad():
            if x is None:
                x = torch.randn((n, 4, self.height, self.width)).to(self.device)
            t = torch.full((n,), self.num_steps - 1, dtype = torch.long, device = self.device)
            x, noise = self.forward(x, t)
            for i in tqdm(reversed(range(1, self.num_steps)), position=0):
                t = torch.full((n,), i, dtype=torch.long, device=self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t].view(-1, 1, 1, 1)
                alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
                beta = self.beta[t].view(-1, 1, 1, 1)

                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                temp = (1 - alpha) / torch.sqrt(1 - alpha_hat)

                temp2 = temp * predicted_noise
                x = x - temp2
               # x = x - temp * predicted_noise
                x = x / torch.sqrt(alpha) + torch.sqrt(beta) * noise
            x = ae.decode(x)
        model.train()

        x = x.clamp(-1,1).mul(0.5).add(0.5)
        return x

if __name__ == "__main__":
    import argparse
    from unet import UNet
    from vqgan.interface import pretrained_vqgan
    import torchvision
    from data import load_data
    parser = argparse.ArgumentParser('diffusion test')
    parser.add_argument('image_path', type = str)
    parser.add_argument('unet_ckpt_path', type = str)
    args = parser.parse_args()
    model = pretrained_vqgan().eval().cuda()
    for param in model.parameters():
        param.requires_grad = False
    unet = UNet().cuda()
    stdict = torch.load(args.unet_ckpt_path)['state_dict']
    new_dict = dict()
    for key in stdict.keys():
        if 'model' in key and 'perceptual_loss' not in key:
            new_dict[key[6:]] = stdict[key]
    unet.load_state_dict(new_dict)
    diff = Diffusion(device = "cuda")
    diff.device = "cuda:0"
    diff.noise_schedule()
    if args.image_path:
        loader = load_data('wikiart_images',4 , 256,0)
        iterator = iter(loader)
        batch = next(iterator)
        images, _ = batch
        for i in range(4):
            torchvision.utils.save_image(images[i,:,:,:].float(), 'in' + str(i) + '.png')
        with torch.no_grad():
            x = model.encode(images.cuda())
    else: x = None
    x = diff.sample_n_images(model, unet, 4, x = x)
    for i in range(4):
        torchvision.utils.save_image(x[i, :,:,:].float(), 'out' + str(i) + '.png')
