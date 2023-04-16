import torch
import tqdm.notebook as tqdm

class Diffusion:
    def __init__(self, image_size = (32, 32), num_steps = 1000, beta_start = 1e-4, beta_end = 0.02, device = "cuda"):
        self.width = image_size[0]
        self.height = image_size[1]
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.prepare_noise_schedule()

    def prepare_noise_schedule(self):
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.num_steps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)


    def forward(self, x, t):
        alpha_hat_t = self.alpha_hat[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Add three new dimensions
        eps = torch.randn_like(x)
        return torch.sqrt(alpha_hat_t) * x + torch.sqrt(1 - alpha_hat_t) * eps, eps

    def sample_n_images(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.height, self.width)).to(self.device)
            for i in tqdm(reversed(range(1, self.num_steps)), position=0):
                t = torch.full((n,), i, dtype=torch.long, device=self.device)
                predicted_noise = model(x, t)

                alpha = self.alpha[t].view(-1, 1, 1, 1)
                alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
                beta = self.beta[t].view(-1, 1, 1, 1)

                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                x = (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) / torch.sqrt(alpha) + torch.sqrt(beta) * noise

        model.train()

        x = x.clamp(-1, 1).mul(0.5).add(0.5).mul(255).type(torch.uint8)
        return x

