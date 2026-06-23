import torch
class Diffusion:
    def __init__(self,noise_steps=1000, beta_start=1e-4, beta_end=0.02,device='cuda'):
        self.noise_steps = noise_steps
        self.device = device
        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    def sample_timesteps(self, batch_size):
            return torch.randint(low=1, high=self.noise_steps, size=(batch_size,), device=self.device)
    def add_noise(self, x0, t):
            sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
            epsilon = torch.randn_like(x0)
            return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * epsilon, epsilon
    @torch.no_grad()
    def sample(self, model, n_samples):
            model.eval()
            x = torch.randn((n_samples, 3, 64, 64), device=self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n_samples) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha_t = self.alpha[t][:, None, None, None]
                alpha_hat_t = self.alpha_hat[t][:, None, None, None]
                beta_t = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
                model.train()
                return x