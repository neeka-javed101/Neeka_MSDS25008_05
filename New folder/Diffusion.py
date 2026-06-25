
import torch

class Diffusion:
    """
    Diffusion model for image generation using DDPM reverse process.
    """
    
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.noise_steps = noise_steps
        self.device = device
        
        # Linear schedule
        self.beta = torch.linspace(beta_start, beta_end, noise_steps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        # Pre-compute for efficiency
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_1_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat)

    def sample_timesteps(self, batch_size):
        """Sample random timesteps for training."""
        return torch.randint(low=0, high=self.noise_steps, size=(batch_size,), device=self.device)

    def add_noise(self, x0, t):
        """Forward diffusion: add noise to image at timestep t."""
        sqrt_alpha_hat_t = self.sqrt_alpha_hat[t]
        sqrt_1_minus_alpha_hat_t = self.sqrt_1_minus_alpha_hat[t]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_hat_t.shape) < len(x0.shape):
            sqrt_alpha_hat_t = sqrt_alpha_hat_t.unsqueeze(-1)
            sqrt_1_minus_alpha_hat_t = sqrt_1_minus_alpha_hat_t.unsqueeze(-1)
        
        epsilon = torch.randn_like(x0)
        x_t = sqrt_alpha_hat_t * x0 + sqrt_1_minus_alpha_hat_t * epsilon
        
        return x_t, epsilon

    @torch.no_grad()
    def sample(self, model, n_samples):
        """Generate images using reverse diffusion."""
        model.eval()
        
        # Start from pure noise
        x = torch.randn((n_samples, 3, 64, 64), device=self.device)
        
        # Reverse diffusion process
        for t in reversed(range(1, self.noise_steps)):
            t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=self.device)
            
            # Model predicts noise
            
            predicted_noise = model(x, t_tensor)
            
            # Compute mean
            sqrt_alpha_t = torch.sqrt(self.alpha[t])
            sqrt_alpha_hat_t = self.sqrt_alpha_hat[t]
            sqrt_1_minus_alpha_hat_t = self.sqrt_1_minus_alpha_hat[t]
            
            # Reshape for broadcasting
            sqrt_alpha_t = sqrt_alpha_t.view(1, 1, 1, 1)
            sqrt_alpha_hat_t = sqrt_alpha_hat_t.view(1, 1, 1, 1)
            sqrt_1_minus_alpha_hat_t = sqrt_1_minus_alpha_hat_t.view(1, 1, 1, 1)
            
            # Mean of reverse process
            mean = (1 / sqrt_alpha_t) * (x - ((1 - self.alpha[t]) / sqrt_1_minus_alpha_hat_t) * predicted_noise)
            
            # Variance
            if t > 1:
                alpha_hat_t_prev = self.alpha_hat[t - 1]
                posterior_variance = self.beta[t] * (1 - alpha_hat_t_prev) / (1 - self.alpha_hat[t])
                std = torch.sqrt(posterior_variance)
                noise = torch.randn_like(x)
                x = mean + std.view(1, 1, 1, 1) * noise
            else:
                x = mean
        
        x=torch.clamp(x, -1, 1)  # Ensure values are in [-1, 1]
        return x
def visualize_forward(self, x0, steps=[0, 200, 400, 600, 800, 999]):
    """Show noising at specific timesteps — needed for report."""
    images = []
    for step in steps:
        t = torch.tensor([step], device=self.device)
        x_t, _ = self.add_noise(x0, t)
        images.append(x_t)
    return images