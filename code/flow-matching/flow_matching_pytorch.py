import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class VectorField(nn.Module):
    """
    Neural network để học vector field cho Flow Matching
    """
    def __init__(self, dim, hidden_dim=256, time_embed_dim=64, num_layers=4):
        super().__init__()
        self.dim = dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU()
        )
        
        # Main vector field network
        layers = []
        input_dim = dim + time_embed_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            ])
        
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x, t):
        """
        Args:
            x: Data points, shape (batch_size, dim)
            t: Time steps, shape (batch_size,)
        Returns:
            Vector field, shape (batch_size, dim)
        """
        batch_size = x.size(0)
        
        # Time embedding
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_embed = self.time_embed(t)
        
        # Concatenate x and time embedding
        input_vec = torch.cat([x, t_embed], dim=-1)
        
        return self.net(input_vec)

class FlowMatching:
    """
    Complete Flow Matching implementation
    """
    def __init__(self, dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.dim = dim
        self.device = device
        self.model = VectorField(dim).to(device)
        self.optimizer = None
        
    def setup_optimizer(self, lr=1e-3, weight_decay=1e-5):
        """Setup optimizer with learning rate scheduling"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
    
    def flow_matching_loss(self, x0, x1, t):
        """
        Compute Flow Matching loss
        
        Args:
            x0: Source distribution (noise), shape (batch_size, dim)
            x1: Target distribution (data), shape (batch_size, dim)
            t: Time steps, shape (batch_size,)
        """
        # Linear interpolation path
        x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        
        # Target vector field (derivative of interpolation)
        u_t = x1 - x0
        
        # Predicted vector field
        v_t = self.model(x_t, t)
        
        # Flow matching loss
        loss = F.mse_loss(v_t, u_t)
        return loss, x_t, v_t, u_t
    
    def train_step(self, real_data):
        """Single training step"""
        batch_size = real_data.size(0)
        
        # Sample source noise
        x0 = torch.randn_like(real_data)
        x1 = real_data
        
        # Sample time steps uniformly
        t = torch.rand(batch_size, device=self.device)
        
        # Compute loss
        loss, x_t, v_t, u_t = self.flow_matching_loss(x0, x1, t)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'grad_norm': torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
        }
    
    def train(self, dataloader, num_epochs, log_interval=100, save_path=None):
        """
        Train Flow Matching model
        """
        if self.optimizer is None:
            self.setup_optimizer()
        
        self.model.train()
        train_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (real_data,) in enumerate(dataloader):
                real_data = real_data.to(self.device)
                
                # Training step
                metrics = self.train_step(real_data)
                epoch_loss += metrics['loss']
                
                # Logging
                if (batch_idx + 1) % log_interval == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, '
                          f'Batch {batch_idx+1}/{len(dataloader)}, '
                          f'Loss: {metrics["loss"]:.6f}, '
                          f'Grad Norm: {metrics["grad_norm"]:.6f}')
            
            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save checkpoint
            if save_path and (epoch + 1) % 100 == 0:
                self.save_checkpoint(f"{save_path}_epoch_{epoch+1}.pt", epoch, avg_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_loss:.6f}')
        
        return train_losses
    
    def sample(self, num_samples, num_steps=100, method='euler'):
        """
        Generate samples using trained Flow Matching model
        
        Args:
            num_samples: Number of samples to generate
            num_steps: Number of integration steps
            method: Integration method ('euler' or 'adaptive')
        """
        self.model.eval()
        
        if method == 'euler':
            return self._sample_euler(num_samples, num_steps)
        elif method == 'adaptive':
            return self._sample_adaptive(num_samples)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def _sample_euler(self, num_samples, num_steps):
        """Euler method for ODE integration"""
        with torch.no_grad():
            # Start from Gaussian noise
            x = torch.randn(num_samples, self.dim, device=self.device)
            
            dt = 1.0 / num_steps
            
            # Solve ODE using Euler method
            for i in range(num_steps):
                t = torch.full((num_samples,), i * dt, device=self.device)
                
                # Get vector field
                v = self.model(x, t)
                
                # Update x
                x = x + v * dt
        
        return x
    
    def _sample_adaptive(self, num_samples, tol=1e-5):
        """Adaptive step size integration using scipy"""
        def ode_func(t, x_flat):
            x_tensor = torch.from_numpy(x_flat.reshape(num_samples, self.dim)).float().to(self.device)
            t_tensor = torch.full((num_samples,), t, device=self.device)
            
            with torch.no_grad():
                v = self.model(x_tensor, t_tensor)
            
            return v.cpu().numpy().flatten()
        
        # Initial condition
        x0 = torch.randn(num_samples, self.dim).numpy().flatten()
        
        # Solve ODE
        sol = solve_ivp(ode_func, [0, 1], x0, rtol=tol, atol=tol, method='RK45')
        
        return torch.from_numpy(sol.y[:, -1].reshape(num_samples, self.dim))
    
    def compute_likelihood(self, x, num_steps=100):
        """
        Compute likelihood using the change of variables formula
        """
        self.model.eval()
        
        with torch.no_grad():
            batch_size = x.size(0)
            
            # Start from data points
            x_t = x.clone()
            log_det = torch.zeros(batch_size, device=self.device)
            
            dt = 1.0 / num_steps
            
            # Reverse integration
            for i in range(num_steps):
                t = torch.full((batch_size,), 1 - i * dt, device=self.device)
                
                # Get vector field
                v = self.model(x_t, t)
                
                # Compute divergence (approximation)
                div_v = self._compute_divergence(x_t, t)
                
                # Update log determinant
                log_det = log_det - div_v * dt
                
                # Update x (reverse direction)
                x_t = x_t - v * dt
            
            # Prior likelihood (standard Gaussian)
            log_prior = -0.5 * torch.sum(x_t ** 2, dim=1) - 0.5 * self.dim * np.log(2 * np.pi)
            
            # Total likelihood
            log_likelihood = log_prior + log_det
        
        return log_likelihood
    
    def _compute_divergence(self, x, t):
        """Compute divergence of vector field (approximation using finite differences)"""
        eps = 1e-4
        batch_size = x.size(0)
        div = torch.zeros(batch_size, device=self.device)
        
        for i in range(self.dim):
            x_plus = x.clone()
            x_minus = x.clone()
            x_plus[:, i] += eps
            x_minus[:, i] -= eps
            
            v_plus = self.model(x_plus, t)
            v_minus = self.model(x_minus, t)
            
            div += (v_plus[:, i] - v_minus[:, i]) / (2 * eps)
        
        return div
    
    def save_checkpoint(self, path, epoch, loss):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss
        }, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {path}")
        return checkpoint['epoch'], checkpoint['loss']

# Utility functions
def visualize_2d_flow(model, device, grid_size=50, t_steps=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    """
    Visualize 2D flow field at different time steps
    """
    model.eval()
    
    # Create grid
    x_range = torch.linspace(-3, 3, grid_size)
    y_range = torch.linspace(-3, 3, grid_size)
    X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
    
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, t in enumerate(t_steps):
            t_tensor = torch.full((grid_points.size(0),), t, device=device)
            v = model(grid_points, t_tensor)
            
            U = v[:, 0].reshape(grid_size, grid_size).cpu()
            V = v[:, 1].reshape(grid_size, grid_size).cpu()
            
            ax = axes[i]
            ax.quiver(X.numpy(), Y.numpy(), U.numpy(), V.numpy(), alpha=0.7)
            ax.set_title(f't = {t:.1f}')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_toy_dataset(dataset_type='moons', n_samples=1000):
    """
    Create toy 2D datasets for testing
    """
    if dataset_type == 'moons':
        from sklearn.datasets import make_moons
        data, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    elif dataset_type == 'circles':
        from sklearn.datasets import make_circles
        data, _ = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    elif dataset_type == 'blobs':
        from sklearn.datasets import make_blobs
        data, _ = make_blobs(n_samples=n_samples, centers=4, n_features=2, 
                            cluster_std=0.5, random_state=42)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return torch.from_numpy(data).float()

# Example usage
if __name__ == "__main__":
    # Create toy dataset
    data = create_toy_dataset('moons', 2000)
    
    # Create data loader
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Initialize Flow Matching
    fm = FlowMatching(dim=2)
    
    # Train
    losses = fm.train(dataloader, num_epochs=500, log_interval=50)
    
    # Generate samples
    samples = fm.sample(1000, num_steps=100)
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
    plt.title('Original Data')
    
    plt.subplot(1, 3, 2)
    plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.6)
    plt.title('Generated Samples')
    
    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()