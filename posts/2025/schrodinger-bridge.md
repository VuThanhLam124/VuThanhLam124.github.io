# Schrödinger Bridge: Connecting Two Distributions

**Ngày đăng:** 18/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Flow-based Models, Stochastic Processes

## 📋 Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Schrödinger Bridge Problem](#schrödinger-bridge-problem)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Training Methods](#training-methods)
5. [Implementation](#implementation)
6. [Applications](#applications)

---

## Giới thiệu

**Schrödinger Bridge** là elegant solution to problem: find most likely **stochastic process** connecting two distributions.

**Key concepts:**
- Entropy-regularized optimal transport
- Stochastic differential equations
- Iterative Proportional Fitting (IPF)
- Diffusion bridges

## 1. Problem Setup

Given distributions $p_0$ và $p_1$, find stochastic process minimizing:

$$
\min_{p_{[0,1]}} \mathbb{KL}(p_{[0,1]} \| \pi_{[0,1]})
$$

subject to:
- $p_t|_{t=0} = p_0$
- $p_t|_{t=1} = p_1$

với $\pi_{[0,1]}$ là reference Brownian motion.

**Solution:** Schrödinger Bridge - unique optimal process.

## 2. Mathematical Foundation

### 2.1 Forward-Backward SDEs

Schrödinger Bridge characterized by coupled SDEs:

**Forward SDE:**
$$
dx_t = b_t^f(x_t) dt + \sigma dW_t
$$

**Backward SDE:**
$$
dx_t = b_t^b(x_t) dt + \sigma d\bar{W}_t
$$

với drifts related to scores:
$$
b_t^f(x) = \sigma^2 \nabla_x \log p_t(x)
$$

### 2.2 Connection to Optimal Transport

Schrödinger Bridge = **entropic regularization** of optimal transport:

$$
\min_{\gamma \in \Pi(p_0, p_1)} \int c(x, y) d\gamma(x,y) + \epsilon \text{KL}(\gamma \| \pi_0 \otimes \pi_1)
$$

As $\epsilon \to 0$, recovers classical optimal transport.

## 3. Iterative Proportional Fitting (IPF)

### 3.1 Algorithm

**Key idea:** Alternate between forward và backward processes.

```
Initialize: p^0_{[0,1]} = reference process

For k = 0, 1, 2, ...:
    # Forward pass: fix endpoint at p1
    p^{k+1/2}_{[0,1]} = condition p^k on ending at p1
    
    # Backward pass: fix starting point at p0
    p^{k+1}_{[0,1]} = condition p^{k+1/2} on starting from p0
```

**Convergence:** Exponentially fast to Schrödinger Bridge!

### 3.2 Score-Based Implementation

Learn scores $s_\theta^f(x,t)$ và $s_\theta^b(x,t)$:

**Forward score:**
$$
\min_\theta \mathbb{E}_{x_t \sim p_t}\left[\|s_\theta^f(x_t,t) - \nabla_x \log p_t(x_t)\|^2\right]
$$

**Backward score:**
$$
\min_\phi \mathbb{E}_{x_t \sim q_t}\left[\|s_\phi^b(x_t,t) - \nabla_x \log q_t(x_t)\|^2\right]
$$

## 4. Implementation

### 4.1 Score Network

```python
import torch
import torch.nn as nn

class ScoreNetwork(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # Score network
        self.net = nn.Sequential(
            nn.Linear(dim + 128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, t):
        """
        x: [B, D] - state
        t: [B, 1] - time
        Returns: score ∇log p(x,t)
        """
        t_emb = self.time_embed(t)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)
```

### 4.2 IPF Training Loop

```python
def train_schrodinger_bridge(
    p0_samples,  # Source distribution
    p1_samples,  # Target distribution
    num_ipf_iterations=10,
    num_epochs_per_iteration=100,
    sigma=1.0
):
    # Initialize forward and backward score networks
    score_forward = ScoreNetwork(dim=p0_samples.shape[1])
    score_backward = ScoreNetwork(dim=p0_samples.shape[1])
    
    for ipf_iter in range(num_ipf_iterations):
        print(f"\n=== IPF Iteration {ipf_iter + 1} ===")
        
        # === Forward Pass ===
        print("Training forward score...")
        optimizer_f = torch.optim.Adam(score_forward.parameters(), lr=1e-4)
        
        for epoch in range(num_epochs_per_iteration):
            # Sample from current forward process
            x0 = p0_samples[torch.randint(len(p0_samples), (256,))]
            t = torch.rand(256, 1)
            
            # Simulate forward SDE (using Euler-Maruyama)
            x_t = simulate_forward_sde(
                x0, t, score_backward, sigma
            )
            
            # Denoising score matching loss
            noise = torch.randn_like(x_t)
            x_noisy = x_t + 0.01 * noise
            score_pred = score_forward(x_noisy, t)
            score_target = -noise / 0.01
            
            loss = torch.mean((score_pred - score_target) ** 2)
            
            optimizer_f.zero_grad()
            loss.backward()
            optimizer_f.step()
        
        # === Backward Pass ===
        print("Training backward score...")
        optimizer_b = torch.optim.Adam(score_backward.parameters(), lr=1e-4)
        
        for epoch in range(num_epochs_per_iteration):
            # Sample from current backward process
            x1 = p1_samples[torch.randint(len(p1_samples), (256,))]
            t = torch.rand(256, 1)
            
            # Simulate backward SDE
            x_t = simulate_backward_sde(
                x1, t, score_forward, sigma
            )
            
            # Denoising score matching loss
            noise = torch.randn_like(x_t)
            x_noisy = x_t + 0.01 * noise
            score_pred = score_backward(x_noisy, t)
            score_target = -noise / 0.01
            
            loss = torch.mean((score_pred - score_target) ** 2)
            
            optimizer_b.zero_grad()
            loss.backward()
            optimizer_b.step()
    
    return score_forward, score_backward

def simulate_forward_sde(x0, t_target, score_net, sigma, num_steps=50):
    """Simulate forward SDE using Euler-Maruyama"""
    x = x0
    dt = t_target / num_steps
    
    for i in range(num_steps):
        t = torch.ones(x.shape[0], 1) * (i * dt)
        
        # Drift from score
        with torch.no_grad():
            drift = sigma**2 * score_net(x, t)
        
        # Euler-Maruyama step
        x = x + drift * dt + sigma * torch.sqrt(dt) * torch.randn_like(x)
    
    return x

def simulate_backward_sde(x1, t_target, score_net, sigma, num_steps=50):
    """Simulate backward SDE"""
    x = x1
    dt = (1 - t_target) / num_steps
    
    for i in range(num_steps):
        t = torch.ones(x.shape[0], 1) * (1 - i * dt)
        
        # Backward drift
        with torch.no_grad():
            drift = sigma**2 * score_net(x, t)
        
        # Backward Euler-Maruyama
        x = x - drift * dt + sigma * torch.sqrt(dt) * torch.randn_like(x)
    
    return x
```

### 4.3 Sampling

```python
def sample_schrodinger_bridge(score_forward, x0, sigma=1.0, num_steps=1000):
    """
    Generate samples by following forward SDE
    """
    x = x0
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.ones(x.shape[0], 1) * (i * dt)
            
            # Forward drift
            drift = sigma**2 * score_forward(x, t)
            
            # SDE step
            x = x + drift * dt + sigma * torch.sqrt(dt) * torch.randn_like(x)
    
    return x
```

## 5. Diffusion Schrödinger Bridge (DSB)

### 5.1 Simplified Approach

Recent work simplifies training using **diffusion models**:

```python
class DiffusionSchrodingerBridge(nn.Module):
    def __init__(self, dim, hidden_dim=512):
        super().__init__()
        self.score_net = ScoreNetwork(dim, hidden_dim)
    
    def forward_process(self, x0, x1, t):
        """
        Bridge process: interpolate between x0 and x1
        """
        # Mean of bridge
        mean = (1 - t) * x0 + t * x1
        
        # Variance of bridge
        var = t * (1 - t) * self.sigma**2
        
        return mean, var
    
    def loss(self, x0, x1):
        """Training loss"""
        t = torch.rand(x0.shape[0], 1)
        
        # Sample from bridge
        mean, var = self.forward_process(x0, x1, t)
        x_t = mean + torch.sqrt(var) * torch.randn_like(x0)
        
        # Score matching
        score_pred = self.score_net(x_t, t)
        score_target = -(x_t - mean) / var
        
        return torch.mean((score_pred - score_target) ** 2)
```

## 6. Applications

### 6.1 Image-to-Image Translation

```python
# Learn bridge between two image domains
source_images = load_domain_A()
target_images = load_domain_B()

score_f, score_b = train_schrodinger_bridge(
    source_images, target_images
)

# Translate
translated = sample_schrodinger_bridge(score_f, source_test)
```

### 6.2 Trajectory Optimization

```python
# Find most likely paths for robot motion
initial_states = sample_start_configurations()
goal_states = sample_goal_configurations()

bridge = train_schrodinger_bridge(initial_states, goal_states)

# Generate smooth trajectories
trajectories = sample_schrodinger_bridge(bridge, initial_states)
```

## Kết luận

Schrödinger Bridge provides principled framework for:

✅ **Stochastic optimal transport**  
✅ **Natural trajectory generation**  
✅ **Entropy regularization benefits**  
✅ **Flexible conditioning**

While more complex than deterministic flows, stochasticity can be beneficial for many applications.

## Tài liệu tham khảo

1. Schrödinger, E. (1932). "Über die Umkehrung der Naturgesetze"
2. De Bortoli, V., et al. (2021). "Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling"
3. Chen, T., et al. (2021). "Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory"

---

**Tags:** #SchrodingerBridge #OptimalTransport #StochasticProcesses #GenerativeModels
