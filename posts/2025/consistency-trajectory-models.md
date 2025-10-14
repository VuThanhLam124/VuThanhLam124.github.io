# Consistency Trajectory Models (CTM)

**Ngày đăng:** 27/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Diffusion Models

## Giới thiệu

CTM improves consistency models bằng cách học **entire trajectories** thay vì single endpoint.

## Innovation

Model predicts không chỉ $x_0$ mà whole path $\{x_t\}_{t \in [0,T]}$.

## Training

```python
def ctm_loss(model, x):
    t1, t2 = sample_timestep_pair()
    x_t1 = diffuse(x, t1)
    x_t2 = diffuse(x, t2)
    
    # Predict trajectories
    traj1 = model(x_t1, t1)
    traj2 = model(x_t2, t2)
    
    # Consistency across trajectories
    return ((traj1[t2] - traj2[t2]) ** 2).mean()
```

## Tài liệu

- Kim et al. "Consistency Trajectory Models" 2023

**Tags:** #CTM #TrajectoryModels #ConsistencyModels
