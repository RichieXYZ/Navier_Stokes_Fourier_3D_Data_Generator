import torch.nn.functional as F
import math
import copy
from timeit import default_timer
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("mps")

# ==========================================================
# Gaussian Random Field
# ==========================================================

class GaussianRF(object):

    def __init__(self, dim, size, alpha=2.5, tau=7, sigma=None, device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))

        k_max = size // 2

        wavenumbers = torch.cat(
            (torch.arange(0, k_max, device=device),
             torch.arange(-k_max, 0, device=device)), 0
        )

        if dim == 3:
            k = wavenumbers.repeat(size, size, 1)
            kx = k.transpose(1, 2)
            ky = k
            kz = k.transpose(0, 2)

            self.sqrt_eig = (
                size ** 3
                * math.sqrt(2.0)
                * sigma
                * ((4 * (math.pi ** 2) * (kx ** 2 + ky ** 2 + kz ** 2) + tau ** 2)
                   ** (-alpha / 2.0))
            )

            self.sqrt_eig[0, 0, 0] = 0.0

        self.size = (size,) * dim

    def sample(self, N):
        coeff = torch.randn(N, *self.size,
                            dtype=torch.cfloat,
                            device=self.device)
        coeff = self.sqrt_eig * coeff
        return torch.fft.ifftn(
            coeff,
            dim=list(range(-1, -self.dim - 1, -1))
        ).real


# ==========================================================
# Utilities
# ==========================================================

def normalize_std(x):
    # Gaussian RF mean ≈ 0, only normalize variance
    return x / x.std()

def downsample_mean(x):
    # x shape: [B, 1, 512, 512, 512]
    return F.avg_pool3d(x, kernel_size=8, stride=8)

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def nonlinear_grf_sample(GRF, batch_size, beta=0.5):
    u = GRF.sample(batch_size).unsqueeze(1)
    u = u / u.std()

    v = u + beta * u**3

    low = F.avg_pool3d(v, 8, 8)
    return low, v

# ==========================================================
# Setup
# ==========================================================

batch_size = 2
resolution = 256

GRF = GaussianRF(3, resolution, device=device)

from Project.Super_resolution.SR_Model import UNetSR3d

model = UNetSR3d(
    in_channels=1,
    out_channels=1,
    resolution_factor=8,
    encoder_depth=3,
    decoder_channels=[32, 16, 8],
).to(device)

print("Parameters:", sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.8,
    patience=3
)

def gradient_loss(pred, target):
    dx = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
    dx_t = torch.abs(target[:, :, 1:] - target[:, :, :-1])
    return (dx - dx_t).abs().mean()

criterion = nn.MSELoss()

# ==========================================================
# Training
# ==========================================================

max_epoch = 500
early_stop_patience = 25

history = defaultdict(list)
best_val_loss = np.inf
patience = 0
best_state_dict = None

start = default_timer()

print("Start Training...")

for epoch in range(max_epoch):

    # =======================
    # TRAIN
    # =======================
    model.train()
    t1 = default_timer()

    x, w0 = nonlinear_grf_sample(GRF, batch_size)

    optimizer.zero_grad()
    out = model(x)
    train_loss = ((out - w0) ** 2).mean() + 0.1*gradient_loss(out, w0)
    train_loss.backward()
    optimizer.step()

    # =======================
    # VALIDATION
    # =======================
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        x_val, w0_val = nonlinear_grf_sample(GRF, batch_size)
        out_val = model(x_val)
        val_loss = ((out_val - w0_val) ** 2).mean() + 0.1*gradient_loss(out_val, w0_val)

    scheduler.step(val_loss)

    # =======================
    # LOGGING
    # =======================
    history["train_losses"].append(train_loss.item())
    history["val_losses"].append(val_loss.item())
    history["lrs"].append(get_lr(optimizer))

    t2 = default_timer()
    print(f"Epoch {epoch:4d} | "
          f"Train: {train_loss.item():.6f} | "
          f"Val: {val_loss.item():.6f} | "
          f"LR: {get_lr(optimizer):.2e}"
          f"| Time : ", np.round(t2 - t1, 2), "s")

    # =======================
    # EARLY STOP + SAVE BEST
    # =======================
    if val_loss < best_val_loss:
        print('Loss Decreasing.. {:.5f} >> {:.5f} '.format(best_val_loss, val_loss))
        best_val_loss = val_loss
        best_state_dict = copy.deepcopy(model.state_dict())
        patience = 0
    else:
        patience += 1
        print(f'Loss Not Decrease for {patience} time')

    if patience >= early_stop_patience:
        print("Early stopping triggered.")
        break

# ==========================================================
# SAVE BEST MODEL
# ==========================================================

if best_state_dict is not None:
    torch.save(best_state_dict, "UNet3D.pt")
    print("Best model saved.")

total_time = (default_timer() - start) / 60
print(f"Total training time: {total_time:.2f} minutes")

def plot_results(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # (rows, cols)
    plt.title(model_name)

    # Plot data
    axes[0].plot(history["test_losses"], label='val')
    axes[0].plot(history["train_losses"], label='train')
    axes[0].set_yscale("log")
    axes[0].set_title(model_name + ' Loss')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(history["lrs"])
    axes[1].set_title(model_name + ' LR ')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('LR')
    axes[1].legend()
    axes[1].grid()

    plt.show()

plot_results(history, "UNet3D")

