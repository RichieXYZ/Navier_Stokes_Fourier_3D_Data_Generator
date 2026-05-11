import torch
import matplotlib.animation as animation
from torch import nn
import math
import torch.nn.functional as F
from SR_Model import UNetSR3d_V0
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real

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

s = 256
device = torch.device("cpu")
GRF = GaussianRF(3, s, alpha=2, tau=7, device=device)
# Model
model = UNetSR3d_V0(
    in_channels=1,
    out_channels=1,
    resolution_factor=8,
    encoder_depth=3,
    decoder_channels=[64, 32, 16],
)
ckpt_path = "/Users/riccardo/PycharmProjects/Fluid_Simulations/Project/Super_resolution/UNet3D.pt"
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.eval()

# Testing
# -------------------------------------------------------------------------------
batch_size = 1
Loss = nn.MSELoss()

def plot_prediction_error3d(gt, pred, model_name):
    artists = []
    # Plot results
    # ------------------------------------------
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    # fig.subplots_adjust(hspace=0., wspace=0.3)
    #fig.title("FLSTM 3D, param = ", count_params(model))

    ax[0, 0].title.set_text('GT - mean x')
    ax[1, 0].title.set_text('GT - mean y')
    ax[2, 0].title.set_text('GT - mean z')

    ax[0, 1].title.set_text('Pred - mean x')
    ax[1, 1].title.set_text('Pred - mean y')
    ax[2, 1].title.set_text('Pred - mean z')

    ax[0, 2].title.set_text('Err - mean x')
    ax[1, 2].title.set_text('Err - mean y')
    ax[2, 2].title.set_text('Err - mean z')

    gt_x = []
    pred_x = []
    gt_y = []
    pred_y = []
    gt_z = []
    pred_z = []
    err_x = []
    err_y = []
    err_z = []

    for i in range(len(pred)):
        gt_frame = gt[i, :, :, :].clone()
        pred_frame = pred[i].clone()
        gt_x.append(gt_frame.mean(dim=0))
        pred_x.append(pred_frame.mean(dim=0))
        gt_y.append(gt_frame.mean(dim=1))
        pred_y.append(pred_frame.mean(dim=1))
        gt_z.append(gt_frame.mean(dim=2))
        pred_z.append(pred_frame.mean(dim=2))
        err_x.append(torch.abs(gt_frame.mean(dim=0)-pred_frame.mean(dim=0)))
        err_y.append(torch.abs(gt_frame.mean(dim=1) - pred_frame.mean(dim=1)))
        err_z.append(torch.abs(gt_frame.mean(dim=2) - pred_frame.mean(dim=2)))

    max_err_x = torch.tensor(np.asarray(err_x)).max()
    min_err_x = torch.tensor(np.asarray(err_x)).min()
    max_err_y = torch.tensor(np.asarray(err_y)).max()
    min_err_y = torch.tensor(np.asarray(err_y)).min()
    max_err_z = torch.tensor(np.asarray(err_z)).max()
    min_err_z = torch.tensor(np.asarray(err_z)).min()

    for i in range(len(pred)):
        gt_x_plot = ax[0, 0].imshow(np.asarray(gt_x[i]), cmap='inferno')
        gt_y_plot = ax[1, 0].imshow(np.asarray(gt_y[i]), cmap='inferno')
        gt_z_plot = ax[2, 0].imshow(np.asarray(gt_z[i]), cmap='inferno')

        pred_x_plot = ax[0, 1].imshow(np.asarray(pred_x[i]), cmap='inferno')
        pred_y_plot = ax[1, 1].imshow(np.asarray(pred_y[i]), cmap='inferno')
        pred_z_plot = ax[2, 1].imshow(np.asarray(pred_z[i]), cmap='inferno')

        err_frame_x = (err_x[i] - min_err_x) / (max_err_x - min_err_x)
        err_frame_y = (err_y[i] - min_err_y) / (max_err_y - min_err_y)
        err_frame_z = (err_z[i] - min_err_z) / (max_err_z - min_err_z)
        err_x_plot = ax[0, 2].imshow(np.asarray(err_frame_x), cmap='grey')
        err_y_plot = ax[1, 2].imshow(np.asarray(err_frame_y), cmap='grey')
        err_z_plot = ax[2, 2].imshow(np.asarray(err_frame_z), cmap='grey')

        artists.append([gt_x_plot, gt_y_plot, gt_z_plot,
                        pred_x_plot, pred_y_plot, pred_z_plot,
                        err_x_plot, err_y_plot, err_z_plot])

    # Add colorbar for error only
    cbar1 = fig.colorbar(err_x_plot, ax=ax[0, 2], orientation='vertical')
    cbar1.set_label("Error magnitude")
    cbar2 = fig.colorbar(err_y_plot, ax=ax[1, 2], orientation='vertical')
    cbar2.set_label("Error magnitude")
    cbar3 = fig.colorbar(err_z_plot, ax=ax[2, 2], orientation='vertical')
    cbar3.set_label("Error magnitude")

    anim = animation.ArtistAnimation(fig=fig, artists=artists, interval=200)
    anim.save(model_name + "_Results.mp4")
    plt.show()

def scalar_projection(x, pred, gt, normalize=False):

    fig, ax = plt.subplots(3, 4, figsize=(11, 8))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    titles = [
        ['Input - mean x', 'Model - mean x', 'GT - mean x', 'Err - mean x'],
        ['Input - mean y', 'Model - mean y', 'GT - mean y', 'Err - mean y'],
        ['Input - mean z', 'Model - mean z', 'GT - mean z', 'Err - mean z']
    ]

    for i in range(3):
        for j in range(4):
            ax[i, j].set_title(titles[i][j])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    # ----------------------------
    # Compute projections
    # ----------------------------
    Xx, Xy, Xz = x.mean(0), x.mean(1), x.mean(2)
    Fx, Fy, Fz = gt.mean(0), gt.mean(1), gt.mean(2)
    Px, Py, Pz = pred.mean(0), pred.mean(1), pred.mean(2)

    Ex = torch.abs(Fx - Px)
    Ey = torch.abs(Fy - Py)
    Ez = torch.abs(Fz - Pz)

    # ----------------------------
    # Normalize fields if requested
    # ----------------------------
    if normalize:
        vmin = gt.min().item()
        vmax = gt.max().item()
    else:
        vmin, vmax = None, None

    # ----------------------------
    # Error normalization (per slice)
    # ----------------------------
    def normalize_err(E):
        Emin, Emax = E.min(), E.max()
        return (E - Emin) / (Emax - Emin + 1e-12)

    Exn = normalize_err(Ex)
    Eyn = normalize_err(Ey)
    Ezn = normalize_err(Ez)

    # ----------------------------
    # Plot
    # ----------------------------
    fields = [
        [Xx, Px, Fx, Exn],
        [Xy, Py, Fy, Eyn],
        [Xz, Pz, Fz, Ezn],
    ]

    cm_field = 'inferno'
    cm_error = 'gray'

    for i in range(3):
        for j in range(4):
            data = fields[i][j].detach().cpu().numpy()

            if j < 3:
                im = ax[i, j].imshow(data, cmap=cm_field,
                                     vmin=vmin, vmax=vmax)
            else:
                im = ax[i, j].imshow(data, cmap=cm_error,
                                     vmin=0.0, vmax=1.0)

                # Proper full-size colorbar
                divider = make_axes_locatable(ax[i, j])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label("Normalized Error")

    plt.show()


print("\nTesting ...")
with torch.no_grad():
    for i in range(10):
        x, w0 = nonlinear_grf_sample(GRF, batch_size)

        out = model(x)
        mae = ((out - w0) ** 2).mean()
        print(mae)
        out, w0 = out.squeeze().detach().cpu(), w0.detach().squeeze().cpu()
        scalar_projection(x.detach().squeeze().cpu(), out, w0)