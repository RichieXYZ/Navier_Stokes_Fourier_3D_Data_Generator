import math
import torch
from Solver_plots import *

s = 32
record_steps = 200

# ---------------------------------------------------
# Force Field - Arnold-Beltrami-Childress
# ---------------------------------------------------
def force_field(s, t, A=0.5, B=0.5, C=0.5, k=23, omega=1):
    '''
    Arnold-Beltrami-Childress forcing : div-free by construction
    '''

    x = torch.linspace(0, 1, s + 1)
    x = x[0:-1]
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')

    # assume X,Y,Z range [0,s)
    argx = 2 * math.pi * ((k * X / s) + omega * t)
    argy = 2 * math.pi * ((k * Y / s) + omega * t)
    argz = 2 * math.pi * ((k * Z / s) + omega * t)
    f_x = A * torch.sin(argz) + C * torch.cos(argy)
    f_y = B * torch.sin(argx) + A * torch.cos(argz)
    f_z = C * torch.sin(argy) + B * torch.cos(argx)
    f = torch.stack((f_x, f_y, f_z), dim=0)
    zero = torch.zeros_like(X)
    g = -1 * torch.ones_like(zero)
    grav = torch.stack((zero, zero, g), dim=0)

    f += grav

    return f


force_tensor = torch.zeros(size=(3, s, s, s, record_steps), device=torch.device("cpu"))
for i in range(record_steps):
     t = i * 1 / record_steps
     force_tensor[..., i] = force_field(s, t)

print("Force tensor shape : ", force_tensor.shape)
vector_field_animation_3d(torch.tensor(force_tensor.permute(4, 0, 1, 2, 3)).cpu().squeeze(), s, step=2, normalize=True, field_name="Force")
force_tensor = torch.tensor(force_tensor.permute(4, 0, 1, 2, 3)).cpu().squeeze()
vector_projection_animation(torch.asarray(force_tensor))
plot_3D_vector_field(np.asfarray(force_field(s,0)), s, step=2)

def get_heat_field(s, sigma):
    x = torch.linspace(0, 1, s + 1)
    x = x[0:-1]
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')

    xc, yc, zc = 0.5, 0.5, 0.5
    Q_phys = torch.exp(-((X - xc) ** 2 + (Y - yc) ** 2 + (Z - zc) ** 2) / (2 * sigma ** 2))
    return Q_phys

Q = get_heat_field(s, sigma=0.25)
scalar_field_animation_3d(Q, s, step=2, normalize=True, field_name="Heat")
