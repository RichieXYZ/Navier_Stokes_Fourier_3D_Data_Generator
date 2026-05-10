# Libraries
import torch
import numpy as np
import math
from timeit import default_timer
from tqdm import tqdm
from scipy.io import savemat

class Simulator3D(object):

    def __init__(self, resolution, time, delta_t, record_steps):

        self.dim = 3
        self.device = self.set_device()

        # Space parameters
        assert np.log2(resolution) % 1 == 0, f"Grid Size must be power of 2, got N = {resolution}"
        self.s = resolution

        self.size = tuple([self.s for _ in range(self.dim)])
        print("Spatial domain size = ", self.size)

        # Time parameters
        self.time = time
        self.dt = delta_t
        self.record_steps = record_steps

        # Total number of steps
        self.steps = math.ceil(self.time / self.dt)
        print("Total number of steps = ", self.steps)

        self.record_time = math.floor(self.steps / self.record_steps)

        # Spectral Derivation Operators
        # --------------------------------------------------
        K = torch.fft.fftfreq(self.s) * self.s  # length N, includes negative frequencies

        Kx, Ky, Kz = torch.meshgrid(K, K, K[:self.s // 2 + 1], indexing='ij')  # shape [N, N, N/2+1]

        self.K2 = 4 * (math.pi ** 2) * (Kx ** 2 + Ky ** 2 + Kz ** 2).to(self.device)  # shape [N, N, N/2+1]

        self.Kx, self.Ky, self.Kz = Kx.to(self.device), Ky.to(self.device), Kz.to(self.device)

        self.I = torch.eye(3, device=self.device).view(3, 3, 1, 1, 1)

        # De-aliasing mask
        self.dealias = (
                (torch.abs(self.Kx) < (2.0 / 3.0) * (self.s // 2)) &
                (torch.abs(self.Ky) < (2.0 / 3.0) * (self.s // 2)) &
                (torch.abs(self.Kz) < (2.0 / 3.0) * (self.s // 2))
        ).float().unsqueeze(0)

        self.Kx = self.Kx * (2 * math.pi * 1j)
        self.Ky = self.Ky * (2 * math.pi * 1j)
        self.Kz = self.Kz * (2 * math.pi * 1j)

        # Initialize zero fields
        self.density_field  = torch.zeros(*self.size).to(self.device)  # ρ
        self.momentum_field = torch.zeros(3, *self.size).to(self.device)  # m = ρV
        self.energy_field   = torch.zeros(*self.size).to(self.device)  # ρe

        self.velocity_field    = torch.zeros(3, *self.size).to(self.device)  # V
        self.pressure_field    = torch.zeros(*self.size).to(self.device)  # P = ρRT
        self.temperature_field = torch.zeros(*self.size).to(self.device)  # T

        self.total_energy = torch.zeros(*self.size).to(self.device)       # e = cvT + 1/2V^2
        self.total_enthalpy = torch.zeros(*self.size).to(self.device)     # H = e + P/ρ
        self.stress_tensor = torch.zeros(3, 3, *self.size).to(self.device)

        # External Force Fields
        self.force_field       = torch.zeros(3, *self.size).to(self.device)
        self.heat_field        = torch.zeros(*self.size).to(self.device)

        # self.vorticity_field = torch.zeros(3, *self.size).to(self.device)

        # Fields parameters
        # --------------------------------

        # Primary field initialization -> ["constant", "gaussian", "random"]
        self.initial_density     = "constant"
        self.initial_temperature = "gaussian"
        self.initial_velocity    = "random"

        self.average_density = 10.0
        self.average_temperature = 3.0
        self.velocity_scaling = 1.0

        self.force_type = "Axial"
        self.force_intensity = 3.0
        self.constant_force = True
        self.buoyancy = False

        self.heat_intensity = 10.0

        self.gravity = None
        self.visc = 1e-4
        self.kappa = 1e-2
        self.R = 1.0
        self.gamma = 1.4
        self.beta = 1.0
        self.cv = self.R / (self.gamma - 1)
        self.cp = self.gamma * self.cv

        # Max dt step
        cs = torch.sqrt(self.gamma * self.R * self.temperature_field).max()
        vmax = self.velocity_field.norm(dim=0).max()
        dx = 1.0 / self.s
        CFL = (vmax + cs) * self.dt / dx
        print("∆t CFL =", np.round(CFL.item(), 6))
        print("∆t v =", np.round((2 / self.visc)  * (1 / (math.pi * self.s)) ** 2, 2))
        print("∆t k =", np.round((2 / self.kappa) * (1 / (math.pi * self.s)) ** 2, 2))

        self.early_stopping = False

    def set_device(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("Training on : ", device)

        return device

    def apply_dealias(self, field):
        field_h = torch.fft.rfftn(field, dim=[-3, -2, -1]) * self.dealias
        return torch.fft.irfftn(field_h, s=self.size, dim=[-3, -2, -1]).real

    def random_scalar_field(self, spectral_slope=-1, k0_frac=0.25):
        """
        Generate a smooth random scalar field (e.g. temperature) via a Gaussian random field.

        Parameters:
        -----------
        N : int
            Grid size (must match solver)
        spectral_slope : float
            Power-law slope of the energy spectrum (e.g. -4 for smooth fields)
        k0_frac : float
            Controls dominant wavelength (k0 = N * k0_frac)
        """
        K = torch.sqrt(torch.abs(self.K2))
        K[0, 0, 0] = 1.0  # avoid division by zero

        # Define spectral amplitude (power law + Gaussian envelope)
        k0 = self.s * k0_frac
        amp = (K ** (spectral_slope / 2.0)) * torch.exp(- (K / k0) ** 2)

        # Random complex phases
        phase = torch.exp(1j * 2 * math.pi * torch.rand(self.s, self.s, self.s // 2 + 1, device=self.device))

        # Combine
        T_h = amp * phase

        # Transform to physical space
        T = torch.fft.irfftn(T_h, s=self.size)

        # Normalize to [0, 1]
        T = (T - T.min()) / (T.max() - T.min())

        return T.real

    def random_divfree_field_from_potential(self, spectral_slope=-2.0):
        K = torch.sqrt(self.K2)
        K[0, 0, 0] = 1.0

        # Generate Gaussian random field for A
        amp = K ** (spectral_slope / 2.0) * torch.exp(- (K / (self.s / 4)) ** 2)
        phase = torch.exp(1j * 2 * math.pi * torch.rand(3, self.s, self.s, self.s // 2 + 1, device=self.device))
        A_h = amp.unsqueeze(0) * phase

        # Compute curl(A) in Fourier space
        vx_h = (self.Ky * A_h[2] - self.Kz * A_h[1])
        vy_h = (self.Kz * A_h[0] - self.Kx * A_h[2])
        vz_h = (self.Kx * A_h[1] - self.Ky * A_h[0])

        v = torch.fft.irfftn(torch.stack([vx_h, vy_h, vz_h]), s=self.size, dim=[-3, -2, -1])

        # Normalize for convenience
        v = v / v.abs().max()

        return v.real

    def get_force_field(self, t):

        assert self.force_type in {"ABC", "Central", "Axial"}, "Force type must be 'ABC', 'Central', or 'Axial'"

        x = torch.linspace(0, 1, self.s + 1)
        x = x[0:-1]
        X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')

        match self.force_type:
            case "ABC":
                    A = 0.2
                    B = 0.2
                    C = 0.2
                    k = 23
                    omega = 3
                    argx = 2 * math.pi * ((k * X / self.s) + omega * t)
                    argy = 2 * math.pi * ((k * Y / self.s) + omega * t)
                    argz = 2 * math.pi * ((k * Z / self.s) + omega * t)
                    f_x = A * torch.sin(argz) + C * torch.cos(argy)
                    f_y = B * torch.sin(argx) + A * torch.cos(argz)
                    f_z = C * torch.sin(argy) + B * torch.cos(argx)

            case "Central":
                    dx = X - X.mean()
                    dy = Y - Y.mean()
                    dz = Z - Z.mean()
                    r = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    # avoid division by zero at the center
                    eps = 5e-1
                    r_safe = r + eps

                    # 4. inverse-square magnitude
                    mag = 1.0 / (r_safe ** 2)

                    # 5. vector field pointing toward center
                    f_x = -dx * mag / r_safe
                    f_y = -dy * mag / r_safe
                    f_z = -dz * mag / r_safe

            case "Axial":
                    dx = X-X.mean()
                    dy = Y-Y.mean()
                    dz = Z-Z.mean()
                    r = torch.sqrt(dx**2 + dy**2)

                    # avoid division by zero at the center
                    eps = 5e-1
                    r_safe = r + eps

                    mag = 1.0 / (r_safe ** 2)

                    f_x = -dy * mag
                    f_y = dx * mag
                    f_z = torch.zeros_like(f_x)

                    ring = False
                    if not ring:
                        R = 0.25  # radius of the ring
                        sigma = 0.08  # thickness
                        w = torch.exp(-((r - R) ** 2) / (2 * sigma ** 2))
                        # restoring force toward z=0
                        Fz_plane = -dz

                        # axial force (keeps direction consistent across plane)
                        Fz_axial = torch.sign(dz)

                        # blend
                        f_z = (1 - w) * Fz_plane + w * Fz_axial
                    else:
                        sigma_z = 0.08  # ring thickness in z
                        A_ring = torch.exp(-(dz ** 2) / (2 * sigma_z ** 2))
                        R = 0.3
                        sigma_r = 0.07
                        A_radial = torch.exp(-((r - R) ** 2) / (2 * sigma_r ** 2))
                        A_xy = A_ring * A_radial * 5
                        Fz_restore = -dz / torch.sqrt(dz ** 2 + sigma_z ** 2)
                        f_x *= A_xy
                        f_y *= A_xy
                        f_z = (1 - A_ring) * Fz_restore

        f = torch.stack((f_x, f_y, f_z), dim=0)

        if self.gravity is not None:
            zero = torch.zeros_like(X)
            g = self.gravity * torch.ones_like(zero)
            grav = torch.stack((zero, zero, g), dim=0)

            f += grav

        return f.to(self.device)

    def gaussian_field(self, xc = 0.5, yc = 0.5, zc = 0.5, sigma = 0.15):
        x = torch.linspace(0, 1, self.s + 1)
        x = x[0:-1]
        X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')

        gauss_field = torch.exp(-((X - xc) ** 2 + (Y - yc) ** 2 + (Z - zc) ** 2) / (2 * sigma ** 2))

        return gauss_field.to(self.device)

    def initialize_fields(self):

        # Initialize primary fields
        assert self.initial_density in ["constant", "gaussian", "random"], \
            "Initial density field must be 'constant', 'gaussian' or 'random'"
        match self.initial_density:
            case "constant":
                self.density_field = self.apply_dealias(torch.ones_like(self.density_field) * self.average_density)
            case "gaussian":
                self.density_field = self.apply_dealias(self.gaussian_field(sigma=0.15) + self.average_density)
            case "random":
                self.density_field = self.apply_dealias(self.random_scalar_field(spectral_slope=-0.5) + self.average_density)

        assert self.initial_temperature in ["constant", "gaussian", "random"], \
            "Initial temperature field must be 'constant', 'gaussian' or 'random'"
        match self.initial_temperature:
            case "constant":
                self.temperature_field = self.apply_dealias(torch.ones_like(self.density_field) * self.average_temperature)
            case "gaussian":
                self.temperature_field = self.apply_dealias(self.gaussian_field(sigma=0.15) + self.average_temperature)
            case "random":
                self.temperature_field = self.apply_dealias(
                    self.random_scalar_field(spectral_slope=-1.0) + self.average_temperature)

        assert self.initial_velocity in ["zero", "random"],  "Initial velocity field must be 'zero' or 'random'"
        match self.initial_velocity:
            case "zero":
                pass
            case "random":
                self.velocity_field = self.apply_dealias(self.random_divfree_field_from_potential() * self.velocity_scaling)

        # Initialize secondary fields - derived from primary
        self.momentum_field = self.apply_dealias(self.density_field * self.velocity_field)
        self.total_energy   = self.cv * self.temperature_field + 0.5 * (self.velocity_field ** 2).sum(dim=0, keepdim=True)
        self.energy_field   = self.apply_dealias(self.density_field * self.total_energy)
        self.pressure_field = self.apply_dealias(self.R * self.density_field * self.temperature_field)
        self.total_enthalpy = self.apply_dealias(self.total_energy + self.R * self.temperature_field)

        # Initialize forcing fields
        if self.force_type is not None:
            self.force_field = self.get_force_field(0) * self.force_intensity
        if self.heat_intensity is not None:
            self.heat_field = self.gaussian_field() * self.heat_intensity

        print("Vector field size = ", self.velocity_field.size())
        print("Scalar field size = ", self.temperature_field.size())

        print("Initial values : ")

        print(f"Initial density : {self.initial_density}")
        print(f"Density Field     : max = {torch.max(self.density_field).cpu().numpy()}, "
              f"min = {torch.min(self.density_field).cpu().numpy()}")

        print(f"Initial temperature : {self.initial_temperature}")
        print(f"Temperature Field : max = {torch.max(self.temperature_field).cpu().numpy()}, "
              f"min = {torch.min(self.temperature_field).cpu().numpy()}")

        mod_v = (self.velocity_field ** 2).sum(dim=0)
        print(f"Initial velocity : {self.initial_velocity}")
        print(f"Velocity Field    : max = {torch.max(mod_v).cpu().numpy()}, min = {torch.min(mod_v).cpu().numpy()}")

        mod_f = (self.force_field ** 2).sum(dim=0)
        print(f"Force field type : {self.force_type}")
        print(f"Force Field       : max = {torch.max(mod_f).cpu().numpy()}, min = {torch.min(mod_f).cpu().numpy()}")

        print(f"Heat field : None") if self.heat_intensity is None else print(f"Heat field : Gaussian")
        print(f"Heat field        :max = {torch.max(self.heat_field).cpu().numpy()}, min = {torch.min(self.heat_field).cpu().numpy()}")

        return

    def compute_gradient(self, tensor):

        scalar_field = True if tensor.size() == 3 else False

        FT_tensor = torch.fft.rfftn(tensor, dim=[-3, -2, -1]) * self.dealias

        grad_x = torch.fft.irfftn(self.Kx * FT_tensor, s=self.size, dim=[-3, -2, -1]).real
        grad_y = torch.fft.irfftn(self.Ky * FT_tensor, s=self.size, dim=[-3, -2, -1]).real
        grad_z = torch.fft.irfftn(self.Kz * FT_tensor, s=self.size, dim=[-3, -2, -1]).real

        if scalar_field:
            grad_x = grad_x[0]
            grad_y = grad_y[0]
            grad_z = grad_z[0]

        return grad_x, grad_y, grad_z

    def compute_divergence(self, tensor):

        FT_tensor = torch.fft.rfftn(tensor, dim=[-3, -2, -1]) * self.dealias

        div_v = self.Kx * FT_tensor[0] + self.Ky * FT_tensor[1] + self.Kz * FT_tensor[2]
        div_v = torch.fft.irfftn(div_v, s=self.size, dim=(-3, -2, -1)).real

        return div_v

    def compute_stress_tensor(self):

        vx, vy, vz = self.compute_gradient(self.velocity_field)
        vx_dx = vx[0]
        vy_dx = vx[1]
        vz_dx = vx[2]
        vx_dy = vy[0]
        vy_dy = vy[1]
        vz_dy = vy[2]
        vx_dz = vz[0]
        vy_dz = vz[1]
        vz_dz = vz[2]

        div_v = (vx_dx + vy_dy + vz_dz)

        grad_v = torch.stack([
            torch.stack([vx_dx, vx_dy, vx_dz]),
            torch.stack([vy_dx, vy_dy, vy_dz]),
            torch.stack([vz_dx, vz_dy, vz_dz])
        ]).squeeze()

        self.stress_tensor = 2*self.visc * (grad_v + grad_v.transpose(0, 1) - (1 / 3) * div_v * self.I)
        self.stress_tensor = self.apply_dealias(self.stress_tensor)

        return

    def update_fields(self, euler_fluxes, viscous_fluxes):

        euler_fluxes   = self.apply_dealias(euler_fluxes)
        viscous_fluxes = self.apply_dealias(viscous_fluxes)
        force_density  = self.apply_dealias(self.density_field * self.force_field)
        heat_density   = self.apply_dealias(self.density_field * self.heat_field)

        # Compute divergence
        rho_rhs = self.compute_divergence(self.momentum_field)
        mx_rhs  = self.compute_divergence(euler_fluxes[0] - self.stress_tensor[0]) - force_density[0]
        my_rhs  = self.compute_divergence(euler_fluxes[1] - self.stress_tensor[1]) - force_density[1]
        mz_rhs  = self.compute_divergence(euler_fluxes[2] - self.stress_tensor[2]) - force_density[2]
        e_rhs   = self.compute_divergence(self.total_enthalpy * self.momentum_field - viscous_fluxes) - heat_density

        # Update fields
        self.density_field     -= self.dt * rho_rhs
        self.momentum_field[0] -= self.dt * mx_rhs
        self.momentum_field[1] -= self.dt * my_rhs
        self.momentum_field[2] -= self.dt * mz_rhs
        self.energy_field      -= self.dt * e_rhs

        self.density_field  = self.apply_dealias(self.density_field)
        self.momentum_field = self.apply_dealias(self.momentum_field)
        self.energy_field   = self.apply_dealias(self.energy_field)

        self.velocity_field = self.apply_dealias(self.momentum_field / self.density_field)
        self.total_energy   = self.apply_dealias(self.energy_field / self.density_field)
        self.temperature_field = self.apply_dealias(
            (self.total_energy - 0.5 * (self.velocity_field ** 2).sum(dim=0, keepdim=True)) / self.cv
        )
        self.pressure_field = self.apply_dealias(self.R * self.density_field * self.temperature_field)
        self.total_enthalpy = self.apply_dealias(self.total_energy + self.R * self.temperature_field)

        return

    def simulate(self):

        # Solution Containers
        # --------------------------------------------------
        sol_V = torch.zeros(*self.velocity_field.size(), self.record_steps, device=self.device)  # Velocity
        sol_T = torch.zeros(*self.temperature_field.size(), self.record_steps, device=self.device)  # Temperature
        sol_p = torch.zeros_like(sol_T)  # Pressure
        sol_d = torch.zeros_like(sol_T)  # Density
        # --------------------------------------------------

        # Record counter
        c = 0
        # Physical time
        t = 0.0

        self.initialize_fields()

        print("Start simulation :")
        start = default_timer()
        for j in tqdm(range(self.steps)):

            # 1. Compute Euler fluxes E_ij = ρ(v_i)(v_j) + I*P
            # --------------------------------------------------
            E = torch.einsum('i... , j... -> ij...', self.momentum_field, self.velocity_field)
            E += self.I * self.pressure_field

            # 2. Compute Stress Tensor  : τ = µ(∇v + (∇v)^T - 2/3 I ∇·v)
            # --------------------------------------------------
            self.compute_stress_tensor()

            # 3. Compute viscous fluxes :
            # --------------------------------------------------
            # Temperature Gradient
            T_x, T_y, T_z = self.compute_gradient(self.temperature_field)

            F = torch.einsum('jxyz,ijxyz->ixyz', self.velocity_field, self.stress_tensor)  # shape : (3,32,32,32)
            F += torch.cat([self.kappa * T_x, self.kappa * T_y, self.kappa * T_z], dim=0)

            self.update_fields(E, F)

            # Update real time (used only for recording)
            t += self.dt

            # Update Forcing
            if not self.constant_force:
                self.force_field = self.get_force_field(t / self.time)
            else:
                pass
            # Compute buoyancy forcing
            if self.buoyancy:
                self.force_field[2] += -self.gravity * self.beta * self.temperature_field.squeeze()

            # Save frame
            # --------------------------------------------------
            if (j + 1) % self.record_time == 0:
                print(f"max|div u|={torch.abs(self.compute_divergence(self.velocity_field)).max().item():.3e}")
                print("Average Density :", torch.mean(self.density_field).cpu().numpy())
                print("Average Energy  :", torch.mean(self.total_energy).cpu().numpy())

                if torch.isnan(self.velocity_field).any():
                    print("Nan detected, interrupt simulation")
                    break

                # Record solution and time
                sol_V[..., c] = self.velocity_field
                sol_T[..., c] = self.temperature_field
                sol_p[..., c] = self.pressure_field
                sol_d[..., c] = self.density_field

                c += 1

                if self.early_stopping:
                    break

        # End Simulation
        # --------------------------------------------------
        end = default_timer()
        print("Total time = {} m".format(np.round((end - start) / 60), 4))

        return sol_V, sol_T, sol_p, sol_d
