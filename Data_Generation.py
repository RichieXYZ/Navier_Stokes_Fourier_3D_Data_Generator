from Solver_plots import *
from Simulator_Class import Simulator3D

# Parameters
# ---------------------------------------------------
s = 32
T_int = 2
dt = 1e-4
record_steps = 200
# ---------------------------------------------------

# Main Simulation
# ---------------------------------------------------
sim = Simulator3D(s, T_int, dt, record_steps)
sol_V, sol_T, sol_p, sol_d = sim.simulate()

# Save to MATLAB file
"""
savemat("5D_Fields.mat", {'V': sol_V.cpu().numpy(), 
                                          'T': sol_T.cpu().numpy(), 
                                          'p': sol_p.cpu().numpy(),
                                          'd' : sol_d.cpu().numpy()})
"""
# Output Visualization
# ---------------------------------------------------
output = sol_V.permute(4, 0, 1, 2, 3).cpu()
print("Velocity field shape :", output.shape)
output_p = sol_p.squeeze().permute(3, 0, 1, 2).cpu()
print("Pressure field shape :", output_p.shape)
output_T = sol_T.squeeze().permute(3, 0, 1, 2).cpu()
print("Temperature field shape :", output_T.shape)
output_d = sol_d.squeeze().permute(3, 0, 1, 2).cpu()
print("Density field shape :", output_d.shape)

step = int(s / 16)
vector_field_animation_3d(output, s, step, normalize=True, field_name="Velocity")
vector_projection_animation(torch.asarray(output))
scalar_field_animation_3d(output_p, s, step=step, field_name="Pressure")
scalar_projection_animation(torch.asarray(output_p))
scalar_field_animation_3d(output_T, s, step=step, field_name="Temperature")
scalar_projection_animation(torch.asarray(output_T))
scalar_field_animation_3d(output_d, s, step=step, field_name="Density")
scalar_projection_animation(torch.asarray(output_d))