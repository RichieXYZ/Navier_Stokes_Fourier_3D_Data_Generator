# Navier_Stokes_Fourier_3D_Data_Generator
This Tool performs data generation for 3D Navier-Stokes-Fourier models of thermo-fluids. It is a pseudo-spectral solver that compute differentiation in Fourier space and non-linear products in physical space. It can simulate the fluid in a wide range of behaviours such as thermal diffusion, wave propagation and vortex formation. 

<img width="500" height="417" alt="Screen Recording 2026-03-08 at 22 47 27" src="https://github.com/user-attachments/assets/5d0afb0b-4094-4ed2-b7e4-8123d93d2f74" />


# 5D Dataset

The output of a simulator run is a 5D tensor organized in the following way [c, t, x, y, z], where "c" indicates the number of channels corresponding to the physical variables: velocity, temperature, pressure and density. The symbol "t" indicates the time frame dimenstion while "x,y,z" the spatial location on a uniform 3D spatial grid.


# Super Resolution U-Net

# Equations
