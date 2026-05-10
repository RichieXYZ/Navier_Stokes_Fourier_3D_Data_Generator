# Navier_Stokes_Fourier_3D_Data_Generator
This Tool performs data generation for 3D Navier-Stokes-Fourier models of thermo-fluids. It is a pseudo-spectral solver that compute differentiation in Fourier space and non-linear products in physical space. It can simulate the fluid in a wide range of behaviours such as thermal diffusion, wave propagation and vortex formation. 

<img width="500" height="417" alt="Screen Recording 2026-03-08 at 22 47 27" src="https://github.com/user-attachments/assets/5d0afb0b-4094-4ed2-b7e4-8123d93d2f74" />


# 5D Dataset

The output of a simulator run is a 5D tensor organized in the following way [c, t, x, y, z], where "c" indicates the number of channels corresponding to the physical variables: velocity, temperature, pressure and density. The symbol "t" indicates the time frame dimenstion while "x,y,z" the spatial location on a uniform 3D spatial grid.

<img width="741" height="380" alt="Screenshot 2026-05-10 at 14 18 28" src="https://github.com/user-attachments/assets/7af3fdbf-83ae-4d55-8710-1f4caab865d2" />

The physical fields can be initialized with various spatial distribution such as unifotm, gaussian or random and the external force field can be selected among three different fields, "ABC" indicates the Arnold-Beltrami-Childress field type, "radial" that resembles a central force field and "axial" that is useful to simulate system with injection of angular momentum.



# Super Resolution U-Net

# Equations
