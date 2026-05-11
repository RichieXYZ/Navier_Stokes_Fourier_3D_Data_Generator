import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation


def plot_3D_vector_field(field, res, step, title):
    nx, ny, nz = res, res, res
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    z = np.linspace(-2, 2, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Extract x,y,z components of the field
    u, v, w = field[0, :, :, :], field[1, :, :, :], field[2, :, :, :]

    # compute vector magnitude
    mag = np.sqrt(u ** 2 + v ** 2 + w ** 2)

    # --- Downsample for arrows ---
    Xs = X[::step, ::step, ::step]
    Ys = Y[::step, ::step, ::step]
    Zs = Z[::step, ::step, ::step]

    U = u[::step, ::step, ::step]
    V = v[::step, ::step, ::step]
    W = w[::step, ::step, ::step]

    M = mag[::step, ::step, ::step]

    # --- Plot ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Normalize magnitudes for coloring
    norm = Normalize(vmin=M.min(), vmax=M.max())
    colors = cm.viridis(norm(M.ravel()))

    # Quiver plot: arrows scaled by magnitude, color mapped by M
    q = ax.quiver(Xs, Ys, Zs, U, V, W, length=0.2, normalize=True, colors=colors)

    # Add colorbar manually
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Magnitude")

    # Labels
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()

def vector_field_animation_3d(field, res, step, normalize, field_name):

    Nx, Ny, Nz = res, res, res
    x = np.linspace(-2, 2, Nx)
    y = np.linspace(-2, 2, Ny)
    z = np.linspace(-2, 2, Nz)
    X, Y, Z = np.meshgrid(x, y, z,indexing="ij")

    # Set up 3D figure
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    arrow_lenght = 0.2 if normalize else 1.0

    # Initial field
    U, V, W = field[:, 0, :, :, :], field[:, 1, :, :, :], field[:, 2, :, :, :]

    U = 2 * (U - U.min()) / (U.max() - U.min()) - 1
    V = 2 * (V - V.min()) / (V.max() - V.min()) - 1
    W = 2 * (W - W.min()) / (W.max() - W.min()) - 1

    U = U[0]
    V = V[0]
    W = W[0]

    # Global normalization # -------------------------
    global_speed = np.sqrt(field[:, 0, :, :, :]**2 + field[:, 1, :, :, :]**2 + field[:, 2, :, :, :]**2)
    norm = plt.Normalize(global_speed.min(), global_speed.max())
    colors = cm.viridis(norm(global_speed)).reshape(-1, 4)

    # --- Downsample for arrows ---
    X = X[::step, ::step, ::step]
    Y = Y[::step, ::step, ::step]
    Z = Z[::step, ::step, ::step]

    U = U[::step, ::step, ::step]
    V = V[::step, ::step, ::step]
    W = W[::step, ::step, ::step]
    # Flatten everything for quiver
    Xf, Yf, Zf = X.flatten(), Y.flatten(), Z.flatten()
    Uf, Vf, Wf = U.flatten(), V.flatten(), W.flatten()

    global quiver
    quiver = ax.quiver(Xf, Yf, Zf, Uf, Vf, Wf, length=arrow_lenght, normalize=normalize, colors=colors)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_title(field_name + " Field")

    # Add colorbar
    mappable = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="Vector Magnitude")

    def update(frame):
        global quiver
        quiver.remove()
        # Extract velocities component
        U = field[frame, 0, ::step, ::step, ::step] # v_x
        V = field[frame, 1, ::step, ::step, ::step] # v_y
        W = field[frame, 2, ::step, ::step, ::step] # v_z

        speed = np.sqrt(U ** 2 + V ** 2 + W ** 2)
        colors = cm.viridis(norm(speed)).reshape(-1, 4)

        # Vector field plot
        quiver = ax.quiver(X.flatten(), Y.flatten(), Z.flatten(), U.flatten(), V.flatten(), W.flatten(),
                           length=arrow_lenght, normalize=normalize, colors=colors)
        return quiver

    # ------------------------- # Animate # -------------------------
    steps = len(field[:, 0, 0, 0, 0])
    print("Total number of frames = ", steps)
    anim = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)
    # anim.save(field_name + '_Field.mp4')
    plt.show()


def scalar_field_animation_3d(temp_field, res, step, field_name):
    # Grid
    Nx, Ny, Nz = res, res, res
    x = np.linspace(-2, 2, Nx)
    y = np.linspace(-2, 2, Ny)
    z = np.linspace(-2, 2, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Setup figure
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    t_min, t_max = temp_field.min(), temp_field.max()
    print(field_name + " Field :")
    print(f"Min value = {t_min.numpy()}, Max value = {t_max.numpy()}")

    # Global normalization
    temp_field = (temp_field - t_min) / (t_max - t_min)
    t_min, t_max = temp_field.min(), temp_field.max()
    norm = plt.Normalize(t_min, t_max)

    cmap = cm.inferno

    # Initial frame
    if temp_field.dim() < 4:
        temp_field = temp_field.reshape(1,res,res,res)
    T0 = temp_field[0]

    # Scatter points (down-sampled grid)
    Xs = X[::step, ::step, ::step].flatten()
    Ys = Y[::step, ::step, ::step].flatten()
    Zs = Z[::step, ::step, ::step].flatten()
    Ts = T0[::step, ::step, ::step].flatten()
    scatter = ax.scatter(Xs, Ys, Zs, c=Ts, cmap=cmap, norm=norm, s=10)

    # Colorbar
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label=field_name)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_title(field_name + " Field")

    # -------------------------
    # Update function
    # -------------------------
    def update(frame):

        # Get frame
        T = temp_field[frame]

        Ts = T[::step, ::step, ::step].flatten()
        scatter.set_array(Ts)

        return scatter,

    # -------------------------
    # Animate
    # -------------------------
    n_frames = temp_field.shape[0]
    print("Total frames = ", n_frames)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=10, blit=False)
    # anim.save(field_name + "_Field.mp4")
    plt.show()

def vector_projection_animation(field):

    artist = []

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    ax[0,0].title.set_text('Vx - plane xy')
    ax[1,0].title.set_text('Vy - plane xy')
    ax[2,0].title.set_text('Vz - plane xy')

    ax[0, 1].title.set_text('Vx - plane yz')
    ax[1, 1].title.set_text('Vy - plane yz')
    ax[2, 1].title.set_text('Vz - plane yz')

    ax[0, 2].title.set_text('Vx - plane zx')
    ax[1, 2].title.set_text('Vy - plane zx')
    ax[2, 2].title.set_text('Vz - plane zx')

    for i in range(field.shape[0]):
        Vx_yz = field[i, 0, :, :, :].mean(dim=0)
        Vx_xz = field[i, 0, :, :, :].mean(dim=1)
        Vx_xy = field[i, 0, :, :, :].mean(dim=2)

        Vy_yz = field[i, 1, :, :, :].mean(dim=0)
        Vy_xz = field[i, 1, :, :, :].mean(dim=1)
        Vy_xy = field[i, 1, :, :, :].mean(dim=2)

        Vz_yz = field[i, 2, :, :, :].mean(dim=0)
        Vz_xz = field[i, 2, :, :, :].mean(dim=1)
        Vz_xy = field[i, 2, :, :, :].mean(dim=2)

        plot1 = ax[0,0].imshow(np.asarray(Vx_yz), cmap='viridis')
        plot2 = ax[0,1].imshow(np.asarray(Vx_xz), cmap='viridis')
        plot3 = ax[0,2].imshow(np.asarray(Vx_xy), cmap='viridis')
        plot4 = ax[1,0].imshow(np.asarray(Vy_yz), cmap='viridis')
        plot5 = ax[1,1].imshow(np.asarray(Vy_xz), cmap='viridis')
        plot6 = ax[1,2].imshow(np.asarray(Vy_xy), cmap='viridis')
        plot7 = ax[2,0].imshow(np.asarray(Vz_yz), cmap='viridis')
        plot8 = ax[2,1].imshow(np.asarray(Vz_xz), cmap='viridis')
        plot9 = ax[2,2].imshow(np.asarray(Vz_xy), cmap='viridis')

        plot_list = [plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9]
        artist.append(plot_list)

    _ = animation.ArtistAnimation(fig=fig, artists=artist, interval=100)
    plt.show()

def scalar_projection_animation(field, field_name = None, normalize = False):

    artist = []

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 8))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    ax[0].title.set_text(field_name + ' x')
    ax[1].title.set_text(field_name + ' y')
    ax[2].title.set_text(field_name + ' z')

    # Colorbar
    # Colorbar
    if normalize:
        norm = Normalize(vmin=field.min(), vmax=field.max())
    else:
        norm = None
    mappable = cm.ScalarMappable(cmap="inferno", norm=norm)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label=field_name)

    for i in range(field.shape[0]):
        i=0
        Fx = field[i, :, :, :].mean(dim=0)
        Fy = field[i, :, :, :].mean(dim=1)
        Fz = field[i, :, :, :].mean(dim=2)

        plot1 = ax[0].imshow(np.asarray(Fx), cmap='inferno', norm=norm)
        plot2 = ax[1].imshow(np.asarray(Fy), cmap='inferno', norm=norm)
        plot3 = ax[2].imshow(np.asarray(Fz), cmap='inferno', norm=norm)

        plot_list = [plot1, plot2, plot3]
        artist.append(plot_list)

    _ = animation.ArtistAnimation(fig=fig, artists=artist, interval=100)
    plt.show()