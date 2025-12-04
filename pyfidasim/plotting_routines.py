import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Global Publication Style Settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'figure.dpi': 120,
})

def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots in Matplotlib."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    zmean = np.mean(zlim)
    
    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])
    
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

def plot_geometry_3d(fields, spec, nbi=None, grid3d=None, plot_crossed_cells=False, s_surfaces=[1.0]):
    """
    Plots the 3D machine geometry including Flux Surfaces, LOS, and NBI vectors.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot Flux Surfaces (Wireframe)
    if fields is not None and 'Rsurf' in fields:
        for surf_val in s_surfaces:
            # Find closest index
            isurf = np.argmin(np.abs(fields['s_surf'] - surf_val))
            
            # Extract profile
            R_prof = fields['Rsurf'][isurf, 0, :]
            Z_prof = fields['Zsurf'][isurf, 0, :]
            
            # Create rings
            phis = np.linspace(fields['phimin'], fields['phimax'], 40)
            
            # Plot Toroidal Rings (black)
            indices = np.linspace(0, len(R_prof)-1, 6, dtype=int)
            for idx in indices:
                x = R_prof[idx] * np.cos(phis)
                y = R_prof[idx] * np.sin(phis)
                z = np.full_like(phis, Z_prof[idx])
                ax.plot(x, y, z, color='black', lw=0.8, alpha=0.3)

            # Plot Poloidal Slices (gray)
            for p in phis[::8]:
                ax.plot(R_prof*np.cos(p), R_prof*np.sin(p), Z_prof, color='gray', lw=0.5, alpha=0.2)

    # 2. Plot LOS
    if spec is not None:
        los_lens = spec['los_pos']
        los_vec = spec['los_vec']
        colors = plt.cm.viridis(np.linspace(0, 1, spec['nlos']))
        
        for i in range(spec['nlos']):
            dist = spec.get('full_los_length', np.full(spec['nlos'], 400.))[i]
            start = los_lens[i]
            end = start + los_vec[i] * dist
            
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    color=colors[i], lw=1.5)
            ax.scatter(start[0], start[1], start[2], color=colors[i], s=15)

    # 3. Plot NBI
    if nbi is not None:
        for src_name in nbi['sources']:
            src = nbi[src_name]
            pos = src['source_position']
            direc = src['direction']
            length = 600
            end = pos + direc * length
            ax.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]], 
                    color='red', lw=3, label=f'NBI {src_name}')
            ax.scatter(pos[0], pos[1], pos[2], color='darkred', s=50, marker='^')

    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    ax.set_title("Observation Geometry")
    ax.legend()
    set_aspect_equal_3d(ax)
    plt.tight_layout()
    
    return fig

def plot_spectra_interactive(spec, labels=['full', 'half', 'third', 'halo'], scale_factor=1e18):
    """
    Creates a figure with a slider to browse through all LOS channels.
    Manual layout management is used to prevent slider overlap.
    """
    if spec is None or 'intens' not in spec: return

    # Create figure without constrained_layout to allow manual slider placement
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Reserve space at the bottom for the slider
    plt.subplots_adjust(bottom=0.20, top=0.92, left=0.10, right=0.95)
    
    # Store line objects
    lines = []
    colors = ['royalblue', 'forestgreen', 'darkorchid', 'firebrick']
    mapping = {'full': 0, 'half': 1, 'third': 2, 'halo': 3}
    
    # Initial plot (Channel 0)
    init_chan = 0
    max_y = 0.0
    
    for i, label in enumerate(labels):
        comp_idx = mapping.get(label)
        if comp_idx < spec['intens'].shape[0]:
            y_data = spec['intens'][comp_idx, init_chan, :] / scale_factor
            l, = ax.plot(spec['wavel'], y_data, label=label.capitalize(), color=colors[i], lw=2)
            lines.append({'line': l, 'comp_idx': comp_idx})
            
            curr_max = np.max(y_data)
            if curr_max > max_y: max_y = curr_max

    # Formatting
    los_name = spec['losname'][init_chan] if 'losname' in spec else f"LOS #{init_chan}"
    ax.set_title(f"Spectra - {los_name}")
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel(r'Intensity [$10^{18}$ ph/(s sr nm m$^2$)]')
    ax.set_xlim(spec['lambda_min'], spec['lambda_max'])
    ax.set_ylim(0, max_y * 1.2 if max_y > 0 else 1.0)
    ax.legend(title="Component", loc='upper right', frameon=False)
    ax.grid(True, alpha=0.3)

    # Slider Axes Placement
    ax_slider = plt.axes([0.20, 0.05, 0.60, 0.03], facecolor='lightgoldenrodyellow')
    
    slider = Slider(
        ax=ax_slider,
        label='Channel Index ',
        valmin=0,
        valmax=spec['nlos'] - 1,
        valinit=init_chan,
        valstep=1,
        color='gray'
    )

    def update(val):
        idx = int(slider.val)
        local_max = 0.0
        
        for item in lines:
            y_new = spec['intens'][item['comp_idx'], idx, :] / scale_factor
            item['line'].set_ydata(y_new)
            if np.max(y_new) > local_max: local_max = np.max(y_new)
        
        new_name = spec['losname'][idx] if 'losname' in spec else f"LOS #{idx}"
        ax.set_title(f"Spectra - {new_name}")
        ax.set_ylim(0, local_max * 1.2 if local_max > 0 else 1.0)
        
        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig._slider = slider 
    return fig

def plot_midplane_heatmap(grid3d, fields, energy_indices=[0, 1, 2]):
    """
    Plots a 2D Heatmap of the beam density at the midplane (Z=0).
    """
    if grid3d is None or 'density' not in grid3d: return

    z_idx = np.argmin(np.abs(grid3d['Z_c']))
    beam_dens_map = np.sum(grid3d['density'][energy_indices, 0, :, z_idx, :], axis=0) 
    beam_dens_map = np.ma.masked_where(beam_dens_map < 1e7, beam_dens_map)

    # Coordinates
    R_edges = np.linspace(grid3d['Rmin'], grid3d['Rmax'], grid3d['nR'] + 1)
    Phi_edges = np.linspace(grid3d['phimin'], grid3d['phimax'], grid3d['nphi'] + 1)
    RR_edge, PP_edge = np.meshgrid(R_edges, Phi_edges, indexing='ij')
    XX_edge = RR_edge * np.cos(PP_edge)
    YY_edge = RR_edge * np.sin(PP_edge)

    # Dynamic Size
    x_min, x_max = np.min(XX_edge), np.max(XX_edge)
    y_min, y_max = np.min(YY_edge), np.max(YY_edge)
    
    aspect_ratio = (y_max - y_min) / (x_max - x_min)
    fig_width = 8
    fig_height = fig_width * aspect_ratio * 0.9 
    fig_height = max(4, min(fig_height, 12))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot
    cm = ax.pcolormesh(XX_edge, YY_edge, beam_dens_map, cmap='magma', shading='flat', rasterized=True)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(cm, cax=cax)
    cbar.set_label(r"Beam Neutral Density [$cm^{-3}$]")
    
    # Overlays
    if fields is not None:
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(fields['Rmin']*np.cos(theta), fields['Rmin']*np.sin(theta), 'k--', lw=1, alpha=0.4, label='Grid Limit')
        ax.plot(fields['Rmax']*np.cos(theta), fields['Rmax']*np.sin(theta), 'k--', lw=1, alpha=0.4)

        if 'Rsurf' in fields:
            idx_lcfs = np.argmin(np.abs(fields['s_surf'] - 1.0))
            R_lcfs = fields['Rsurf'][idx_lcfs, 0, :]
            R_in = np.min(R_lcfs)
            R_out = np.max(R_lcfs)
            ax.plot(R_in*np.cos(theta), R_in*np.sin(theta), 'k-', lw=2, label='LCFS')
            ax.plot(R_out*np.cos(theta), R_out*np.sin(theta), 'k-', lw=2)

    ax.set_aspect('equal')
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_title(f"Midplane Beam Density (Z={grid3d['Z_c'][z_idx]:.1f} cm)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    
    return fig

def plot_profiles(profiles, savefig=False):
    """Plots the 1D Kinetic Profiles."""
    if profiles is None: return

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(6, 9), layout='constrained')
    x_axis = profiles['s']
    
    # Density
    if 'dene' in profiles:
        ax1.plot(x_axis, profiles['dene'] / 1e13, 'k-', lw=2, label=r'$n_e$')
    if 'denp' in profiles:
        ax1.plot(x_axis, profiles['denp'] / 1e13, 'r--', lw=2, label=r'$n_i$')
    if 'denimp' in profiles:
        imp = profiles['denimp']
        if imp.ndim > 1: imp = np.sum(imp, axis=0)
        scale_imp = 10.0
        ax1.plot(x_axis, (imp * scale_imp) / 1e13, 'b:', lw=2, label=r'$n_{imp} \times 10$')
        
    ax1.set_ylabel(r'Density [$10^{13} cm^{-3}$]')
    ax1.legend(loc='upper right', frameon=False, fontsize=10)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Input Plasma Profiles")

    # Temperature
    if 'te' in profiles:
        ax2.plot(x_axis, profiles['te'], 'k-', lw=2, label=r'$T_e$')
    if 'ti' in profiles:
        ax2.plot(x_axis, profiles['ti'], 'r--', lw=2, label=r'$T_i$')
    
    ax2.set_ylabel(r'Temperature [keV]')
    ax2.legend(loc='upper right', frameon=False, fontsize=10)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    # Rotation
    if 'omega' in profiles:
        ax3.plot(x_axis, profiles['omega'] / 1e3, 'g-', lw=2)
        ax3.set_ylabel(r'$\omega_{rot}$ [krad/s]')
    
    ax3.set_xlabel(r'Normalized Flux Coordinate $s$')
    ax3.set_xlim(0, 1.1)
    ax3.grid(True, alpha=0.3)
    
    if savefig:
        plt.savefig('input_kinetic_profiles.png', bbox_inches='tight')
        
    return fig

def plot_magnetic_equilibrium(fields):
    """
    Plots a 2D Poloidal cross-section of the Magnetic Equilibrium.
    Shows Flux Surfaces (contours of s) and Magnetic Field Magnitude (|B|).
    """
    if fields is None: return

    # Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True, layout='constrained')

    # Data Handling
    # fields['s'] is usually [nR, nZ, nPhi]. We take the first toroidal slice.
    # Note: meshgrid requires X, Y. Pcolormesh(X, Y, C).
    # If C has shape (nX, nY), we need to transpose to match (nY, nX) for some plotters,
    # but here: R is axis 0, Z is axis 1.
    # We want R on X-axis, Z on Y-axis.
    
    # Grid
    R = fields['R']
    Z = fields['Z']
    RR, ZZ = np.meshgrid(R, Z)
    
    # 1. Flux Coordinate s
    # s array is [R, Z, Phi]. We want [Z, R] for plotting over meshgrid(R, Z)
    s_map = fields['s'][:, :, 0].T 
    
    # Plot contours
    # Levels: Focus on 0 to 1.2
    levels = np.linspace(0, 1.2, 25)
    cp1 = ax1.contourf(RR, ZZ, s_map, levels=levels, cmap='viridis')
    
    # LCFS Highlight (s=1.0)
    # 1. Contour approach
    ax1.contour(RR, ZZ, s_map, levels=[1.0], colors='red', linewidths=2, label='LCFS')
    
    # 2. Explicit Rsurf approach (Ground Truth Overlay)
    # This verifies if the grid interpolation matches the definition
    if 'Rsurf' in fields:
        idx_lcfs = np.argmin(np.abs(fields['s_surf'] - 1.0))
        # Rsurf is [s, phi, theta] -> [theta]
        R_lcfs = fields['Rsurf'][idx_lcfs, 0, :]
        Z_lcfs = fields['Zsurf'][idx_lcfs, 0, :]
        ax1.plot(R_lcfs, Z_lcfs, 'w--', lw=1.5, alpha=0.7, label='Parametric LCFS')

    ax1.set_title("Normalized Flux Coordinate $s$")
    ax1.set_xlabel("R [cm]")
    ax1.set_ylabel("Z [cm]")
    ax1.set_aspect('equal')
    cbar1 = fig.colorbar(cp1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(r"$s \sim \rho^2$")
    ax1.legend(loc='upper right', framealpha=0.8, fontsize=10)

    # 2. Magnetic Field Magnitude |B|
    # B = sqrt(Br^2 + Bz^2 + Bphi^2)
    Br = fields['Br'][:, :, 0].T
    Bz = fields['Bz'][:, :, 0].T
    Bphi = fields['Bphi'][:, :, 0].T
    B_mag = np.sqrt(Br**2 + Bz**2 + Bphi**2)

    cp2 = ax2.contourf(RR, ZZ, B_mag, levels=25, cmap='plasma')
    # Overlay LCFS
    ax2.contour(RR, ZZ, s_map, levels=[1.0], colors='white', linestyles='--', linewidths=1.5)
    
    ax2.set_title("Magnetic Field Magnitude $|B|$")
    ax2.set_xlabel("Major Radius R [cm]")
    ax2.set_aspect('equal')
    cbar2 = fig.colorbar(cp2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r"$|B|$ [Tesla]")

    return fig