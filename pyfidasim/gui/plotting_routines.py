# gui/plotting_routines.py

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui


def _qcolor_to_rgba_tuple(qcolor):
    return qcolor.getRgbF()

def plot_los(
    view_widget, # <-- Pass the GLViewWidget to draw on
    sgrid,
    grid3d,
    spec,
    nbi=None,
    plot_crossed_cells=False,
    colors=None,
    plot_ssurf=[1.0]
):
    """
    Populates a given gl.GLViewWidget with the 3D LOS geometry.
    This function MODIFIES the view_widget, it does not return a new one.
    """
    w = view_widget
    w.clear() # Clear any previous plots
    
    all_points = []
    
    if spec is None:
        return
        
    nlos = int(spec.get('nlos', 0))

    # 1) Plot LOS lines
    for ilos in range(nlos):
        dist = float(spec.get('full_los_length', [400.0]*nlos)[ilos])
        start = np.array(spec['los_pos'][ilos], dtype=float)
        direction = np.array(spec['los_vec'][ilos], dtype=float)
        end = start + dist * direction
        all_points.extend([start, end])
        pts = np.vstack((start, end))
        
        qcol = pg.Color(pg.intColor(ilos, hues=nlos, minHue=0, maxHue=300))
        line_color = _qcolor_to_rgba_tuple(qcol)

        line_item = gl.GLLinePlotItem(pos=pts, color=line_color, width=4, antialias=True)
        w.addItem(line_item)

    # 2) Plot crossed-cell intersection points
    if plot_crossed_cells and grid3d and 'los_grid_intersection_indices' in spec and spec['los_grid_intersection_indices'] is not None:
        for ilos in range(nlos):
            indices = spec['los_grid_intersection_indices'][ilos]
            if not isinstance(indices, np.ndarray) or indices.ndim != 2: continue
            
            valid = indices[(indices[:, 0] >= 0) & (indices[:, 1] >= 0) & (indices[:, 2] >= 0)]
            if valid.shape[0] == 0: continue

            ir, iz, iphi = valid[:, 0].astype(int), valid[:, 1].astype(int), valid[:, 2].astype(int)
            Rvals = grid3d['R_c'][ir]
            phis = grid3d['phi_c'][iphi] - grid3d.get('rotate_phi_grid', 0.0)
            X, Y, Z = Rvals * np.cos(phis), Rvals * np.sin(phis), grid3d['Z_c'][iz]
            
            pts = np.vstack((X, Y, Z)).T
            all_points.append(pts)

    # 3) Plot s-surface loops
    if sgrid is not None and sgrid.get('Rsurf') is not None and grid3d is not None:
        phi_min = grid3d.get('phimin', 0.0) - grid3d.get('rotate_phi_grid', 0.0)
        phi_max = grid3d.get('phimax', 2.0 * np.pi) - grid3d.get('rotate_phi_grid', 0.0)
        step = 0.05
        phi_range_for_plot = np.arange(phi_min, phi_max + step, step)

        for surf_val in plot_ssurf:
            isurf = int(np.argmin(np.abs(sgrid['s_surf'] - surf_val)))
            for phi_val in phi_range_for_plot:
                iphi = 0
                if sgrid.get('nphi', 1) > 1:
                    phi_mod = phi_val % sgrid.get('dphi_sym', 2.0 * np.pi)
                    iphi = int(np.argmin(np.abs(sgrid['phi'] - phi_mod)))

                Rvals, Zvals = sgrid['Rsurf'][isurf, iphi, :], sgrid['Zsurf'][isurf, iphi, :]
                X_loop = Rvals * np.cos(phi_val)
                Y_loop = Rvals * np.sin(phi_val)
                pts = np.vstack((X_loop, Y_loop, Zvals)).T
                all_points.append(pts)

                surf_item = gl.GLLinePlotItem(pos=pts, color=(0.7, 0.7, 0.7, 0.5), width=2, antialias=True)
                w.addItem(surf_item)

    # 4) Draw NBI rays
    if nbi is not None and 'sources' in nbi:
        los_lenses = np.array(spec.get('los_pos', []), dtype=float)
        for source_key in nbi['sources']:
            source = nbi.get(source_key, {})
            if 'uvw_xyz_rot' not in source or 'source_position' not in source: continue
            pos0 = np.array(source['source_position'], dtype=float)
            unit_dir = np.dot(source['uvw_xyz_rot'], np.array([1.0, 0.0, 0.0]))
            
            length = 800.0
            if los_lenses.shape[0] > 0:
                rel_pos = los_lenses - pos0[np.newaxis, :]
                t_vals = rel_pos.dot(unit_dir)
                t_positive = t_vals[t_vals > 0.0]
                if t_positive.size > 0:
                    length = np.max(t_positive) * 1.1

            end = pos0 + length * unit_dir
            all_points.extend([pos0, end])
            ray_item = gl.GLLinePlotItem(pos=np.vstack((pos0, end)), color=(0.0, 0.0, 0.0, 1.0), width=4, antialias=True)
            w.addItem(ray_item)

    # 5) Compute bounding box and setup camera
    if not all_points:
        bbox_min, bbox_max = np.array([-1,-1,-1]), np.array([1,1,1])
    else:
        stacked = np.vstack([p.reshape(-1, 3) for p in all_points])
        bbox_min, bbox_max = stacked.min(axis=0), stacked.max(axis=0)

    max_extent = float(np.max(bbox_max - bbox_min)) if np.max(bbox_max-bbox_min) > 0 else 1.0
    
    grid = gl.GLGridItem()
    grid.scale(max_extent/10, max_extent/10, 1)
    w.addItem(grid)

    w.opts['center'] = pg.Vector((bbox_max + bbox_min) / 2)
    w.opts['distance'] = max_extent * 1.5

# (The other plotting functions: plot_spectra, plot_r_density, plot_u_density, plot_profiles remain the same as my previous answer)
def plot_spectra(plot_widget, spec_main, ilos=0, labels=['full', 'half', 'third', 'halo']):
    pw = plot_widget
    pw.clear()
    pw.addLegend()
    pw.setLabel('bottom', 'Wavelength (nm)')
    pw.setLabel('left', 'Intensity (×10^18 ph/(s sr nm m²))')
    pw.showGrid(x=True, y=True)

    if spec_main is None: return

    nlos = int(spec_main.get('nlos', spec_main.get('intens', np.empty((0,0))).shape[1]))
    if ilos < 0 or ilos >= nlos:
        pw.setTitle(f"Error: Invalid LOS index {ilos}")
        return

    pw.setTitle(f"Spectra for LOS: {spec_main['losname'][ilos]}")
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    max_y = 0.0

    label_map = {'full': 0, 'half': 1, 'third': 2, 'halo': 3}
    for i, label in enumerate(labels):
        j = label_map.get(label)
        if j is not None and j < spec_main['intens'].shape[0]:
            y = spec_main['intens'][j, ilos, :] / 1.0e18
            x = spec_main['wavel']
            if y.size > 0: max_y = max(max_y, float(np.max(y)))
            pen = pg.mkPen(colors[i % len(colors)], width=2)
            pw.plot(x, y, pen=pen, name=label)

    pw.setYRange(0, max_y * 1.1 if max_y > 0 else 1)
    if 'lambda_min' in spec_main and 'lambda_max' in spec_main:
        pw.setXRange(spec_main['lambda_min'], spec_main['lambda_max'])

def plot_r_density(plot_widget, grid_main, labels=['full', 'half', 'third', 'halo']):
    pw = plot_widget
    pw.clear()
    pw.addLegend()
    pw.setLabel('bottom', 'R (cm)')
    pw.setLabel('left', 'n=1 Beam Density (cm⁻³)')
    pw.showGrid(x=True, y=True)
    pw.setTitle('Radial Beam Density')

    if grid_main is None or 'density' not in grid_main or grid_main['density'].ndim < 5:
        return

    density_main = np.sum(
        grid_main['density'][:, 0, :, :, :], axis=(2, 3)
    ) * grid_main.get('dZ', 1.0) * grid_main.get('dphi', 1.0)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    label_map = {'full': 0, 'half': 1, 'third': 2, 'halo': 3}
    for i, label in enumerate(labels):
        j = label_map.get(label)
        if j is not None and j < density_main.shape[0]:
            y = density_main[j, :]
            pen = pg.mkPen(colors[i % len(colors)], width=2)
            pw.plot(grid_main['R_c'], y, pen=pen, name=label)

def plot_u_density(plot_widget, grid_main, labels=['full', 'half', 'third'], eng_lvl=0):
    pw = plot_widget
    pw.clear()
    pw.addLegend()
    pw.setLabel('bottom', 'U (cm)')
    pw.setLabel('left', f'n={eng_lvl+1} Beam Density (cm⁻³)')
    pw.showGrid(x=True, y=True)
    pw.setTitle('Beam-Aligned Density')

    if grid_main is None or 'density_uvw' not in grid_main or grid_main['density_uvw'].ndim != 5:
        pw.setTitle("UVW density not available")
        return

    density_main = np.sum(
        grid_main['density_uvw'][:, eng_lvl, :, :, :], axis=(2, 3)
    ) * grid_main.get('dv', 1.0) * grid_main.get('dw', 1.0)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    label_map = {'full': 0, 'half': 1, 'third': 2}
    for i, label in enumerate(labels):
        j = label_map.get(label)
        if j is not None and j < density_main.shape[0]:
            y = density_main[j, :]
            pen = pg.mkPen(colors[i % len(colors)], width=2)
            pw.plot(grid_main['uc'], y, pen=pen, name=label)

def plot_profiles(layout_widget, profiles):
    """Plots input profiles on a GraphicsLayoutWidget."""
    lw = layout_widget
    lw.clear()
    
    if profiles is None: return
    x_axis_key = 's' if 's' in profiles else 'ra'
    if x_axis_key not in profiles: return
    x_axis = profiles[x_axis_key]

    # Density Plot
    p1 = lw.addPlot(row=0, col=0)
    p1.setLabel('left', 'Density (10^20 m⁻³)')
    p1.addLegend()
    p1.showGrid(x=True, y=True)
    if 'dene' in profiles:
        p1.plot(x_axis, profiles['dene']/1e20, pen='k', name='n_e')
    if 'denp' in profiles:
        p1.plot(x_axis, profiles['denp']/1e20, pen='r', name='n_i')
    if 'denimp' in profiles:
        p1.plot(x_axis, profiles['denimp']/1e20, pen='b', name='n_imp')

    # Temperature Plot
    p2 = lw.addPlot(row=1, col=0)
    p2.setLabel('left', 'Temperature (keV)')
    p2.addLegend()
    p2.showGrid(x=True, y=True)
    if 'te' in profiles:
        p2.plot(x_axis, profiles['te'], pen='k', name='T_e')
    if 'ti' in profiles:
        p2.plot(x_axis, profiles['ti'], pen='r', name='T_i')

    # Rotation Plot
    p3 = lw.addPlot(row=2, col=0)
    p3.setLabel('left', 'Rotation (krad/s)')
    p3.setLabel('bottom', f'{x_axis_key}')
    p3.addLegend()
    p3.showGrid(x=True, y=True)
    if 'omega' in profiles:
        p3.plot(x_axis, profiles['omega']/1e3, pen='g', name='Omega')

    # Link X axes
    p2.setXLink(p1)
    p3.setXLink(p1)