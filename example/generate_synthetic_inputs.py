"""
generate_synthetic_inputs.py
============================
Configuration Builder for pyFIDASIM.

This module generates the dictionary input structure used by pyFIDASIM.
It mirrors the structure expected by 'input_prep.py'.
"""

import numpy as np

# =============================================================================
# 1. USER CONFIGURATION (EDITABLE)
# =============================================================================
def get_default_config():
    """
    Returns the default dictionary of user-adjustable parameters.
    Keys are aligned with input_prep.py to serve as a tutorial.
    """
    config = {
        # --- General Simulation Settings ---
        'simulation': {
            'nmarker': 50000,
            'calc_halo': True,
            'calc_spectra': True,
            'calc_rzp_dens': True,
            'calc_uvw_dens': True,
            'calc_photon_origin': False,
            'calc_photon_origin_type': False,
            'calc_PSF': False,
            'separate_dcx': False,
            'calc_extended_emission': False,
            'respawn_if_aperture_is_hit': True,
            'seed': 12345,
            'batch_marker': 50000,
            'verbose': True
        },

        # --- Grid Definitions ---
        'grid': {
            'grid_drz': 2.0,       # Spatial resolution [cm]
            'r_ran': [90.0, 210.0], # R range [min, max]
            'z_ran': [-100.0, 100.0], # Z range [min, max]
            'phi_ran': [0.6 * np.pi, 1.4 * np.pi], # Toroidal range
            'u_range': [-100.0, 700.0], # Beam grid length
            'v_width': 50.0,
            'w_width': 50.0,
            'du': 1.0, # Beam grid resolution
            'dvw': 1.0
        },

        # --- Machine Geometry (Synthetic Miller) ---
        # Note: In real cases, this comes from GEQDSK/WOUT files.
        'machine': {
            'R0': 150.0,
            'a': 50.0,
            'B0': 2.0,
            'kappa': 1.5,
            'delta': 0.5
        },

        # --- Profiles (Synthetic) ---
        # Note: In real cases, these come from TRANSP/Text files.
        'profiles': {
            'ne_axis': 8e13, 'ne_ped': 4e13, 'ne_sep': 1e12,
            'te_axis': 3.0,  'te_ped': 1.0,  'te_sep': 0.05,
            'ti_axis': 3.5,  'ti_ped': 1.2,  'ti_sep': 0.05,
            'omega_axis': 5e4,
            'zeff': 1.6,
            'ion_mass': 2.0  # Deuterium plasma
        },

        # --- NBI Definition ---
        'nbi': {
            'ab': 2.0, # Deuterium Beam
            'sources': ['SRC_1'],
            'SRC_1': {
                'power': 2.0,       # MW
                'voltage': 80.0,    # kV
                'current_fractions': [0.6, 0.3, 0.1],
                'source_position': [-130.0, -350.0, 0.0],
                'target_position': [-130.0, 350.0, 0.0], # Used to calc direction
                'divergence': [0.015, 0.015],
                'focal_length': [400., 400.],
                'ion_source_size': [10., 20.],
                
                # Aperture 1
                'aperture_1_distance': 100.,
                'aperture_1_size': [20., 50.],
                'aperture_1_offset': [0., 0.],
                'aperture_1_rectangular': True,
                
                # Aperture 2
                'aperture_2_distance': 150.,
                'aperture_2_size': [20., 50.],
                'aperture_2_offset': [0., 0.],
                'aperture_2_rectangular': True,
            }
        },

        # --- Spectroscopy ---
        'spec': {
            'nlos': 20,
            'lens_center': [-280.0, -150.0, 0.0],
            'target_y_start': -150.0,
            'target_y_end': -50.0,
            'target_x': -130.0, # Where LOS intersects beam
            'lambda_min': 650.0,
            'lambda_max': 665.0,
            'dlam': 0.02
        }
    }
    return config

# =============================================================================
# 2. INTERNAL CALCULATIONS (HIDDEN)
# =============================================================================

def _calc_rot(direction):
    """Hidden: Calculate rotation matrix (UVW -> XYZ)."""
    norm = np.linalg.norm(direction)
    if norm == 0: return np.eye(3)
    direction = direction / norm
    
    x_sq_y_sq = np.sqrt(direction[0]**2 + direction[1]**2)
    b = np.arctan2(direction[2], x_sq_y_sq)
    Arot = np.array([[np.cos(b), 0., np.sin(b)],[0., 1., 0.],[-np.sin(b), 0., np.cos(b)]])
    
    a = np.arctan2(direction[1], direction[0])
    Brot = np.array([[np.cos(a), -np.sin(a), 0.],[np.sin(a),  np.cos(a), 0.],[0., 0., 1.]])
    return Brot @ Arot

def _solve_miller_s(R_grid, Z_grid, R0, a, kappa, delta):
    """Hidden: Numerical inversion of Miller equilibrium."""
    R_eff = R_grid - R0
    rho = np.sqrt((R_eff/a)**2 + (Z_grid/(kappa*a))**2)
    theta = np.arctan2(Z_grid / kappa, R_eff)
    
    # Newton Iteration
    mask_solve = rho < 3.0
    for i in range(5):
        r_c, t_c = rho[mask_solve], theta[mask_solve]
        sin_t, cos_t = np.sin(t_c), np.cos(t_c)
        T = t_c + delta * sin_t
        
        f1 = (R0 + a*r_c*np.cos(T)) - R_grid[mask_solve]
        f2 = (kappa*a*r_c*sin_t) - Z_grid[mask_solve]
        
        dT_dt = 1 + delta*cos_t
        J11, J12 = a*np.cos(T), -a*r_c*np.sin(T)*dT_dt
        J21, J22 = kappa*a*sin_t, kappa*a*r_c*cos_t
        det = J11*J22 - J12*J21
        
        valid = np.abs(det) > 1e-5
        dr = -(J22*f1 - J12*f2)/det
        dt = -(-J21*f1 + J11*f2)/det
        
        r_c[valid] += np.clip(dr[valid], -0.5, 0.5)
        t_c[valid] += np.clip(dt[valid], -0.5, 0.5)
        
        rho[mask_solve] = r_c
        theta[mask_solve] = t_c
        
    s = rho**2
    s = np.abs(s)
    s[np.isnan(s)] = 2.0
    return s

def _hybrid_profile(s, val_core, val_edge, val_wall, s_split=0.9):
    """Hidden: Profile generation logic."""
    prof = np.zeros_like(s)
    mask_core = s < s_split
    
    # Cubic Core
    B = (val_edge - val_core) / (s_split**3)
    prof[mask_core] = val_core + B * (s[mask_core]**3)
    
    # Tanh Edge
    mask_edge = ~mask_core
    decay_len = 0.05
    
    prof[mask_edge] = val_wall + (val_edge - val_wall) * (1.0 - np.tanh((s[mask_edge] - s_split) / decay_len))
    
    prof[prof < 1e-4] = 1e-4 # Positivity
    return prof

# =============================================================================
# 3. DATA BUILDER
# =============================================================================
def build_simulation_data(config):
    """
    Takes the user configuration and returns the heavy data dictionaries 
    (sim_settings, fields, profiles, nbi, spec) required by pyFIDASIM.
    """
    print("--- Building Simulation Data from Configuration ---")
    c_grid = config['grid']
    c_mach = config['machine']
    c_prof = config['profiles']
    c_nbi  = config['nbi']
    c_spec = config['spec']

    # 1. Grid & Fields
    R0, a, B0 = c_mach['R0'], c_mach['a'], c_mach['B0']
    kappa, delta = c_mach['kappa'], c_mach['delta']
    
    nr = int((c_grid['r_ran'][1] - c_grid['r_ran'][0]) / c_grid['grid_drz'])
    nz = int((c_grid['z_ran'][1] - c_grid['z_ran'][0]) / c_grid['grid_drz'])
    
    R_avg = (c_grid['r_ran'][0] + c_grid['r_ran'][1])/2
    dphi_approx = c_grid['grid_drz'] / R_avg
    nphi = int((c_grid['phi_ran'][1] - c_grid['phi_ran'][0]) / dphi_approx)
    
    R = np.linspace(c_grid['r_ran'][0], c_grid['r_ran'][1], nr)
    Z = np.linspace(c_grid['z_ran'][0], c_grid['z_ran'][1], nz)
    phi = np.linspace(c_grid['phi_ran'][0], c_grid['phi_ran'][1], nphi, endpoint=False)
    
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    s_2d = _solve_miller_s(RR, ZZ, R0, a, kappa, delta)
    
    s = np.repeat(s_2d[:, :, np.newaxis], nphi, axis=2)
    s[s > 1.3] = 1.3
    
    # Fields
    RR_3d = np.repeat(RR[:, :, np.newaxis], nphi, axis=2)
    ZZ_3d = np.repeat(ZZ[:, :, np.newaxis], nphi, axis=2)
    
    Bphi = B0 * R0 / RR_3d
    Bz = 0.1 * B0 * ((RR_3d - R0) / a)
    Br = -0.1 * B0 * (ZZ_3d / (kappa*a))
    Er = np.zeros_like(Br)
    
    # Visualization Surfaces
    n_s, n_theta = 50, 100
    s_surf_1d = np.linspace(0, 1.0, n_s)
    theta_surf = np.linspace(0, 2*np.pi, n_theta)
    miller_ang = theta_surf + delta*np.sin(theta_surf)
    R_surf = (R0 + np.sqrt(s_surf_1d)[:,None]*a*np.cos(miller_ang))[:, np.newaxis, :]
    Z_surf = (np.sqrt(s_surf_1d)[:,None]*a*kappa*np.sin(theta_surf))[:, np.newaxis, :]

    fields = {
        'R': R, 'Z': Z, 'phi': phi,
        'Rmin': R[0], 'Rmax': R[-1], 'dR': R[1]-R[0], 'nr': nr,
        'Zmin': Z[0], 'Zmax': Z[-1], 'dZ': Z[1]-Z[0], 'nz': nz,
        'phimin': phi[0], 'phimax': phi[-1] + (phi[1]-phi[0]), 'dphi': phi[1]-phi[0], 'nphi': nphi,
        's': s, 'Br': Br, 'Bz': Bz, 'Bphi': Bphi, 'Er': Er,
        'nsym': 1, 'dphi_sym': 2*np.pi, 'rotate_phi_grid': 0.0,
        'flux_s_min': 0.0, 'flux_ds': 0.01, 'flux_nr': 100,
        'flux_ns': 100, 'flux_dvol': 1.0, 
        's_surf': s_surf_1d, 'Rsurf': R_surf, 'Zsurf': Z_surf, 'Rmean': R0, 'btipsign': -1
    }

    # 2. Profiles
    s_prof = np.linspace(0, 1.3, 100)
    ne = _hybrid_profile(s_prof, c_prof['ne_axis'], c_prof['ne_ped'], c_prof['ne_sep'])
    te = _hybrid_profile(s_prof, c_prof['te_axis'], c_prof['te_ped'], c_prof['te_sep'])
    ti = _hybrid_profile(s_prof, c_prof['ti_axis'], c_prof['ti_ped'], c_prof['ti_sep'])
    omega = c_prof['omega_axis'] * (1.0 - s_prof**3)
    omega[omega < 0] = 0.0
    
    zeff = np.full_like(s_prof, c_prof['zeff'])
    zimp = 6
    nimp = ne * (zeff - 1) / (zimp * (zimp - 1))
    ni = ne - nimp * zimp
    
    profiles = {
        's': s_prof, 'te': te, 'ti': ti, 'dene': ne, 'denp': ni, 
        'denimp': nimp[np.newaxis, :], 'omega': omega, 'ai': c_prof['ion_mass']
    }

    # 3. NBI
    nbi_out = {'sources': c_nbi['sources'], 'ab': c_nbi['ab']}
    for src_name in c_nbi['sources']:
        s_conf = c_nbi[src_name]
        
        src_pos = np.array(s_conf['source_position'])
        tgt_pos = np.array(s_conf['target_position'])
        direction = tgt_pos - src_pos
        direction /= np.linalg.norm(direction)
        
        nbi_out[src_name] = s_conf.copy()
        nbi_out[src_name]['direction'] = direction
        nbi_out[src_name]['uvw_xyz_rot'] = _calc_rot(direction)
        for k in ['current_fractions', 'divergence', 'focal_length', 'ion_source_size', 
                  'aperture_1_size', 'aperture_1_offset', 'aperture_2_size', 'aperture_2_offset']:
            nbi_out[src_name][k] = np.array(s_conf[k])

    # 4. Spectroscopy
    nlos = c_spec['nlos']
    lens = np.array(c_spec['lens_center'])
    target_ys = np.linspace(c_spec['target_y_start'], c_spec['target_y_end'], nlos)
    
    los_pos = np.tile(lens, (nlos, 1))
    los_vec = []
    names = []
    
    for y_t in target_ys:
        intersect = np.array([c_spec['target_x'], y_t, 0.0])
        vec = intersect - lens
        vec /= np.linalg.norm(vec)
        los_vec.append(vec)
        
        px, py = lens[0], lens[1]
        ux, uy = vec[0], vec[1]
        t = -(px*ux + py*uy)/(ux**2 + uy**2)
        r = np.sqrt((px + t*ux)**2 + (py + t*uy)**2)
        names.append(f"Ch (Rtan={r:.0f})")
        
    spec = {
        'nlos': nlos, 'los_pos': los_pos, 'los_vec': np.array(los_vec), 'losname': names,
        'lambda_min': c_spec['lambda_min'], 'lambda_max': c_spec['lambda_max'], 
        'dlam': c_spec['dlam'],
        'nlam': int((c_spec['lambda_max']-c_spec['lambda_min'])/c_spec['dlam']),
        'wavel': np.arange(c_spec['lambda_min'], c_spec['lambda_max'], c_spec['dlam']),
        'sigma_to_pi_ratio': np.ones(nlos),
        'output_individual_stark_lines': False,
        'los_grid_intersection_weight': None, 
        'los_f_len': np.array([0.0])
    }

    return config['simulation'], fields, profiles, nbi_out, spec