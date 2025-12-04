import numpy as np
import scipy.constants as consts
from .toolbox import rotate_uvw, cross_line_plane
from .pyfidasim_core_routine import pyfidasim_core_routine


# -----------------------------------------------------------------------------------
# ----------------------- Beam Attenuation Simulation -------------------------------
# -----------------------------------------------------------------------------------
def get_nbi_source_vectors(geom, nmarker=1, respawn_if_aperture_is_hit=True):
    """
    Initializes random NBI markers based on NBI geometry.

    Parameters
    ----------
    geom : dict
        NBI geometry dictionary.
    nmarker : int, optional
        Number of markers to generate, by default 1.
    respawn_if_aperture_is_hit : bool, optional
        Respawn markers if aperture is hit, by default True.

    Returns
    -------
    tuple of np.ndarray
        - xyz_start: Initial start positions at the NBI source in xyz coordinates.
        - xyz_ray: Initial directions in xyz coordinates.
    """
    MAX_ATTEMPTS = 10000 if respawn_if_aperture_is_hit else 1

    half_ion_source_size_v = geom['ion_source_size'][0] / 2.0
    half_ion_source_size_w = geom['ion_source_size'][1] / 2.0
    aperture_1_dist_half_size = 0.5 * geom['aperture_1_size']
    aperture_2_dist_half_size = 0.5 * geom['aperture_2_size']
    aperture_1_plane = np.array([-geom['aperture_1_distance'], 0.0, 0.0])
    aperture_2_plane = np.array([-geom['aperture_2_distance'], 0.0, 0.0])
    v_norm = np.array([1.0, 0.0, 0.0])

    xyz_starts = []
    xyz_rays = []

    attempts = 0
    while len(xyz_starts) < nmarker and attempts < MAX_ATTEMPTS:
        attempts += 1

        # Generate random start position within the ion source
        uvw_start = np.array([
            0.0,
            np.random.uniform(-half_ion_source_size_v, half_ion_source_size_v),
            np.random.uniform(-half_ion_source_size_w, half_ion_source_size_w)
        ])

        # Generate random direction with divergence
        randomn = np.random.randn(2)
        uvw_ray = np.array([
            1.0,
            -(uvw_start[1] / geom['focal_length'][0]) + np.tan(geom['divergence'][0] * randomn[0]),
            -(uvw_start[2] / geom['focal_length'][1]) + np.tan(geom['divergence'][1] * randomn[1])
        ])
        uvw_ray /= np.linalg.norm(uvw_ray)

        # Check for aperture hits
        hit_aperture = False
        for aperture_plane, offset, size, is_rectangular in [
            (aperture_1_plane, geom['aperture_1_offset'], aperture_1_dist_half_size, geom['aperture_1_rectangular']),
            (aperture_2_plane, geom['aperture_2_offset'], aperture_2_dist_half_size, geom['aperture_2_rectangular'])
        ]:
            cross_point = cross_line_plane(uvw_start, uvw_ray, aperture_plane, v_norm)

            if is_rectangular:
                if np.any(np.abs(cross_point[1:3] - offset[:2]) > size):
                    hit_aperture = True
                    break
            else:
                if np.linalg.norm(cross_point[1:3] - offset[:2]) > size[0]:
                    hit_aperture = True
                    break

        if not hit_aperture:
            # Valid ray found
            xyz_ray = rotate_uvw(geom['uvw_xyz_rot'], uvw_ray)
            xyz_start = rotate_uvw(geom['uvw_xyz_rot'], uvw_start) + geom['source_position']
            xyz_starts.append(xyz_start)
            xyz_rays.append(xyz_ray)
            attempts = 0  # Reset attempts after a successful spawn

    return np.array(xyz_starts), np.array(xyz_rays)

def get_nbi_vector_batch(batch_size, total_size, source, respawn_if_aperture_is_hit):
    """
    Generator that yields batches of start_pos and vnorm arrays.
    
    Parameters:
    - batch_size: Number of entries per batch.
    - total_size: Total number of entries to generate.
    - generate_start_pos_func: Function to generate start_pos given a batch size.
    - generate_vnorm_func: Function to generate vnorm given a batch size.
    
    Yields:
    - Tuple of (start_pos_batch, vnorm_batch) for each batch.
    """
    for start_idx in range(0, total_size, batch_size):
        current_batch_size = min(batch_size, total_size - start_idx)
        start_pos_batch, vnorm_batch = get_nbi_source_vectors(
            source, nmarker=current_batch_size,
            respawn_if_aperture_is_hit=respawn_if_aperture_is_hit
        )
        yield start_pos_batch, vnorm_batch

def calc_attenuation(sim_settings,profiles=None, nbi=None, spec=None, fields=None, grid3d=None, tables=None,
                      ncdf=None, PSF=None, fbm=None,im_blur=None):
    """
    Calculates beam attenuation based on provided simulation parameters.

    Parameters
    ----------
    profiles : dict, optional
        Plasma profiles.
    nbigeom : dict, optional
        NBI geometry.
    nbiparams : dict, optional
        NBI parameters.
    spec : dict, optional
        Spectral settings.
    fields : dict, optional
        Spectral grid settings.
    grid3d : dict, optional
        3D grid settings.
    tables : dict, optional
        Simulation tables.
    nmarker : int, optional
        Number of markers, by default 100.
    halo : bool, optional
        Enable halo simulation, by default False.
    verbose : bool, optional
        Enable verbose output, by default True.
    calc_photon_origin : bool, optional
        Calculate photon origin, by default False.
    calc_density : bool, optional
        Calculate density, by default True.
    respawn_if_aperture_is_hit : bool, optional
        Respawn markers if aperture is hit, by default True.
        Trevor NCDF settings.
    PSF : dict, optional
        Point Spread Function settings, by default None.

    Returns
    -------
    tuple
        Updated grid3d and spec dictionaries with calculated attenuation data.
    """

    # Initialize random seed for reproducibility
    if sim_settings['seed'] >= 0:
        seed = sim_settings['seed']
        np.random.seed(seed)
    else:
        np.random.seed()
        seed = -1

    # Halo simulation settings
    ncx_iter = 120 if sim_settings['calc_halo'] else 1
    if sim_settings['calc_halo'] and sim_settings['verbose']:
        print('Halo simulation enabled with {} iterations.'.format(ncx_iter))
    
    if sim_settings['calc_halo']:
        if sim_settings['separate_dcx']:
            h = 2
        else:
            h = 1
    else:
        h = 0
        
    if fbm['afbm'] > 0:
        h += 1
        
    # Define step-length of tracking
    step_length = 0.4 * (grid3d['dR'] + grid3d['dZ'])

    # Initialize storage arrays
    density1d = np.zeros((6, fields['flux_ns']))
    if sim_settings['calc_rzp_dens']:
        density = np.zeros((3 + h, 6, grid3d['nR'], grid3d['nZ'], grid3d['nphi']))
    else:
        density = np.zeros((1, 1, 1, 1, 1))
    if sim_settings['calc_uvw_dens']:
        density_uvw=np.zeros((3 + h,6,grid3d['nu'],grid3d['nv'], grid3d['nw']))
    else:
        density_uvw=np.zeros((1,1,1,1,1))
        
    if sim_settings['calc_spectra']:
        if spec['output_individual_stark_lines'] and ncdf['active']:
            intensity_shape = (3 + h, spec['nlos'], ncdf['trans'], spec['nlam'])
        elif spec['output_individual_stark_lines']:
            intensity_shape = (3 + h, spec['nlos'], 15, spec['nlam'])
        else:
            intensity_shape = (3 + h, spec['nlos'], 1, spec['nlam'])

        intensity = np.zeros(intensity_shape)
    else:
        intensity = np.zeros((1, 1, 1, 1))

    # Define the result array dimension for the 2D photon origin array
    if sim_settings['calc_photon_origin']:
        if not sim_settings['calc_photon_origin_type']:
            photon_origin = np.zeros((1, spec['nlos'], grid3d['nR'],grid3d['nZ'], grid3d['nphi']))
        else:
            photon_origin = np.zeros((3 + h, spec['nlos'], grid3d['nR'],grid3d['nZ'], grid3d['nphi']))
    else:
        photon_origin = np.zeros((1, 1, 1, 1, 1))

    # Iterate over NBI sources
    for source_key in nbi['sources']:
        if sim_settings['verbose']:
            print(f"NBI Source: {source_key}")
        source = nbi[source_key]

        # Precompute inverse rotation matrices
        source['rot_inv'] = np.linalg.inv(source['uvw_xyz_rot'])

        if source['power'] == 0:
            continue

        # Calculate number of neutrals
        nneutrals = 1e6 * source['power'] / (1e3 * source['voltage'] * consts.e * (
            source['current_fractions'][0] +
            source['current_fractions'][1] / 2.0 +
            source['current_fractions'][2] / 3.0
        ))
        source['nneutrals'] = nneutrals

        # Determine number of markers per energy component
        nmarker_arr = (sim_settings['nmarker'] * np.array(source['current_fractions'][:])).astype(int)

        # Number of energy components
        nenergy = 3

        for ikind in range(nenergy):
            if sim_settings['verbose']:
                if ikind == 0:
                    print("NBI Energy Fraction: Full")
                elif ikind == 1:
                    print("NBI Energy Fraction: Half")
                elif ikind ==2:
                    print("NBI Energy Fraction: One-third")
            if source['current_fractions'][ikind] == 0:
                continue

            E0 = np.array(source['voltage']) / (ikind + 1) * 1e3  # [eV]
            factor = nneutrals * source['current_fractions'][ikind] / nmarker_arr[ikind]
            
            current_marker = 0
            # Get initial NBI particle positions and directions
            data_gen = get_nbi_vector_batch(
                sim_settings['batch_marker'],nmarker_arr[ikind],source,
                respawn_if_aperture_is_hit=sim_settings['respawn_if_aperture_is_hit']
            )

            for batch_num, (start_pos, vnorm) in enumerate(data_gen, start=1):
                if sim_settings['verbose']:
                    print(f"Marker Progress: {current_marker/nmarker_arr[ikind]*100:.2f}%")
                    current_marker += np.shape(start_pos)[0]
                jump = source['aperture_2_distance']
                mass = consts.atomic_mass * nbi['ab']
                vabs = np.sqrt(2.0 * E0 * consts.e / mass) * 100.0  # cm/s
    
                # Initialize states: start with neutral in the ground state (n=1)
                states = np.zeros((start_pos.shape[0], tables['levels']))
                states[:, 0] = 1.0
    
                # Call the core simulation routine
                density, density_uvw, intensity, photon_origin_out, density1d = pyfidasim_core_routine(
                    fields['rotate_phi_grid'], fields['dphi_sym'], fields['s'],
                    fields['Rmin'], fields['Rmax'], fields['R'], fields['dR'], fields['nr'],
                    fields['Zmin'], fields['Zmax'], fields['Z'], fields['dZ'], fields['nz'],
                    fields['phimin'], fields['phimax'], fields['phi'], fields['dphi'], fields['nphi'],
                    fields['flux_ns'], fields['flux_ds'], fields['flux_s_min'], fields['Br'], fields['Bz'], fields['Bphi'], fields['Er'],
                    grid3d['rotate_phi_grid'], grid3d['dphi_sym'], grid3d['dvol'],
                    grid3d['Rmin'], grid3d['nR'], grid3d['dR'], grid3d['Zmin'], grid3d['nZ'], grid3d['dZ'],
                    grid3d['phimin'], grid3d['dphi'], grid3d['nphi'],
                    grid3d['umin'], grid3d['du'], grid3d['vmin'], grid3d['dv'], grid3d['wmin'],
                    grid3d['dw'], grid3d['nu'], grid3d['nv'], grid3d['nw'],
                    spec['grid_cell_crossed_by_los'], spec['dl_per_grid_intersection'],
                    spec['los_grid_intersection_indices'], spec['los_grid_intersection_weight'],
                    spec['nlos'], spec['los_pos'], spec['los_vec'],
                    spec['lambda_min'], spec['nlam'], spec['dlam'], spec['sigma_to_pi_ratio'],
                    spec['output_individual_stark_lines'],
                    fbm['fbm'],fbm['denf'],fbm['afbm'],fbm['btipsign'], 
                    fbm['emin'],fbm['eran'],fbm['nenergy'],fbm['energy'],fbm['dE'], 
                    fbm['pmin'],fbm['pran'],fbm['npitch'],fbm['pitch'], fbm['dP'], 
                    profiles['s'], profiles['te'], profiles['ti'], profiles['dene'], profiles['denp'],
                    profiles['denimp'], profiles['omega'], profiles['ai'],
                    tables['levels'], tables['energy_ax'], tables['temp_ax'],
                    len(tables['impurities']), tables['zimps'], tables['impurities'],
                    tables['qptable_no_cx'], tables['qptable'], tables['qetable'], tables['qitables'],
                    tables['einstein'], tables['neutrates'],
                    source['direction'], source['source_position'], source['rot_inv'],
                    ikind, factor, states, start_pos, jump, step_length, vnorm, vabs, E0, mass,
                    ncx_iter, seed, sim_settings['calc_spectra'],sim_settings['calc_photon_origin'],
                    sim_settings['calc_photon_origin_type'], sim_settings['calc_PSF'],sim_settings['calc_rzp_dens'], 
                    sim_settings['calc_uvw_dens'], density_uvw, sim_settings['separate_dcx'],
                    density, intensity, photon_origin, density1d,
                    ncdf['cdfvars'], ncdf['lambda0'], ncdf['trans'], ncdf['l_to_dwp'], ncdf["spectrum_extended"],
                    PSF['los_image_arr'],PSF['n_rand'],PSF['image_pos'],PSF['image_vec'], PSF['image_blur'],PSF['f_lens']
                )
            
            if sim_settings['verbose']:
                print("Marker Progress: 100%")

    # Process intensity
    if intensity.shape[2] == 1:
        intensity = intensity[:, :, 0, :]
    spec['intens'] = intensity / spec['dlam'] / (4 * np.pi) * 1e4  # [ph/s/sr/nm/m^2]
    if sim_settings['calc_spectra']:
        spec['photon_origin'] = photon_origin_out / (4 * np.pi) * 1e4  # [ph/s/sr/m^2]
    grid3d['density'] = density
    grid3d['density_uvw']=density_uvw

    # Normalize density1d
    for istate in range(6):
        density1d[istate, :] /= fields['flux_dvol']

    return grid3d, spec
