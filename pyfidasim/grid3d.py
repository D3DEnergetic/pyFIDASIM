import numpy as np

try:
    import numba
    numba_is_available = True
except:
    numba_is_available = False
    print('Numba is not available; consider installing Numba')

def conditional_numba(skip_numba=False):
    def decorator(func):
        if numba_is_available and not skip_numba:
            return numba.jit(func, 
                             cache=True, 
                             nopython=True, 
                             nogil=True)
        else:
            return func
    return decorator

@conditional_numba(skip_numba=False)
def get_grid3d_indices(xyzc_list, g3_rotate_phi_grid, g3_dphi_sym, 
                                        g3_Rmin, g3_dR, g3_nR, g3_Zmin, g3_dZ, g3_nZ, 
                                        g3_phimin, g3_dphi, g3_nphi):
    '''
    Function to process a list of xyzc_arr arrays.
    '''

    # Number of arrays in xyzc_list
    num_arrays = xyzc_list.shape[0]
    num_cols = xyzc_list.shape[2]

    # Pre-allocate numpy arrays
    ir_array = np.zeros((num_arrays, num_cols), dtype=np.int16)
    iz_array = np.zeros((num_arrays, num_cols), dtype=np.int16)
    iphi_array = np.zeros((num_arrays, num_cols), dtype=np.int16)
    
    for idx, xyzc_arr in enumerate(xyzc_list):
        rpos = np.sqrt(xyzc_arr[0, :]**2 + xyzc_arr[1, :]**2)
        zpos = xyzc_arr[2, :]

        ir = np.floor((rpos - g3_Rmin) / g3_dR).astype(np.int16)
        iz = np.floor((zpos - g3_Zmin) / g3_dZ).astype(np.int16)

        phipos = np.arctan2(xyzc_arr[1, :], xyzc_arr[0, :]) + g3_rotate_phi_grid
        phipos[phipos < 0] += 2. * np.pi
        phipos2 = phipos % g3_dphi_sym
        iphi = np.floor((phipos2 - g3_phimin) / g3_dphi).astype(np.int64)

        # Assign to the pre-allocated arrays
        ir_array[idx, :] = ir
        iz_array[idx, :] = iz
        iphi_array[idx, :] = iphi

    return ir_array, iz_array, iphi_array

@conditional_numba(skip_numba=True)
def get_uvw_grid_indices(uvw_start,uvw_ray,dl,first_step,\
                          g3_umin,g3_du, \
                          g3_vmin,g3_dv, \
                          g3_wmin,g3_dw, \
                          g3_nu,g3_nv,g3_nw):
    """
    Calculate grid cell indices for a beam-aligned UVW grid in a vectorized manner.

    Parameters:
    - uvw_start (np.ndarray): Starting UVW coordinates for each beam. Shape: (N, 3)
    - uvw_ray (np.ndarray): UVW direction vectors for each beam. Shape: (N, 3)
    - dl (np.ndarray): Differential steps along each beam. Shape: (N, M)
    - first_step (np.ndarray): Initial step distance for each beam. Shape: (N,)
    - g3_umin, g3_du, g3_vmin, g3_dv, g3_wmin, g3_dw (float): Grid parameters.
    - g3_nu, g3_nv, g3_nw (int): Number of grid cells along U, V, W axes.

    Returns:
    - iu_arr, iv_arr, iw_arr (np.ndarray): Grid indices along U, V, W axes respectively. Shape: (N, M)
    """
    
    #uvw_ray[:,0]=np.abs(uvw_ray[:,0])
    
    # Compute the cumulative sum of dl along the steps for each beam
    cumsum_dl = np.cumsum(dl, axis=1)  # Shape: (N, M)

    # Compute the total distance for each step: first_step + cumulative dl
    total_distance = first_step[:, np.newaxis] + cumsum_dl  # Shape: (N, M)

    # Calculate UVW coordinates for each beam and step
    # Resulting shape: (N, M, 3)
    uvw_coord = uvw_start[:, np.newaxis, :] + uvw_ray[:, np.newaxis, :] * total_distance[:, :, np.newaxis]

    # Compute grid indices by translating UVW coordinates to grid cell indices
    iu_arr = np.floor((uvw_coord[:, :, 0] - g3_umin) / g3_du).astype(np.int64)  # Shape: (N, M)
    iv_arr = np.floor((uvw_coord[:, :, 1] - g3_vmin) / g3_dv).astype(np.int64)  # Shape: (N, M)
    iw_arr = np.floor((uvw_coord[:, :, 2] - g3_wmin) / g3_dw).astype(np.int64)  # Shape: (N, M)

    return iu_arr, iv_arr, iw_arr
    

def define_grid3d(fields, drz=2., nphi = None, u_range=None,du=1.,dvw=1., v_width=5,w_width=5, r_ran = None, z_ran = None, phi_ran = None):


    grid3d = {}
    grid3d['nsym'] = fields['nsym']
    grid3d['dphi_sym'] = fields['dphi_sym']
    if phi_ran:
        grid3d['phimin'] =  np.float64(phi_ran[0])
        grid3d['phimax'] = np.float64(phi_ran[1])
    else:
        grid3d['phimin'] = fields['phimin']
        grid3d['phimax'] = fields['phimax']
    
    
    if nphi:
        grid3d["nphi"] = nphi
    else:
        if r_ran:
            dphi_initial = 2*drz / (r_ran[0]+r_ran[1])
        else:    
            dphi_initial = drz / fields['Rmean']
        grid3d['nphi'] = int((grid3d['phimax']-grid3d['phimin'])/dphi_initial+0.5)
        
    grid3d['dphi'] = (grid3d['phimax']-grid3d['phimin'])/grid3d['nphi']
    grid3d['phi_bnd'] = np.linspace(grid3d['phimin'],grid3d['phimax'],grid3d['nphi'],endpoint=False)
    # grid center positions
    grid3d['phi_c'] = grid3d['phi_bnd'] + 0.5 * grid3d['dphi']
    grid3d['rotate_phi_grid'] = fields['rotate_phi_grid']   
    # define R-output array
    if r_ran:
        grid3d["Rmin"] = np.float64(r_ran[0])
        grid3d["Rmax"] = np.float64(r_ran[1])
    else:
        grid3d['Rmin'] = fields['Rmin']
        grid3d['Rmax'] = fields['Rmax']
    grid3d['nR']=int((grid3d['Rmax']-grid3d['Rmin'])/drz+0.5) 
    grid3d['dR'] = (grid3d['Rmax']-grid3d['Rmin'])/grid3d['nR']
    # grid boundary position
    grid3d['R_bnd'] = np.linspace(grid3d['Rmin'], grid3d['Rmax'], grid3d['nR'],endpoint=False)
    # grid center positions
    grid3d['R_c'] = grid3d['R_bnd'] + 0.5 * grid3d['dR']


    # define Z-output array   
    if z_ran:
        grid3d["Zmin"] = z_ran[0]
        grid3d["Zmax"] = z_ran[1]
    else:
        grid3d['Zmin'] = fields['Zmin']
        grid3d['Zmax'] = fields['Zmax']
    grid3d['nZ']=int((grid3d['Zmax']-grid3d['Zmin'])/drz+0.5) 
    grid3d['dZ'] = (grid3d['Zmax']-grid3d['Zmin'])/grid3d['nZ']
    # grid boundary positions
    grid3d['Z_bnd'] =  np.linspace(grid3d['Zmin'], grid3d['Zmax'], grid3d['nZ'],endpoint=False)
    # grid center positions
    grid3d['Z_c'] = grid3d['Z_bnd'] + 0.5 * grid3d['dZ']
    
    grid3d['dvol'] = grid3d['dZ'] * 0.5*grid3d['dphi']*((grid3d['R_c']+0.5*grid3d['dR'])**2-(grid3d['R_c']-0.5*grid3d['dR'])**2)

    ## beam aligned grid       
    if u_range:
        ## define parameters of a beam-aligned grid (uvw coordinates)
        grid3d['vmin']=-0.5*v_width
        grid3d['vmax']=0.5*v_width
        grid3d['wmin']=-0.5*w_width
        grid3d['wmax']=0.5*w_width
        grid3d['umin']=u_range[0]
        grid3d['umax']=u_range[1]   
        
        grid3d['nu']=int((grid3d['umax']-grid3d['umin'])/du)   
        grid3d['nv']=int((grid3d['vmax']-grid3d['vmin'])/dvw)

        grid3d['nw']=int((grid3d['wmax']-grid3d['wmin'])/dvw)
        grid3d['du']=(grid3d['umax']-grid3d['umin'])/grid3d['nu']
        grid3d['dv']=(grid3d['vmax']-grid3d['vmin'])/grid3d['nv']
        grid3d['dw']=(grid3d['wmax']-grid3d['wmin'])/grid3d['nw']  
        

        grid3d['uc']=np.linspace(grid3d['umin']+0.5*grid3d['du'],grid3d['umax']-0.5*grid3d['du'],grid3d['nu'])
        grid3d['vc']=np.linspace(grid3d['vmin']+0.5*grid3d['dv'],grid3d['vmax']-0.5*grid3d['dv'],grid3d['nv'])    
        grid3d['wc']=np.linspace(grid3d['wmin']+0.5*grid3d['dw'],grid3d['wmax']-0.5*grid3d['dw'],grid3d['nw'])       
    else:
        grid3d['vmin']=0
        grid3d['vmax']=0
        grid3d['wmin']=0
        grid3d['wmax']=0
        grid3d['umin']=0
        grid3d['umax']=0
        
        grid3d['nu']=1
        grid3d['nv']=1
        grid3d['nw']=1
        grid3d['du']=1
        grid3d['dv']=1
        grid3d['dw']=1
        grid3d['uc']=np.zeros(1)
        grid3d['vc']=np.zeros(1)   
        grid3d['wc']=np.zeros(1)       
        
    return grid3d
