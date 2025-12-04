import numpy as np
from .grid3d import get_grid3d_indices
from .toolbox import ER_Matrix
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
                             nogil=True,
                             debug = False
                             )
        else:
            return func
    return decorator

@conditional_numba(skip_numba=True)
def point_samp(npoint,point_arr,d_point):
    f_array = np.zeros((3,npoint*len(point_arr)))
    nr = int(np.sqrt(npoint)/np.sqrt(3))
    dr = (d_point/2)/nr
    r_arr = np.linspace(0., (d_point/2.)-dr, nr)
    r_arr = r_arr +dr/2
    da = (np.pi*(d_point**2)/4.)/npoint
    f_arr_i = 0
    for p in point_arr:
        i = 0
        for r in r_arr:
            dphi = da/(dr*(r))
            nphi = round(2*np.pi/dphi)
            theta = np.linspace(0,2*np.pi-dphi, nphi)
            for t in theta:
                t= t+dphi/2
                f_array[:,f_arr_i]= [p[0]+np.cos(t)*r,p[1]+np.sin(t)*r,p[2]]
                f_arr_i = f_arr_i + 1
            i = i + 1
    return f_array.T

@conditional_numba(skip_numba=True)
def blur_image(npoint,p_fibre,d_fibre, spec_los_vec, plot_im = False):
    z_axis = np.array([0.0, 0.0, 1.0])
    im_blur = np.zeros((np.shape(spec_los_vec)[0],npoint*len(p_fibre),3))
    im_arr = point_samp(npoint, p_fibre, d_fibre)
    for ii_image in range(np.shape(spec_los_vec)[0]):
        # Dotted into the los_vec to find dist to iteration's image plane not abs distance
        # Calculate rotation matrices for PSF
        los_vec_inplane = np.array([spec_los_vec[ii_image,0],spec_los_vec[ii_image,1],0.])/np.linalg.norm(np.array([spec_los_vec[ii_image,0],spec_los_vec[ii_image,1],0.]))
        RotAxis = np.cross(z_axis, los_vec_inplane)
        RotAngle = np.arccos(np.dot(z_axis, spec_los_vec[ii_image, :]))
        RotMat = ER_Matrix(RotAxis, RotAngle)
        azim_angle = (np.pi / 2.0) + np.arctan2(spec_los_vec[ii_image, 1], spec_los_vec[ii_image, 0])
        RotMat_azim = ER_Matrix(z_axis, azim_angle)
        if plot_im:
            import matplotlib.pyplot as plt
            im_arr = im_arr @ RotMat_azim @ RotMat
            fig2 = plt.figure()
            ax1 = fig2.add_subplot(111, projection='3d')
            ax1.scatter3D(im_arr[:,0],im_arr[:,1],im_arr[:,2])
        im_blur[ii_image,:,:] = im_arr @ RotMat_azim @ RotMat
    return im_blur

@conditional_numba(skip_numba=True)
def PSF(photon_origin,calc_photon_origin_type,dis,cross_pos,indices,dens,tb_einstein,g3_dvol,ir,factor, ii,
              matrix,v_xyz,g3_rotate_phi_grid, g3_dphi_sym,g3_Rmin, g3_dR, g3_nR, g3_Zmin, g3_dZ, g3_nZ,
              g3_phimin, g3_dphi, g3_nphi,ikind,xyzc_arr,los_image_arr,n_rand,image_pos,image_vec,image_blur,f_lens):
    

    # Step 1: Identify relevant rays
    ray_indices = np.nonzero(dis)[0][cross_pos]
    
    # Step 2: Compute photon flux
    photons = dens[cross_pos, 2] * tb_einstein[1, 2] / g3_dvol[ir[cross_pos]] * factor  # Shape: (N_rays,)
    
    # Step 3: Compute decay times
    tau = -1.0 / matrix[cross_pos, 2, 2]  # Shape: (N_rays,)
    tau = np.where(tau > 0, tau, 1e-10)[indices[0,:]]
    dist_image = np.sum((xyzc_arr[ray_indices, :, ii][indices[0,:],:] - image_pos[los_image_arr,:][indices[1,:],:])*image_vec[los_image_arr,:][indices[1,:],:],axis=1) #Change los_vec to lens_vec when nlos is not broken

    mag = np.multiply(dist_image,(1.0 / f_lens - 1.0 / dist_image))
    n_crossings = len(dist_image)
    random_values = np.random.uniform(0, 1, (n_rand,n_crossings))
    t = -np.log(random_values) * tau[np.newaxis,:]  # Shape: (n_rand,n_crossings)
    random_pick = np.random.randint(0, np.shape(image_blur)[1], (n_rand,n_crossings))
    im_blur_per_crossing = image_blur[los_image_arr,:,:][indices[1,:],:,:]
    pos = im_blur_per_crossing[np.arange(n_crossings),random_pick,:]*mag[np.newaxis,:,np.newaxis]
    # Step 4: Backtrack to excitation positions
    excitation_position = xyzc_arr[ray_indices, :, ii][indices[0,:],:]  + pos - t[:,:, np.newaxis] * v_xyz[ray_indices, :][indices[0,:],:]   # Shape: (N_crossing, 3)
    # Step 5: Get grid indices for excitation positions
    
    excitation_position_flat = excitation_position.reshape(-1,3)
    
    ir_ex, iz_ex, iphi_ex = get_grid3d_indices(
        excitation_position_flat[...,np.newaxis], g3_rotate_phi_grid, g3_dphi_sym,
        g3_Rmin, g3_dR, g3_nR, g3_Zmin, g3_dZ, g3_nZ,
        g3_phimin, g3_dphi, g3_nphi
    )  # Each has shape: (N_rays, N_points)

    ir_flat = ir_ex.reshape(n_rand,n_crossings)
    iz_flat = iz_ex.reshape(n_rand,n_crossings)
    iphi_flat = iphi_ex.reshape(n_rand,n_crossings)
    
    # Tile spec_nlos indices to match the flattened emission points
    ilos_flat = indices[1,:]
    ikind_flat = ikind[indices[0,:]]
    
    # Step 7: Compute photon values to add
    # Expand photons to match the number of spec_nlos and N_points
    values = photons[indices[0,:]] / ((len(los_image_arr)/np.shape(image_pos)[0])*n_rand)
    # Step 8: Accumulate photon counts using np.add.at
    if calc_photon_origin_type:
        np.add.at(
            photon_origin,
            (ikind_flat, ilos_flat, ir_flat, iz_flat, iphi_flat),
            values
        )
    else:
        np.add.at(
            photon_origin,
            (0, ilos_flat, ir_flat, iz_flat, iphi_flat),
            values
        )
    return photon_origin