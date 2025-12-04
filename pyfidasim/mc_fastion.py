import sys
sys.path.insert(1, '../')
## --------------------------------
## get random fast-ion vectors
## --------------------------------
from ._fields import xyz_to_Bxyz
import numpy as np
import scipy.constants as consts
import numba

def conditional_numba(skip_numba=False):
    def decorator(func):
        return numba.jit(func, cache=True, nopython=True,  nogil=True,debug = False)
    return decorator

def mc_fastion(xyz_array, fields, fbm):
    """
    Wrapper for the vectorized fast-ion velocity calculation.
    """
    vi_array = mc_fastion_core(
        xyz_array, fbm['fbm'], fbm['afbm'], fbm['btipsign'],
        fbm['emin'], fbm['eran'], fbm['nenergy'], fbm['energy'], fbm['dE'],
        fbm['pmin'], fbm['pran'], fbm['npitch'], fbm['pitch'], fbm['dP'],
        fields['Br'], fields['Bz'], fields['Bphi'], fields['rotate_phi_grid'], fields['dphi_sym'],
        fields['R'], fields['Rmin'], fields['Rmax'], fields['dR'], fields['nr'],
        fields['Z'], fields['Zmin'], fields['Zmax'], fields['dZ'], fields['nz'],
        fields['phi'], fields['phimin'], fields['phimax'], fields['dphi'], fields['nphi']
    )
    return vi_array

@conditional_numba(skip_numba=False)
def mc_fastion_core(xyz_array, fbm, afbm, btipsign, \
                               emin, eran, nenergy, energy, dE, \
                               pmin, pran, npitch, pitch, dP, \
                               f_Br, f_Bz, f_Bphi, f_rotate_phi_grid, f_dphi_sym, \
                               f_R, f_Rmin, f_Rmax, f_dR, f_nr, \
                               f_Z, f_Zmin, f_Zmax, f_dZ, f_nz, \
                               f_phi, f_phimin, f_phimax, f_dphi, f_nphi):
    """
    Vectorized core function to calculate fast-ion velocities for multiple positions.
    """
    n_particles = xyz_array.shape[0]
    vi_array = np.zeros((n_particles, 3))

    Bxyz_array = xyz_to_Bxyz(xyz_array, f_Br, f_Bz, f_Bphi, f_rotate_phi_grid, f_dphi_sym, \
                               f_R, f_Rmin, f_Rmax, f_dR, f_nr, \
                               f_Z, f_Zmin, f_Zmax, f_dZ, f_nz, \
                               f_phi, f_phimin, f_phimax, f_dphi, f_nphi)

    # Numba compatible normalization
    b_norm_mag = np.empty(n_particles)
    for i in range(n_particles):
        b_norm_mag[i] = np.sqrt(np.sum(Bxyz_array[i]**2))
    
    # Avoid division by zero
    b_norm_mag[b_norm_mag == 0] = 1.0
    b_norm = Bxyz_array / b_norm_mag.reshape(-1, 1)

    # Orthogonal basis vectors
    a_norm_unnormalized = np.empty((n_particles, 3))
    a_norm_unnormalized[:, 0] = b_norm[:, 1]
    a_norm_unnormalized[:, 1] = -b_norm[:, 0]
    a_norm_unnormalized[:, 2] = 0.0
    
    a_norm_mag = np.empty(n_particles)
    for i in range(n_particles):
        a_norm_mag[i] = np.sqrt(np.sum(a_norm_unnormalized[i]**2))

    # Avoid division by zero
    a_norm_mag[a_norm_mag == 0] = 1.0
    a_norm = a_norm_unnormalized / a_norm_mag.reshape(-1, 1)
    
    c_norm = np.cross(a_norm, b_norm)

    R = np.sqrt(xyz_array[:, 0]**2 + xyz_array[:, 1]**2)
    Z = xyz_array[:, 2]

    ir = ((R - f_Rmin + 0.5 * f_dR) / f_dR).astype(np.int64)
    iz = ((Z - f_Zmin + 0.5 * f_dZ) / f_dZ).astype(np.int64)

    energy2d = np.ravel(np.outer(np.ones(npitch), energy))
    pitch2d = np.ravel(np.outer(pitch, np.ones(nenergy)))

    for p_idx in range(n_particles):
        fbm_slice = fbm[:, :, ir[p_idx], iz[p_idx]]
        max_fbm = np.max(fbm_slice)
        if max_fbm == 0:
            continue
            
        fbm2 = np.ravel(fbm_slice / max_fbm)
        
        index = fbm2 > 0
        nfields = np.sum(index)
        
        if nfields == 0:
            continue
            
        fbm2_filtered = fbm2[index]
        energy2d_filtered = energy2d[index]
        pitch2d_filtered = pitch2d[index]

        ntry = 100000
        for i in range(ntry):
            randu = np.random.rand(2)
            ii_rand = int(nfields * randu[0])
            
            fbm_val = fbm2_filtered[ii_rand]
            
            if fbm_val > randu[1]:
                eb = energy2d_filtered[ii_rand]
                ptch = pitch2d_filtered[ii_rand]
                
                vabs = np.sqrt(2 * (eb * 1.e3 * consts.e) / (consts.atomic_mass * afbm)) * 1.e2
                phi_rand = 2. * np.pi * np.random.rand()
                sinus = np.sqrt(1. - ptch**2)
                
                vi = vabs * (sinus * np.cos(phi_rand) * a_norm[p_idx, :] +
                             ptch * b_norm[p_idx, :] * btipsign +
                             sinus * np.sin(phi_rand) * c_norm[p_idx, :])
                             
                vi_array[p_idx, :] = vi
                break
        else:
            pass # vi_array remains zeros if no solution found

    return vi_array