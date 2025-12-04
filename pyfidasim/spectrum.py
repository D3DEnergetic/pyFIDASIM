import numpy as np
import scipy.constants as const
from .toolbox import rotate_vectors_ER
import numba

def conditional_numba(skip_numba=False):
    def decorator(func):
        if not skip_numba:
            return numba.jit(func, 
                             cache=True, 
                             nopython=True, 
                             nogil=True)
        else:
            return func
    return decorator

_STARK_WAVEL = np.array([
    -2.20200e-06, -1.65200e-06, -1.37700e-06, -1.10200e-06,
    -8.26400e-07, -5.51000e-07, -2.75600e-07, 0.0,
     2.75700e-07,  5.51500e-07,  8.27400e-07,  1.10300e-06,
     1.38000e-06,  1.65600e-06,  2.20900e-06
], dtype=np.float64) / 10.0

_STARK_INTENS = np.array([
    1.0, 18.0, 16.0, 1681.0, 2304.0,
    729.0, 1936.0, 5490.0, 1936.0, 729.0,
    2304.0, 1681.0, 16.0, 18.0, 1.0
], dtype=np.float64)

_STARK_PI = np.array([1,0,0,1,1,1,0,0,0,1,1,1,0,0,1], dtype=np.int8)
_STARK_SIGMA = (1 - _STARK_PI).astype(np.int8)
_STARK_SIGN = _STARK_SIGMA - _STARK_PI  # +1 for sigma, -1 for pi


from numba import njit, prange
import math
@njit(parallel=True,fastmath=True)
def _build_bins_and_weights_numba(
    vr, los, B, ilos, cells,
    exyz_is_const, ex0, ey0, ez0, exyz_per,
    ratio, dl_per_intersection, photons,
    lambda0, inv_c, spec_lambda_min, inv_dlam, wav_scale,
    spec_nlam, per_line
):
    N = ilos.shape[0]
    nstark = 15
    out_bins = np.empty(N*nstark, dtype=np.int64)
    out_w    = np.empty(N*nstark, dtype=np.float64)

    for i in prange(N):
        vx = vr[i, 0]; vy = vr[i, 1]; vz = vr[i, 2]
        lx = los[i, 0]; ly = los[i, 1]; lz = los[i, 2]
        Bx = B[i, 0];   By = B[i, 1];   Bz = B[i, 2]

        # Doppler
        doppler = (vx*lx + vy*ly + vz*lz) * inv_c
        lam_d   = lambda0 * (1.0 + doppler)
        lam_scaled = (lam_d - spec_lambda_min) * inv_dlam

        # Exyz base
        if exyz_is_const:
            Ex = ex0; Ey = ey0; Ez = ez0
        else:
            Ex = exyz_per[i, 0]; Ey = exyz_per[i, 1]; Ez = exyz_per[i, 2]

        # E = Exyz + v x B
        Ex = Ex +  vy*Bz - vz*By
        Ey = Ey +  vz*Bx - vx*Bz
        Ez = Ez +  vx*By - vy*Bx

        # |E| and cos(los,E)
        E  = math.sqrt(Ex*Ex + Ey*Ey + Ez*Ez)
        num = lx*Ex + ly*Ey + lz*Ez
        cos_los_E = 0.0
        if E != 0.0:
            cos_los_E = num / E
        cos2 = cos_los_E * cos_los_E

        # per-ray params
        ratio_i = ratio[i]
        dlphot  = dl_per_intersection[i] * photons[i]
        ilos_i  = ilos[i]

        # first pass: raw intensities and sum
        tmp = np.empty(nstark, dtype=np.float64)
        s = 0.0
        for k in range(nstark):
            base = _STARK_INTENS[k] * (1.0 + _STARK_SIGN[k] * cos2)
            fac  = 1.0 + (ratio_i - 1.0) * _STARK_SIGMA[k]
            val  = base * fac
            tmp[k] = val
            s += val

        inv_s = 0.0
        if s != 0.0:
            inv_s = 1.0 / s

        # second pass: finalize weights and bins
        base_offset_los = ilos_i * spec_nlam
        base_offset_lsl = ilos_i * (nstark * spec_nlam)

        for k in range(nstark):
            # weight
            w = tmp[k] * inv_s * dlphot

            # wavelength bin (clipped)
            idx = int(lam_scaled + E * wav_scale[k])
            if idx < 0:
                idx = 0
            elif idx >= spec_nlam:
                idx = spec_nlam - 1

            # linearized bin
            if per_line:
                bin_id = base_offset_lsl + k * spec_nlam + idx
            else:
                bin_id = base_offset_los + idx

            out_bins[i*nstark + k] = bin_id
            out_w[i*nstark + k]    = w

    return out_bins, out_w

def spectrum(mass, lambda0, v, indices, Exyz, Bxyz,
                   spec_nlos, spec_los_vec, spec_lambda_min, spec_nlam, spec_dlam,
                   spec_dl_per_grid_intersection, spec_sigma_to_pi_ratio,
                   spec_output_individual_stark_lines, photons):
    rays  = indices[0, :]
    ilos  = indices[1, :].astype(np.int64)
    cells = indices[2, :].astype(np.int64)

    # Gather
    los = np.ascontiguousarray(spec_los_vec[ilos])  # (N,3)
    vr  = np.ascontiguousarray(v[rays])             # (N,3)

    B = np.ascontiguousarray(Bxyz[rays])

    # Exyz: allow (3,) or (N,3)
    if Exyz.ndim == 1:
        exyz_is_const = True
        ex0, ey0, ez0 = float(Exyz[0]), float(Exyz[1]), float(Exyz[2])
        exyz_per = np.zeros((1, 3), dtype=np.float64)
    else:
        exyz_is_const = False
        ex0 = ey0 = ez0 = 0.0
        exyz_per = np.ascontiguousarray(Exyz)  # expect shape (N,3)

    ratio = np.ascontiguousarray(spec_sigma_to_pi_ratio[ilos])
    dl_per_intersection = np.ascontiguousarray(spec_dl_per_grid_intersection[ilos, cells])
    photons = np.ascontiguousarray(photons)  # (N,)

    inv_c   = 1.0 / const.c
    inv_dlam = 1.0 / spec_dlam
    wav_scale = _STARK_WAVEL * inv_dlam

    # Build bins + weights
    bins_flat, w_flat = _build_bins_and_weights_numba(
        vr, los, B, ilos, cells,
        exyz_is_const, ex0, ey0, ez0, exyz_per,
        ratio, dl_per_intersection, photons,
        lambda0, inv_c, spec_lambda_min, inv_dlam, wav_scale,
        int(spec_nlam), bool(spec_output_individual_stark_lines)
    )

    # Reduce with bincount
    if spec_output_individual_stark_lines:
        nstark = 15
        flat = np.bincount(
            bins_flat,
            weights=w_flat,
            minlength=spec_nlos * nstark * spec_nlam
        )
        return flat.reshape(spec_nlos, nstark, spec_nlam)
    else:
        flat = np.bincount(
            bins_flat,
            weights=w_flat,
            minlength=spec_nlos * spec_nlam
        )
        return flat.reshape(spec_nlos, 1, spec_nlam)

@conditional_numba(skip_numba=False)
def spectrum_other(cdfvars,lambda0,trans, l_to_dwp, \
                    v,indices,Bxyz, spec_sigma_to_pi_ratio,\
                    spec_dl_per_grid_intersection,spec_lambda_min, \
                    spec_nlos,spec_los_vec,spec_nlam,spec_dlam,spec_output_individual_stark_lines):
    ## Note that the following code assumes that the electric field is produced solely from
    ##vxB (imposed from the data from Dr. Raplh Dux). So no static electric field is considered
    
    ##Additionally the line intensities of sigma and pi polarized light are considered together 
    ##and exist in the same line emmissions so cannot be trivially considered seperately. To add the 
    ##sigma_to_pi ratio for the physical optics further analysis must be completed. 
    if spec_output_individual_stark_lines:
        intens_out = np.zeros((spec_nlos, trans, spec_nlam))
    else:
        intens_out = np.zeros((spec_nlos, 1, spec_nlam))
    # intens_out = np.zeros((spec_nlos, spec_nlam))
    for ii in range(len(indices[:, 0])):
        
        ilos = indices[ii, 0]
        lambda_Doppler = lambda0 * (1.0 + np.dot(v, spec_los_vec[ilos, :]) / const.c)
        #vp = norm(cross_(v_new,Bxyz_new))/norm(Bxyz_new) #perpendicular magnitude
        
        vp = np.linalg.norm(np.cross(v,Bxyz))/np.linalg.norm(Bxyz) 
        para_interp = np.zeros((5,trans))
        for j in range(5):
            for k in range(trans):
                para_interp[j,k] = np.interp(vp,cdfvars[0,k,:],cdfvars[j+1,k,:])
                # index = np.argmin(np.abs(vp-cdfvars[0,k,:]))
                # para_interp[j,k] = cdfvars[j+1,k,index]
                # print(cdfvars[j+1,k,index])
                
        # print(v)
        #beta = dot(v_new,los_vec)/const.c
        #lambda_Doppler = lambda0*(1.0 + dot(v_new,los_vec)/const.c)#np.sqrt((1+beta)/(1-beta))#(1.0 + dot(v,los_vec)/const.c)

        # calculate the wavelength
        wavel =  lambda_Doppler + np.linalg.norm(Bxyz)*para_interp[0,:]
        #Determine p,t (and ct2,cp2) from los_Rot=[cos(p)*sin(t),sin(p)*sin(t),cos(t)]
        #if norm(cross_(los_vec,Bxyz_new)) == 0.0:
        if np.linalg.norm(np.cross(spec_los_vec[ilos,:],Bxyz)) == 0:
            #checks if los is same direction as B-field
            #system implies t=0 from 3rd element's equation
            #elements 1 and 2 are equal to zero implying sin(p)=cos(p)
            #p,t found independently --> unique solution to system
            p = 0
            t = 0
            raise ValueError("los in direction of B-field: Case undefined (phi undefined)")
            
        else:
            #p obtained by dividing 2nd element's equation by first, then solving
            #t obtained by solving 3rd element's equation
            #p,t found independently --> unique solution to system
            #Bxyz_new,v_new,los_vec,Exyz = rotate_vectors_ER(np.copy(Bxyz),np.copy(v),np.copy(spec_los_vec[ilos,:]))
            
            
            #Rotate vectors to match description in netCDF file
            los_vec_new = rotate_vectors_ER(Bxyz,v,spec_los_vec[ilos,:])

            
            p = np.arctan(los_vec_new[1]/los_vec_new[0])
            t = np.arccos(los_vec_new[2])
    
        cp2 = np.cos(p)**2
        ct2 = np.cos(t)**2
        
        
        # # determine line strengths  
        intens = (1/2 * (para_interp[2]+para_interp[3]) * (1+ct2)) + \
            (para_interp[1]+para_interp[4] - 2*para_interp[4]*cp2) * (1-ct2)
            
        # L = 1/2*(p2+m2)*(1+ct2)+[z2+pm-2*pm*cp2]*(1-ct2)
        # [0]=wvl_shift, [1]=z2, [2]=p2, [3]=m2, [4]=pm
        ## -------------------------------------    
        ## normalize  
        ## -------------------------------------            
        intens /= np.sum(intens)
        # -------------------------------------------------------------
        # get the intersection length of this LOS with the grid-cell.
        # -------------------------------------------------------------            
        dl = spec_dl_per_grid_intersection[ilos, indices[ii, 1]] 
        ## multiply intens with the intersection length
        intens *= dl  
        for j in range(trans):
            index = int((wavel[j] - spec_lambda_min) / spec_dlam)
            if index < 0:
                index = 0
            if index >= spec_nlam:
                index = spec_nlam - 1
            if spec_output_individual_stark_lines:
                intens_out[ilos, j, index] += intens[j]
            else:
                intens_out[ilos, 0, index] += intens[j]

    return(intens_out)