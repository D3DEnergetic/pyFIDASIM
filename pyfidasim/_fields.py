import numpy as np
from scipy.interpolate import griddata
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

@conditional_numba(skip_numba=False)
def xyz_to_s(xyz, f_rotate_phi_grid, f_dphi_sym, f_s, \
                   f_R,f_Rmin, f_Rmax, f_dR, f_nr, \
                   f_Z,f_Zmin, f_Zmax, f_dZ, f_nz, \
                   f_phi,f_phimin, f_phimax, f_dphi, f_nphi):
    
    
    '''
    Get the magnetic field vecotr in x,y,z coordinates from the fields by
    using the 2d interpolation in "interp_fields"
    '''
    #R = np.sqrt(xyz[0]**2 + xyz[1]**2)
    #Z = xyz[2]
    #phi = np.arctan2(xyz[1], xyz[0]) + f_rotate_phi_grid


    R_arr = np.array([np.sqrt(xyz[0]**2 + xyz[1]**2)])
    Z_arr = np.array([xyz[2]])
    phi_arr = np.array([np.arctan2(xyz[1], xyz[0]) + f_rotate_phi_grid])
    phi_arr = phi_arr % f_dphi_sym
    

    s_arr = interp_fields(R_arr, Z_arr, phi_arr, f_s, \
                     f_R,f_Rmin,f_dR,f_nr, \
                     f_Z,f_Zmin,f_dZ,f_nz, \
                     f_phi,f_phimin,f_dphi,f_nphi)
    s=s_arr[0]
    
    if s != s:
        s=2
    return(s)
        


@conditional_numba(skip_numba=False)
def xyz_to_denf(xyzc_arr,fbm_denf,f_rotate_phi_grid,f_dphi_sym, \
                            f_R,f_Rmin,f_dR,f_nr, \
                            f_Z,f_Zmin,f_dZ,f_nz, \
                            f_phi,f_phimin,f_dphi,f_nphi):
    '''
    Get the fast-ion density from the 2D denf values stored.
    This version is vectorized to handle multiple rays at once.
    '''
    # Store original shape to reshape the output later
    original_shape = xyzc_arr.shape
    # If input is 3D (n_rays, 3, n_steps), reshape to 2D for processing
    if xyzc_arr.ndim == 3:
        n_rays, _, n_steps = original_shape
        # Transpose, CREATE A CONTIGUOUS COPY, and then reshape
        xyz_flat = np.transpose(xyzc_arr, (1, 0, 2)).copy().reshape(3, -1)
    else: # Assuming it's already (3, n_points)
        xyz_flat = xyzc_arr

    # get the R,Z,phi positions from the flattened xyz array
    R = np.sqrt(xyz_flat[0, :]**2 + xyz_flat[1, :]**2)
    Z = xyz_flat[2, :]
    phi = np.arctan2(xyz_flat[1, :], xyz_flat[0, :]) + f_rotate_phi_grid
    
    # Use np.where for Numba compatibility instead of boolean indexing
    phi = np.where(phi < 0, phi + 2. * np.pi, phi)

    # if the simulation is only for one period and we make use of the symmetry,
    # then, dphi_sym is not 2.pi and we get another phi
    phi = phi % f_dphi_sym
    
    denf_arr_flat = interp_fields(R, Z, phi, fbm_denf, \
                                  f_R, f_Rmin, f_dR, f_nr, \
                                  f_Z, f_Zmin, f_dZ, f_nz, \
                                  f_phi, f_phimin, f_dphi, f_nphi)
        
    denf_arr_flat = np.nan_to_num(denf_arr_flat)

    # Reshape back to the original (n_rays, n_steps) if necessary
    if xyzc_arr.ndim == 3:
        denf_arr = denf_arr_flat.reshape(n_rays, n_steps)
    else:
        denf_arr = denf_arr_flat

    return denf_arr
    

@conditional_numba(skip_numba=False)
def xyz_to_Bxyz(xyz,f_Br,f_Bz,f_Bphi,f_rotate_phi_grid,f_dphi_sym, \
                             f_R,f_Rmin,f_Rmax,f_dR,f_nr, \
                             f_Z,f_Zmin,f_Zmax,f_dZ,f_nz, \
                             f_phi,f_phimin,f_phimax,f_dphi,f_nphi):
    '''
    Get the magnetic field vector in x,y,z coordinates from the fields by
    using the 2d interpolation in "interp_fields"
    '''
    R = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
    Z = xyz[:,2]
    phi = np.arctan2(xyz[:,1], xyz[:,0]) + f_rotate_phi_grid
    phi[phi<0] += 2. * np.pi
    # re-define a new phi (phi2) in case we make use of the symmetry.
    phi2 = phi % f_dphi_sym

    Bxyz = np.zeros((np.shape(xyz)[0],3))
    Br2 = interp_fields(R, Z, phi2, f_Br, \
                     f_R,f_Rmin,f_dR,f_nr, \
                     f_Z,f_Zmin,f_dZ,f_nz, \
                     f_phi,f_phimin,f_dphi,f_nphi)
    # check for NANs (this seems to occur for edge positions)
    nan_index = ~np.isnan(Br2)
    Bz2 = interp_fields(R, Z, phi2, f_Bz, \
                     f_R,f_Rmin,f_dR,f_nr, \
                     f_Z,f_Zmin,f_dZ,f_nz, \
                     f_phi,f_phimin,f_dphi,f_nphi)
    
    Bphi2 = interp_fields(R, Z, phi2, f_Bphi, \
                     f_R,f_Rmin,f_dR,f_nr, \
                     f_Z,f_Zmin,f_dZ,f_nz, \
                     f_phi,f_phimin,f_dphi,f_nphi)    

    # to finally get Bxyz, remove the grid rotation (the LOS is not in the
    # rotated one!)
    phi -= f_rotate_phi_grid
    Bxyz[nan_index,0] = -np.cos(np.pi * 0.5 - phi[nan_index]) * Bphi2[nan_index] + np.cos(phi[nan_index]) * Br2[nan_index]
    Bxyz[nan_index,1] = np.sin(np.pi * 0.5 - phi[nan_index]) * Bphi2[nan_index] + np.sin(phi[nan_index]) * Br2[nan_index]
    Bxyz[nan_index,2] = Bz2[nan_index]
    return(Bxyz)

@conditional_numba(skip_numba=False)
def interp_fields(R, Z, phi, data, \
                             f_R,f_Rmin,f_dR,f_nr, \
                             f_Z,f_Zmin,f_dZ,f_nz, \
                             f_phi,f_phimin,f_dphi,f_nphi):
    '''
    Numba-friendly, high-performance 2D or 3D linear interpolation on a regular grid.
    This version uses a hybrid vectorized/loop approach to work around Numba's
    advanced indexing limitations.
    '''
    npos = R.shape[0]
    
    # --- Vectorized Calculation of Indices and Fractions ---
    # This part remains fast and vectorized.
    r_pos = (R - f_Rmin) / f_dR
    z_pos = (Z - f_Zmin) / f_dZ
    
    rind = r_pos.astype(np.int32)
    zind = z_pos.astype(np.int32)
    
    dr_frac = r_pos - rind
    dz_frac = z_pos - zind
    
    rind = np.clip(rind, 0, f_nr - 2)
    zind = np.clip(zind, 0, f_nz - 2)
    dr_frac = np.clip(dr_frac, 0.0, 1.0)
    dz_frac = np.clip(dz_frac, 0.0, 1.0)
    
    s = np.zeros(npos, dtype=np.float64)

    for i in range(npos):
        ri, zi = rind[i], zind[i]
        dr_f, dz_f = dr_frac[i], dz_frac[i]

        if f_nphi == 1:
            # Toroidally symmetric case
            c00 = data[ri, zi, 0]
            c10 = data[ri + 1, zi, 0]
            c01 = data[ri, zi + 1, 0]
            c11 = data[ri + 1, zi + 1, 0]
            
            # Interpolate along R
            c0 = c00 * (1.0 - dr_f) + c10 * dr_f
            c1 = c01 * (1.0 - dr_f) + c11 * dr_f
            
            # Interpolate along Z
            s[i] = c0 * (1.0 - dz_f) + c1 * dz_f
        else:
            # 3D case
            # Phi calculations are scalar within the loop
            phi_pos = (phi[i] - f_phimin) / f_dphi
            p1_idx = int(phi_pos)
            dphi_f = phi_pos - p1_idx
            
            p1_idx = p1_idx % f_nphi
            p2_idx = (p1_idx + 1) % f_nphi
            dphi_f = min(max(dphi_f, 0.0), 1.0)
            
            # Interpolate at phi plane 1
            c00_p1 = data[ri, zi, p1_idx]
            c10_p1 = data[ri + 1, zi, p1_idx]
            c01_p1 = data[ri, zi + 1, p1_idx]
            c11_p1 = data[ri + 1, zi + 1, p1_idx]
            c0_p1 = c00_p1 * (1.0 - dr_f) + c10_p1 * dr_f
            c1_p1 = c01_p1 * (1.0 - dr_f) + c11_p1 * dr_f
            s1 = c0_p1 * (1.0 - dz_f) + c1_p1 * dz_f

            # Interpolate at phi plane 2
            c00_p2 = data[ri, zi, p2_idx]
            c10_p2 = data[ri + 1, zi, p2_idx]
            c01_p2 = data[ri, zi + 1, p2_idx]
            c11_p2 = data[ri + 1, zi + 1, p2_idx]
            c0_p2 = c00_p2 * (1.0 - dr_f) + c10_p2 * dr_f
            c1_p2 = c01_p2 * (1.0 - dr_f) + c11_p2 * dr_f
            s2 = c0_p2 * (1.0 - dz_f) + c1_p2 * dz_f
            
            # Interpolate in phi
            s[i] = s1 * (1.0 - dphi_f) + s2 * dphi_f
            
    return s

@conditional_numba(skip_numba=False)
def get_s_along_path(p0, v, dl,
                                    f_rotate_phi_grid, f_dphi_sym, f_s, prof_s,
                                    f_Rmin, f_Rmax, f_R, f_dR, f_nr,
                                    f_Zmin, f_Zmax, f_Z, f_dZ, f_nz,
                                    f_phimin, f_phimax, f_phi, f_dphi, f_nphi,
                                    jump=0., seed=-1):

    if seed >= 0:
        np.random.seed(seed)

    # number of rays
    n_rays = p0.shape[0]
    init_R = np.max(np.sqrt(p0[:,0]**2 + p0[:,1]**2 + p0[:,2]**2))
    if init_R > f_Rmax or np.max(p0[:,2]) > f_Zmax:
        max_dist = init_R + f_Rmax * 2.5 - jump
    else:
        max_dist = f_Rmax * 2.5 - jump
    
    two_pi = 2. * np.pi
    s_threshold = np.max(prof_s)
    max_len = int((max_dist) / dl)
    
    s_results = np.zeros((n_rays, max_len),dtype=np.float32)
    xyzc_arr_results = np.zeros((n_rays, 3, max_len),dtype=np.float32)
    dl_arr_results = np.zeros((n_rays, max_len),dtype=np.float32)
    first_step_results = np.zeros(n_rays,dtype=np.float32)
    dist_offset_arr = np.random.rand(n_rays).astype(np.float32) * dl if jump > 0 else np.full(n_rays, 0.5 * dl,dtype=np.float32)

    for track in range(n_rays):
        p0_i = p0[track, :]
        v_i = v[track, :]
        dist_offset = dist_offset_arr[track]
        dist_arr = np.arange(jump + dist_offset, jump + max_dist + dist_offset, dl)

        xyzc_arr = p0_i[:, None] + v_i[:, None] * dist_arr
        R = np.sqrt(np.sum(xyzc_arr[:2, :]**2, axis=0))
        Z = xyzc_arr[2, :]
        phi = np.arctan2(xyzc_arr[1, :], xyzc_arr[0, :]) + f_rotate_phi_grid
        phi = (phi + two_pi) % two_pi
        phi = phi % f_dphi_sym

        index = (R > f_Rmin) & (R < f_Rmax*1.2) & (Z > f_Zmin) & (Z < f_Zmax) & (phi >= f_phimin) & (phi < f_phimax)
        if not np.any(index):
            continue

        R, Z, phi, xyzc_arr, dist_arr = R[index], Z[index], phi[index], xyzc_arr[:, index], dist_arr[index]
        s = interp_fields(R, Z, phi, f_s, f_R, f_Rmin, f_dR, f_nr, f_Z, f_Zmin, f_dZ, f_nz, f_phi, f_phimin, f_dphi, f_nphi)
        s[np.isnan(s)] = 2
        index = s < s_threshold
        if not np.any(index):
            continue

        plasma_mask = s < s_threshold

        # FIRST entry index
        entry_idx = np.where(plasma_mask)[0][0]

        # Build a mask that keeps:
        #  - every step before entry_idx (even if outside)
        #  - only those steps inside plasma after entry
        keep_mask = np.zeros_like(plasma_mask)
        keep_mask[:entry_idx] = True
        keep_mask[entry_idx:] = plasma_mask[entry_idx:]
        
        # apply final mask
        s          = s[keep_mask]
        xyzc_arr   = xyzc_arr[:, keep_mask]
        dist_arr   = dist_arr[keep_mask]
        
        first_step = dist_arr[0] - dl
        dl_arr = np.full(len(s), dl)

        result_len = len(s)
        s_results[track, :result_len] = s
        xyzc_arr_results[track, :, :result_len] = xyzc_arr
        dl_arr_results[track, :result_len] = dl_arr
        first_step_results[track] = first_step

    return s_results, xyzc_arr_results, dl_arr_results, first_step_results
              
def generate_fields_cylindrical(rminor,rmajor,dbound,drz=4.,flux_nr=100):  
    fields={}     
    fields['flux_nr']=flux_nr
    fields['rminor']=rminor
    fields['rmajor']=rmajor

    s=np.linspace(0,((rminor+dbound)/rminor)**2,flux_nr+1)
    

    fields['smax']=max(s)
    print('smax:',fields['smax'])
    
    rgrid=np.sqrt(s)*rminor
    vol= 2.*np.pi**2 * rmajor * rgrid**2 #[cm^3]
    fields['flux_dvol']=np.diff(vol) #[cm^3]
    
    
    ds=np.diff(s)
    sc = s[0:-1] + ds * 0.5  # grid centers
    fields['flux_s'] = sc
    fields['flux_ra'] = np.sqrt(sc)    
    fields['flux_ds'] = ds[0]
    
    fields['s_surf']=sc
    fields['nphi']=1
    fields['ntheta']=100
    fields['Rsurf']=np.zeros((flux_nr,fields['nphi'],fields['ntheta']))
    fields['Zsurf']=np.zeros((flux_nr,fields['nphi'],fields['ntheta']))    
    theta=np.linspace(0,2.*np.pi,fields['ntheta'])
    for i in range(flux_nr):
        fields['Rsurf'][i,0,:]=np.cos(theta)*rminor*fields['flux_ra'][i]+fields['rmajor']
        fields['Zsurf'][i,0,:]=np.sin(theta)*rminor*fields['flux_ra'][i]
        

    fields['n0dens']=np.ones((1,fields['ntheta']))
    fields['Rmax'] = rmajor+rminor+dbound
    fields['Rmin'] = rmajor-rminor-dbound
    fields['Rmean'] = (fields['Rmin'] + fields['Rmax']) / 2.
    fields['Zmax'] = rminor+dbound
    fields['Zmin'] = -rminor-dbound


    
    nr = int((fields['Rmax'] - fields['Rmin']) / drz)
    nz = int((fields['Zmax'] - fields['Zmin']) / drz)
    
    
    rgrid=np.linspace(fields['Rmin'],fields['Rmax'],nr) 
    zgrid=np.linspace(fields['Zmin'],fields['Zmax'],nr) 
    fields['nphi']=1
 
    s_grid = np.zeros([nr, nz, fields['nphi']])
    for ir in range(nr):
        for iz in range(nz):
            ra=np.sqrt((rgrid[ir]-fields['rmajor'])**2+zgrid[iz]**2)/fields['rminor']
            s_grid[ir,iz,0]=ra**2
    fields['s']=s_grid     
    fields['nsym']=1    
    fields['dphi_sym']=2.*np.pi  
    fields['nr'] = nr
    fields['nz'] = nz

    fields['R'] = rgrid
    fields['Z'] = zgrid
 
    fields['dR'] = rgrid[1]-rgrid[0]
    fields['dZ'] = zgrid[1]-zgrid[0]
    
    fields['phi'] = np.array([0.])
    fields['phimin'] = 0.
    fields['phimax'] = 2.*np.pi
    fields['dphi'] = 2.*np.pi    
    fields['rotate_phi_grid'] = np.float64(0.)


    # magnetic fields: Br, Bz, Bphi
    fields['calcb'] = False
    fields['Br'] =np.zeros((1,1,1))
    fields['Bz'] =np.zeros((1,1,1))
    fields['Bphi'] =np.zeros((1,1,1))    
    
    return(fields)



def calc_cylnd_Bvec(wout):
    '''
    Transforms magnetic field vector from flux coordinates to cylindrical
    coordinates {Br,Bp,Bz}.
    '''
    try:
        Bs = wout.invFourAmps['Bs']
        Bv = wout.invFourAmps['Bv']
        Bu = wout.invFourAmps['Bu']

        R = wout.invFourAmps['R']
        dR_ds = wout.invFourAmps['dR_ds']
        dR_dv = wout.invFourAmps['dR_dv']
        dR_du = wout.invFourAmps['dR_du']

        dZ_ds = wout.invFourAmps['dZ_ds']
        dZ_dv = wout.invFourAmps['dZ_dv']
        dZ_du = wout.invFourAmps['dZ_du']

    except BaseException:
        raise ValueError(
            'invFourAmps does not have all necessary variables to perform cylindrical transformation.')
    B_norm = R * 0.
    index = (dR_ds * dZ_du - dR_du * dZ_ds) != 0
    B_norm[index] = 1. / ((dR_ds * dZ_du - dR_du * dZ_ds)[index])
    wout.Br = (dZ_du * Bs - dZ_ds * Bu) * B_norm
    wout.Bz = (dR_ds * Bu - dR_du * Bs) * B_norm
    wout.Bphi = (((Bs * (dR_du * dZ_dv - dR_dv * dZ_du) + Bu *
                   (dR_dv * dZ_ds - dR_ds * dZ_dv)) * B_norm) + Bv) / R
                       

def generate_fields(woutfile, nsym, ntheta=80, drz=2., calc_brzphi=True, 
                   extended_vmec_factor=1., bfield_exp_factor = 1., 
                   phi_ran=None,scale_up_factor=1):

    """
    Routine to generate the fields dictionary as used in pyFIDASIM
    INPUTS:
        woutfile: 
            (str), path for the VMEC wout.nc file
            for further information see 
            http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/#output
        nsym:
            (int), symmetry of the equillibrium, like 5 for W7X
        ntheta:
            (int), number of poloidal resolution
            default: 80
        drz:
            (dbl), radial and vertical minimum resolution in cm
            default: 2.
        calc_brzphi:
            (bool), whether or not calculate magnetic field vectors
            default: True
        extended_vmec_factor:
            (dbl), rescaling factor for the flux labelings (using 1.5 s = 1 --> 1.5)
            default: 1.
        bfield_exp_factor:
            (dbl), rescaling factor of the magnetic field
            default: 1.
        phi_ran:
            [(dbl),(dbl)], toroidal angle range where quantities should be calculated, if set to
            None, then the whole 0,2pi interval is being used.
            default: None
    OUTPUTS:
        fields dictionary see below
    """
    ## ------------------------------------------------------------------------
    ## --- Start defining the fields output sturcture --------------------------
    ## ------------------------------------------------------------------------
    fields = {'ntheta': ntheta,'calc_brzphi':calc_brzphi,
             'extended_vmec_factor':extended_vmec_factor,
             'drz': drz,'nsym': nsym, 'bfield_exp_factor' : bfield_exp_factor}

    ## ------------------------------------------------------------------------
    ## --------- Initialize reading the WOUT VMEC file ------------------------
    ## ------------------------------------------------------------------------
    try:
        from pyfidasim.wout import readWout
    except:
         ## when starting fields directly from here, it 
         ## somehow doesn't fine the pyfidasim. routines...
        from wout import readWout 
    wout = readWout(path='', name=woutfile, diffAmps=calc_brzphi)

    ## -------------------------------------------------------------------------
    ## - Def. phi and theta-grids for which VMEC fourier moments are evaluated -
    ## -------------------------------------------------------------------------
    if phi_ran:
        ## if a phi-range is defined, we start still with the 
        ## range from phi=0 to 2pi which is needed for the calculation
        ## of the flux surface area and volume. Then later in the code,
        ## the phi-range is reduced.
        fields['dphi_sym'] = 2. * np.pi 
        fields['phimin'] = 0. 
        fields['phimax'] = 2.*np.pi
    else:
        ## If no phi-range is provided, we make use of the symmmetry of 
        ## a given stellarator device and therefore calcualte phi only for a 
        ## certain fraction of the device. By using "dphi_sym" pyfidasim can 
        ## still determine the magnetic field everywhere in the plasma.
        print('Consider the symmetry of the device:', nsym)
        fields['dphi_sym'] = 2. * np.pi/nsym
        fields['phimin'] = 0.
        fields['phimax'] = fields['dphi_sym']
        

    ## We use "drz" as input for the grid spacing. While we use dr=drz and 
    ## dz=drz, we need an estimate of Rmajor radius to get dphi = drz*Rmajor:
    wout.transForm_3D(np.linspace(0, 2 * np.pi, 50),
                      np.linspace(0, 2 * np.pi, 120),
                      ['R', 'Z'])
    dphi_estimate = drz / np.mean(wout.invFourAmps['R'] *100.)
    fields['nphi'] = int((fields['phimax'] - fields['phimin']) / dphi_estimate)
    # Phi should close by itself when defining it from 0-2pi. This is needed for the interpolation 
    ## which expects boundary values at phi_min and phi_max.
    fields['phi'] = np.linspace(fields['phimin'],fields['phimax'],fields['nphi'],endpoint=True)
    
    fields['dphi'] = fields['phi'][1]-fields['phi'][0]
    print('dphi [deg]:', np.round(np.rad2deg(fields['dphi']),3))
    # Theta has to close itself for the calculation of the area and volume
    fields['theta'] = np.linspace(0, 2*np.pi,fields['ntheta'],endpoint=True)
 
    ## ------------------------------------------------------------------------
    ## ---- Run the VMEC evaluation code to get surface contours and B-fields!
    ## ------------------------------------------------------------------------
    ## then run the code:
    keys = ['R', 'Z'] 
    wout.transForm_3D(fields['theta'],fields['phi'], keys)
    # Finally get the R and Z positions of the VMEC output
    R = wout.invFourAmps['R'] * 100.*scale_up_factor
    Z = wout.invFourAmps['Z'] * 100.*scale_up_factor
    ns = R.shape[0] ## number of s-domain points
    

    ## ------------------------------------------------------------------------    
    ## ----  Extend the Equilibrium accross the LCFS --------------------------
    ## ------------------------------------------------------------------------  
    if extended_vmec_factor >1:
        ## --------------------------------------------------    
        ## ---- Apply extended VMEC factor ------------------
        ## --------------------------------------------------  
        print('VMEC extension factor =', extended_vmec_factor)
        s_dom =wout.s_dom * extended_vmec_factor           
    else:
        ## --------------------------------------------------    
        ## ----otherwise add points in the SOL region  ------
        ## --------------------------------------------------  
        ## extend the s_domain to about s=1.5
        print('No extended VMEC, so extend boundary manually!')
        s_dom=wout.s_dom    
        ds=np.mean(np.diff(s_dom))
        ns_sol=int(0.5/ds)
        
        s_sol=np.linspace(s_dom[-1]+ds,s_dom[-1]+ns_sol*ds,num=ns_sol)
        s_extended=np.r_[s_dom,s_sol]
        ns_extended=ns+ns_sol
        ## Generate enlarged r_surf and z_surf array and fill with VMEC data   
        r_surf_sol=np.zeros((ns_extended,fields['nphi'],fields['ntheta']))
        z_surf_sol=np.zeros((ns_extended,fields['nphi'],fields['ntheta']))  
        r_surf_sol[0:ns,:,:]=R[:, :, :]
        z_surf_sol[0:ns,:,:]=Z[:, :, :]
      
        ## ----------------------------------------------------
        ## -- get avg. area of the last closed flux surface ---
        ## ----------------------------------------------------
        ## this area is needed to find areas of the extended flux surfaces 
        avg_area_lcfs = 0
        ## loop only until nphi-1 since the last and first points are identical when going from 0 - 2pi.
        for iphi in range(fields['nphi']-1):
            RR = R[-1, iphi, :]
            ZZ = Z[-1, iphi, :]
            area =np.abs(0.5*np.sum(ZZ[:-1]*np.diff(RR) - RR[:-1]*np.diff(ZZ)))
            lenght = np.sum(np.sqrt(np.diff(RR)**2 + np.diff(ZZ)**2))
            avg_area_lcfs += area
        avg_area_lcfs /= (fields['nphi']-1)
        aeff=np.sqrt(avg_area_lcfs/np.pi)
      
        ## --------------------------------------------------------------------                   
        ## -------Extend Radial surfaces --------------------------------------  
        ## --------------------------------------------------------------------      
        ## calcualte vector into the radial direction (last two grid-points) 
        p1 = np.array([r_surf_sol[ns-2,:, :],z_surf_sol[ns-2,:, :]])
        p2 = np.array([r_surf_sol[ns-1,:, :],z_surf_sol[ns-1,:, :]])
        vec=p2-p1       
        avg_area = np.zeros(ns_extended)
        nphi_minus1 = fields['nphi'] - 1  # Precompute for efficiency

        for j in range(ns, ns_extended):
            dl = aeff / ns  # Initialize dl for the first iteration
    
            for ii in range(5):  # Perform 5 iterations to adjust dl
                if ii == 0:
                    dl = aeff / ns  # Set initial dl
    
                # Update the surface solutions using vectorized operations
                r_surf_sol[j, :, :] = r_surf_sol[j - 1, :, :] + vec[0, :, :] * dl
                z_surf_sol[j, :, :] = z_surf_sol[j - 1, :, :] + vec[1, :, :] * dl
    
                # Vectorize the iphi loop to compute areas
                # Extract relevant slices
                RR = r_surf_sol[j, :nphi_minus1, :]  # Shape: (nphi-1, ...)
                ZZ = z_surf_sol[j, :nphi_minus1, :]  # Shape: (nphi-1, ...)
    
                # Compute differences along the phi axis
                diff_RR = RR[:, 1:] - RR[:, :-1]  # Shape: (nphi-1, ...)
                diff_ZZ = ZZ[:, 1:] - ZZ[:, :-1]  # Shape: (nphi-1, ...)
    
                # Compute the area using the shoelace formula vectorized across iphi
                # Assuming the last dimension is compatible for summation
                # Modify axis as per the actual data shape
                # Here, we assume the last axis is the one to sum over
                term = ZZ[:, :-1] * diff_RR - RR[:, :-1] * diff_ZZ  # Shape: (nphi-1, ...)
                area = 0.5 * np.abs(np.sum(term, axis=-1))  # Shape: (nphi-1,)
    
                # Accumulate the areas for avg_area[j]
                avg_area[j] += np.sum(area)
    
                # Compute average area
                avg_area[j] /= nphi_minus1
    
                # Calculate reff and adjust dl
                reff = np.sqrt(avg_area[j] / np.pi)
                s_test = (reff / aeff) ** 2
    
                # Prevent division by zero or invalid factors
                denominator = s_test - s_extended[j - 1]
                if denominator == 0:
                    factor = 1.0  # Default factor if denominator is zero
                else:
                    factor = (s_extended[j] - s_extended[j - 1]) / denominator
    
                dl *= factor  # Scale dl for the next iteration
        R=r_surf_sol
        Z=z_surf_sol
        ns=ns_extended
        s_dom=s_extended



    ## ------------------------------------------------------------------------               
    ## ---------- Calculate surface area and volume ---------------------------
    ## ------------------------------------------------------------------------
    surf = np.zeros(ns)
    vol = np.zeros(ns)
    avg_area = np.zeros(ns)
    for nn in range(ns):
        ## loop only until nphi-1 since the last and first points are identical when going from 0 - 2pi.
        for iphi in range(fields['nphi']-1):
            RR = R[nn, iphi, :]
            ZZ = Z[nn, iphi, :]
            area = np.abs(0.5 * np.sum(ZZ[:-1] * np.diff(RR) - RR[:-1] * np.diff(ZZ)))
            lenght = np.sum(np.sqrt(np.diff(RR)**2 + np.diff(ZZ)**2)) ## circumference of the poloidal area
            avg_area[nn] += area
            dl = np.mean(RR) * fields['dphi']
            vol[nn] += area * dl
            surf[nn] += lenght * dl
        avg_area[nn] /= (fields['nphi']-1)

    # if the phi-range doesn't cover 2pi-consider a scaling factor:
    factor = (2 * np.pi / (fields['phimax'] - fields['phimin']))
    surf *= factor
    vol *= factor
    
    
    

    '''
    s=vol/np.max(vol[np.arange(len(wout.s_dom))])
    plt.figure()
    plt.plot(s_dom)
    plt.plot(s)
    '''
    
    isep=np.argmin(np.abs(s_dom-1.))
    print('LCFS area: %5.2f'%(surf[isep]/1.e4),'m^2')
    print('Plasma volume: %5.2f'%(vol[isep]/1.e6),'m^3')

    ## determine the major radius (average of r_surf)
    rmajor=np.mean(R[isep,:,:])
    ## calcaulte the effective minor radius
    # V=2*np.pi*rmajor *np.pi rminor**2
    # --> rminor=np.sqrt(V/(2*np.pi**2*rmajor))
    rminor=np.sqrt(vol[isep]/(2*np.pi**2*rmajor))
    print('Major radius: %5.2f'%(rmajor/1.e2),' m')
    print('Minor radius: %5.2f'%(rminor/1.e2),' m') 


    ## ------------------------------------------------------------------------
    ## -------------- Reduce Array sizes when phi-ran is specified-------------
    ## ------------------------------------------------------------------------
    if phi_ran:
        ## reduce the size of the phi-grid according to phi-ran:
        index=(fields['phi']> phi_ran[0]%fields['dphi_sym']) & (fields['phi'] < phi_ran[1]%fields['dphi_sym']) 
        fields['nphi']=np.sum(index)
        fields['phi']=fields['phi'][index]
        R=R[:,index,:]
        Z=Z[:,index,:]
    fields['phimin']=fields['phi'][0]
    fields['phimax']=fields['phi'][-1]
    print(np.rad2deg(fields['phimin']),np.rad2deg(fields['phimax']))
        
    
    ## ------------------------------------------------------------------------
    ## ---------------- Get the Magnetic Field vectors ------------------------
    ## ------------------------------------------------------------------------
    if calc_brzphi:
        keys=['R','Z','Bs', 'Bv', 'Bu',
              'dR_ds', 'dR_dv', 'dR_du',
              'dZ_ds', 'dZ_dv', 'dZ_du']
        wout.transForm_3D(fields['theta'],fields['phi'], keys)
        calc_cylnd_Bvec(wout)
        if extended_vmec_factor >1:
                wout_Br= wout.Br[:, :, :]
                wout_Bz= wout.Bz[:, :, :]     
                wout_Bphi= wout.Bphi[:, :, :]             
        else:
            ## -----------------------------------------               
            ## ------- extend magnetic field -----------
            ## -----------------------------------------
            ## Populate SOL grid with the B-field of the outermost point of the wout file
            wout_Br= np.zeros((ns,fields['nphi'],fields['ntheta'])) 
            wout_Br[0:ns-ns_sol,:,:]=wout.Br[:, :, :]
            wout_Bz= np.zeros((ns,fields['nphi'],fields['ntheta']))
            wout_Bz[0:ns-ns_sol,:,:]= wout.Bz[:, :, :]     
            wout_Bphi= np.zeros((ns,fields['nphi'],fields['ntheta']))
            wout_Bphi[0:ns-ns_sol,:,:]=wout.Bphi[:, :, :]    
            for j in range(ns-ns_sol,ns):
                wout_Br[j,:,:]=wout.Br[-1, :, :]
                wout_Bz[j,:,:]=wout.Bz[-1, :, :]            
                wout_Bphi[j,:,:]=wout.Bphi[-1, :, :]        

     
        
        
    
    ## ------------------------------------------------------------------------
    ## -------------- Define 3D output cylindrical grid -----------------------
    ## ------------------------------------------------------------------------
    nr = int((np.max(R) - np.min(R)) / drz)
    nz = int((np.max(Z) - np.min(Z)) / drz)
    rgrid, zgrid = np.mgrid[np.min(R):np.max(R):complex(nr),
                            np.min(Z):np.max(Z):complex(nz)]  
    print('nr,nphi,nz: ', nr, fields['nphi'], nz)


    ## ------------------------------------------------------------------------  
    ## ------ perform the 2D interpolation using griddata ---------------------
    ## ------------------------------------------------------------------------ 
    # R and Z have dimensions [npts,nphi,ntheta]
    ## ------- make s_dom 2D (r,theta) --
    s_dom2d = np.zeros([ns, fields['ntheta']])
    for i in range(fields['ntheta']):
        s_dom2d[:, i] = s_dom
    s_grid = np.zeros([nr, nz, fields['nphi']])
    for iphi in range(fields['nphi']):
        rzdata = np.transpose(np.array([R[:, iphi, :].flatten(), Z[:, iphi, :].flatten()]))
        s_grid[:, :, iphi] = griddata(rzdata, s_dom2d.flatten(), (rgrid, zgrid),method='cubic',fill_value=np.nan)
        
    if calc_brzphi:
        Br = np.zeros([nr, nz, fields['nphi']])
        Bz = np.zeros([nr, nz, fields['nphi']])
        Bphi = np.zeros([nr, nz, fields['nphi']])     
        ## The Magnetic field from VMEC has some problem in the very core. Therefore we are going to 
        ## iscard s-value points lower than s=0.05
        score=np.argmin(np.abs(s_dom-0.05))
        
        for iphi in range(fields['nphi']):
            rzdata = np.transpose(np.array([R[score::, iphi, :].flatten(), Z[score::, iphi, :].flatten()]))
            Br[:, :, iphi] = griddata(rzdata, (wout_Br[score::, iphi, :]).flatten(), (rgrid, zgrid),method='cubic')
            Bz[:, :, iphi] = griddata(rzdata, (wout_Bz[score::, iphi, :]).flatten(), (rgrid, zgrid),method='cubic')
            Bphi[:, :, iphi] = griddata(rzdata, (wout_Bphi[score::, iphi, :]).flatten(), (rgrid, zgrid),method='cubic')
    # set ra to 2 when NAN otherwise
    s_grid[np.isnan(s_grid)] = 2.

    ## ------------------------------------------------------------------------
    ## --------------- Store data in the fields dictionary ---------------------
    ## ------------------------------------------------------------------------
    ds = np.mean(np.diff(s_dom))
    sc = s_dom[0:-1] + ds * 0.5  # grid centers
    fields['flux_s'] = sc
    fields['flux_ds'] = ds
    fields['flux_s_min'] = fields['flux_s'][0]-0.5*fields['flux_ds']
    fields['flux_ra'] = np.sqrt(sc)
    fields['flux_nr'] = int(len(sc))
    fields['flux_ns'] = fields['flux_nr']
    fields['flux_dvol'] = np.diff(vol)
    fields['flux_area'] = area
    fields['flux_surf'] = surf[1::] 
    ##
    fields['rminor']=rminor
    fields['rmajor']=rmajor


    # 3D grid
    fields['nr'] = nr
    fields['nz'] = nz
    fields['s'] = s_grid

    fields['R'] = rgrid[:, 0] ## boundaries for linear interpolation
    fields['Z'] = zgrid[0, :]
    fields['dR'] = (fields['R'][1] - fields['R'][0])
    fields['dZ'] = fields['Z'][1] - fields['Z'][0]
    fields['dvol'] = fields['dR'] * fields['dZ'] * (fields['R'] * fields['dphi'])
    ## rmin and rmax are exactly at the boundaryies as the s-grid interpolation needs it like that!
    fields['Rmax'] = np.max(fields['R'])
    fields['Rmin'] = np.min(fields['R'])
    fields['Zmax'] = np.max(fields['Z'])
    fields['Zmin'] = np.min(fields['Z'])
    


    # Flux-surface contour lines
    ## save only few contour lines to save memory:
    s_vals=[0.05,0.2,0.4,0.6,0.8,1.0,1.1,1.2]
    ns_surf=len(s_vals)

    indices=np.zeros(ns_surf,dtype=int)
    for ii in range(ns_surf):
        indices[ii]=np.argmin(np.abs(s_dom-s_vals[ii]))
    fields['s_surf'] = s_dom[indices]
    fields['ns_surf'] = ns_surf
    fields['Rsurf'] = R[indices, :, :]
    fields['Zsurf'] = Z[indices, :, :]
    fields['Xaxis'] = np.mean(R[0, :, :], axis=1) * np.cos(fields['phi'])
    fields['Yaxis'] = np.mean(R[0, :, :], axis=1) * np.sin(fields['phi'])
    fields['Zaxis'] = np.mean(Z[0, :, :], axis=1)
    fields['Rmean'] = (fields['Rmin'] + fields['Rmax']) / 2.
    # magnetic fields: Br, Bz, Bphi
    fields['calcb'] = calc_brzphi
    if calc_brzphi:
        fields['Br'] = Br*bfield_exp_factor
        fields['Bz'] = Bz*bfield_exp_factor
        fields['Bphi'] = Bphi*bfield_exp_factor
    else:
        fields['Br'] =np.zeros((1,1,1))
        fields['Bz'] =np.zeros((1,1,1))
        fields['Bphi'] =np.zeros((1,1,1))        
    # this is for axi-symmetric equilibria where we are able to rotate...
    fields['rotate_phi_grid'] = 0.
    
    
    ## freeup some memory
    del R,Z,s_grid,wout,s_dom2d
    if calc_brzphi:
        del Br,Bz,Bphi,wout_Br,wout_Bz,wout_Bphi


    return(fields)