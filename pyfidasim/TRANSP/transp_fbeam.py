def transp_fbeam(file,fields,emin=0, emax=100, pmin=-1,pmax=1):
    from scipy.io import netcdf
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator    
    import numpy as np

    # Ensure arrays are native-endian & contiguous (Numba hates >f8 / big-endian)
    def _to_native(a, dtype=np.float64):
        a = np.asarray(a, dtype=dtype)
        if not a.dtype.isnative:
            a = a.byteswap().newbyteorder()
        return np.ascontiguousarray(a)

    ## --------------------------
    ## -- read TRANSP cdf file --
    ## ---------------------------
    # read variable from the NCDF woutfile
    f = netcdf.netcdf_file(file, mode='r')
    try:
        r2d = _to_native(f.variables['R2D'].data) # rposition of cells
        z2d = _to_native(f.variables['Z2D'].data) # zposition of cells
        energy = _to_native(f.variables['E_D_NBI'].data)
        pitch  = _to_native(f.variables['A_D_NBI'].data)
        fbm    = _to_native(f.variables['F_D_NBI'].data) # fast-ion distribution function
        rsurf  = _to_native(f.variables['RSURF'].data) # flux surface
        zsurf  = _to_native(f.variables['ZSURF'].data) # flux surface
    finally:
        f.close()
    
    ## given that we read the D-ion distribution function, the mass is 2:
    afbm=2
    
    ## plot flux surface and locations of 2d points with FBMs
    #plt.figure()
    #npts2d=len(r2d)
    #for i in range(npts2d):
    #    plt.plot([r2d[i]],[z2d[i]],'kx')
    
    nr_flx = len(rsurf[:,0])
    #for i in range(nr_flx):
    #    plt.plot(rsurf[i,:],zsurf[i,:])
    
    #raxis=rsurf[0,0]
    #zaxis=zsurf[0,0]
    #plt.plot(raxis,zaxis,'o')
    #plt.xlabel('R [cm]')
    #plt.ylabel('Z [cm]')
    #plt.axis('equal')
    #plt.tight_layout()

    
    ##-------------Convert eV-> keV
    energy*=1.e-3  # fidasim needs energy in kev  
    fbm*=1.e3          # now, this needs to be corrected
    # as we now calculate with fast-ions/omega/keV/cm^3
    #------------Convert d_omega --> pitch
    # Fast-ion distribution is given as a function of cm^3, energy
    # and d_omega/4Pi. omega is the solild angle in 3D velocity space. In
    # order to transform this to a function depending on pitch instead
    # of d_omega/4PI, one has to multiply by 0.5!
    fbm*=0.5  
    #make sure that fbm is >=0:
    fbm[fbm <0]=0.
    

    # TRANSP and FIDASIM define the pitch along the current direction. 
    npitch=len(pitch)
    
    dE  = energy[2] - energy[1]
    dP  = abs(pitch[2]  - pitch[1])
    
    #----------- select energy range -------
    index=(energy >= emin) & (energy <= emax)
    energy=energy[index]    
    nenergy=np.sum(index)
    fbm=fbm[:,:,index]
    emin=energy[0]
    emax=energy[nenergy-1]
    
    # --------- select Pitch range --------
    index=(pitch >= pmin) & (pitch <= pmax)
    fbm=fbm[:,index,:]   
    npitch=np.sum(index)
    pitch=pitch[index] 
    pmin=pitch[0]  -  0.5*dP
    pmax=pitch[npitch-1]+ 0.5*dP

    # determine total fast-ion density
    fdens=np.sum(fbm,axis=((1,2)))*dE*dP

    # Build mesh from fields and normalize dtypes/endianness too
    R = _to_native(fields['R'])
    Z = _to_native(fields['Z'])
    r_mesh, z_mesh = np.meshgrid(R, Z, indexing='ij')
    r_mesh = _to_native(r_mesh)
    z_mesh = _to_native(z_mesh)
    nr, nz = r_mesh.shape
    
    ## -----------------------------------------------------------------------------
    ## Map irregular (r2d,z2d) data onto regular (r_mesh,z_mesh) via single reused triangulation
    ## -----------------------------------------------------------------------------
    points = _to_native(np.column_stack([r2d.ravel(), z2d.ravel()]))
    tri = Delaunay(points)

    # Interpolate total density once (linear), fill holes with nearest to avoid NaNs
    fdens_native = _to_native(fdens.ravel())
    den_lin  = LinearNDInterpolator(tri, fdens_native)
    den_near = NearestNDInterpolator(points, fdens_native)
    denf = den_lin(r_mesh, z_mesh)
    if np.isnan(denf).any():
        nanmask = np.isnan(denf)
        denf[nanmask] = den_near(r_mesh[nanmask], z_mesh[nanmask])

    # Make 3D, native, contiguous, float64
    denf3d = _to_native(denf)[..., None]

    # Map the full fbm stack using the same triangulation (per-slice interpolators are cheap)
    fbm_mapped = np.zeros((nenergy, npitch, nr, nz), dtype=np.float64)
    for ie in range(nenergy):
        for ip in range(npitch):
            vals = _to_native(fbm[:, ip, ie].ravel())
            lin  = LinearNDInterpolator(tri, vals)
            near = NearestNDInterpolator(points, vals)
            m = lin(r_mesh, z_mesh)
            if np.isnan(m).any():
                m_nan = np.isnan(m)
                m[m_nan] = near(r_mesh[m_nan], z_mesh[m_nan])
            fbm_mapped[ie, ip, :, :] = _to_native(m)

    # Just in case any NaNs slipped through because of geometry degeneracy
    fbm_mapped[np.isnan(fbm_mapped)] = 0.
    ## dump distribution into a dictionary
    fbm_dict={'fbm':fbm_mapped,'denf':denf3d,'afbm':afbm, \
         'energy':energy,'nenergy':nenergy,'dE':dE,'emax':emax,'emin':emin,'eran':emax-emin, \
         'pitch':pitch,'npitch':npitch,'dP':dP,'pmax':pmax,'pmin':pmin,'pran':pmax-pmin, \
         'R':R,'Z':Z,'r_mesh':r_mesh,'z_mesh':z_mesh, \
         'rsurf':rsurf,'zsurf':zsurf,'nr_flx':nr_flx,'btipsign':fields['btipsign']}    
    return(fbm_dict)


'''
if __name__ == "__main__":
    import sys
    sys.path.insert(1, '../../') 
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as matplotlib
    matplotlib.rcParams.update({'font.size': 20})
    plt.close('all')


    transp = {'time': 1.7, 'runid': '179854Z03', 'directory': '../../examples/DIIID/Data/'}
    from pyfidasim.TRANSP.fields import generate_s_grid_transp
    fields = generate_s_grid_transp(transp,Bt_sign=-1.,Ip_sign=+1.,phi_ran=[-0.4 * np.pi, 0.1 * np.pi],rmin=120)


    file = '../../examples/DIIID/Data/179854Z03_fi_1.cdf'
    fbm=transp_fbeam(file,fields,emin=0, emax=100, pmin=-1,pmax=1)
    
    ## plot the fast ion density distribution
    plt.figure()
    plt.contourf(fields['R'], fields['Z'], np.transpose(fbm['denf'][:,:,0]))
    cax = plt.colorbar()
    cax.set_label(r'fast-ion density $[cm^{-3}]$')
    for i in range(fbm['nr_flx']):
        if i%3==0:
            plt.plot(fbm['rsurf'][i,:],fbm['zsurf'][i,:],color='r',alpha=0.5)
    plt.xlabel('R [cm]')
    plt.ylabel('Z [cm]')
    plt.axis('equal')
    plt.tight_layout()


    ## plot the fast ion density velocity space
    plt.figure()
    Rplot=170.
    Zplot=10.
    a=(fbm['r_mesh']-Rplot)**2+(fbm['z_mesh']-Zplot)**2
    ind = np.unravel_index(np.argmin(a, axis=None), a.shape)


    ir=np.argmin(np.abs(fields['R']-Rplot))
    iz=np.argmin(np.abs(fields['Z']-Zplot))

    print(ir,iz,ind[0],ind[1])
    print(np.shape(fbm['r_mesh']))
    print('R=',fbm['r_mesh'][ind[0],ind[1]],'Z=',fbm['z_mesh'][ind[0],ind[1]])

                                                            ## ir ,  iz
    plt.contourf(fbm['energy'], fbm['pitch'], np.transpose(fbm['fbm'][:,:,ind[1],ind[0]]))
    #---------------------------------------------------
    # rough estimate of the tranpped-passing boundary
    #---------------------------------------------------    
    a=(fbm['rsurf']-Rplot)**2+(fbm['zsurf']-Zplot)**2
    ind = np.unravel_index(np.argmin(a, axis=None), a.shape)
    rmin=np.min(fbm['rsurf'][ind[0],:])
    pitch_boundary=np.sqrt(1.-rmin/Rplot)
    plt.plot([min(fbm['energy']),max(fbm['energy'])],[pitch_boundary,pitch_boundary],'--k')
    plt.plot([min(fbm['energy']),max(fbm['energy'])],[-pitch_boundary,-pitch_boundary],'--k')


    cax = plt.colorbar()
    cax.set_label(r'fast-ion density $[cm^{-3}/keV]$')
    plt.ylim([-1,1])
    plt.xlabel('energy [keV]')
    plt.ylabel('pitch []')
    plt.tight_layout()


    from pyfidasim.hdf5 import save_dict
    save_dict(fbm,'fbm_for_fidasim.hdf5')
'''