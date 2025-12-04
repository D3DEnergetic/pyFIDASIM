from pathlib import Path
from scipy.io import netcdf  # <--- This is the library to import.
import numpy as np
from scipy.interpolate import griddata
import scipy.constants as const
import matplotlib.pyplot as plt
import copy as cp

def generate_s_grid_transp(transp, ntheta=300, nr=0,nz=0, drz=1.,rmin=None, rmax=None, phi_ran=[0., 2. * np.pi],Bt_sign=1., Ip_sign=1.):
    '''
    Inputs:
        transp:    dictionary containing a TRANSP file name (runid+directory) and the timepoint of interest
        ntheta:    The flux surfaces in the TRANSP output are given as fourier modes. To get points that can then be interpolated
                   we use a grid of ntheta points.
        drz=1:     Resolution of the output grid in the R and Z directions
        rmin:      Output grid dimensions. If not set, the full plasma extension is considered
        rmax:      Output grid dimensions. If not set, the full plasma extension is considered
        phi_ran:   The phi-range of the output grid. THis is somewhat obsolete as we deal with 2D equilibria, but its and input used in grid3d
        Ip_sign:   Direction of the plasma current (helicity of the field)
        Bt_sign:   Direction of the magnetic field
    Output:
        fields structure containing inputs required by pyFIDASIM
    '''
    
    
    if (nr>0) or (nz >0):
        print('--------------------------------------')
        print('Attention, the nr and nz definition in generate_s_grid_transp are not used any longer.')
        print('Instead, we do now use the "drz" value which provides the grid spacing in the R and Z dimensions.')
        print('By default we use drz=1 cm.')
        print('--------------------------------------')
        
    
    file = Path(transp['directory'] + transp['runid'] + '.CDF').resolve()
    if not file.exists():
        raise FileNotFoundError(f'{file.as_posix()} does not exist')
    theta = np.linspace(0, 2 * np.pi, ntheta,endpoint=False)
    # read variable from the NCDF woutfile
    f = netcdf.netcdf_file(file.as_posix(), mode='r', version=4)
    time = cp.deepcopy(f.variables['TIME3'])
    XB = cp.deepcopy(f.variables['XB'])
    PLFLX = cp.deepcopy(f.variables['PLFLX'])  # poloidal flux on XB in Webers [XB]
    TRFLX = cp.deepcopy(f.variables['TRFLX'])   # poloidal flux on XB in Webers [XB]

    BZXR = cp.deepcopy(f.variables['BZXR'])  # vacuum field B*R
    plcurtot = cp.deepcopy(f.variables['PLCURTOT'] ) # TOTOL POLOIDAL CURRENT

    DVOL = cp.deepcopy(f.variables['DVOL'])
    DAREA = cp.deepcopy(f.variables['DAREA'])
    RAXIS = cp.deepcopy(f.variables['RAXIS'].data)    
    ZAXIS = cp.deepcopy(f.variables['YAXIS'].data)        
    ntim, nrr = XB.data.shape
    # read the moments describing the flux surfaces and generate r and z
    # positions along them
    nmom = 0
    for i in range(20):
        if i < 10:
            rc = 'RMC0%d' % i  # [XB]
        else:
            rc = 'RMC%d' % i
        try:
            RMCXX = cp.deepcopy(f.variables[rc])
            # print_info(RMCXX)
            nmom += 1
        except BaseException:
            x = 1
    r_surf = np.zeros((ntim, nrr, ntheta))
    z_surf = np.zeros((ntim, nrr, ntheta))
    for jmom in range(0, nmom):
        if jmom < 10:
            rc = 'RMC0%d' % jmom
            zc = 'YMC0%d' % jmom
        else:
            rc = 'RMC%d' % jmom
            zc = 'YMC%d' % jmom
        jtheta = jmom * theta
        thec = np.cos(jtheta)
        rr = cp.deepcopy(f.variables[rc].data)
        zz = cp.deepcopy(f.variables[zc].data)
        r_surf += rr[:, :][..., None] * thec
        z_surf += zz[:, :][..., None] * thec
    for jmom in range(1, nmom):
        if jmom < 10:
            rs = 'RMS0%d' % jmom
            zs = 'YMS0%d' % jmom
        else:
            rs = 'RMS%d' % jmom
            zs = 'YMS%d' % jmom
        jtheta = jmom * theta
        thec = np.sin(jtheta)
        rr = cp.deepcopy(f.variables[rs].data)
        zz = cp.deepcopy(f.variables[zs].data)
        r_surf += rr[:, :][..., None] * thec
        z_surf += zz[:, :][..., None] * thec
    f.close()
    ##-------------------------------------------------------------------
    ## determine the time index which correspons to time_out.
    ##-------------------------------------------------------------------
    tindex = np.abs(transp['time'] - time.data).argmin()



    ## determine the major radius (average of r_surf)
    rmajor=np.mean(r_surf[tindex,-1,:])
    print('rmajor: %5.2f'%(rmajor/1.e2),'m')
    # calculate the poloidal surface area
    poloidal_area = np.abs(0.5 * np.sum(z_surf[tindex,-1,:-1] * np.diff(r_surf[tindex,-1,:]) - r_surf[tindex,-1,:-1] * np.diff(z_surf[tindex,-1,:])))
    ## get the volume
    volume=poloidal_area * 2.*np.pi * rmajor# cm^3
    print('volume: %5.1f'%(volume/1.e6),'m^3') 
    ## calcaulte the effective minor radius
    rminor=np.sqrt(volume/(2*np.pi**2*rmajor))
    print('rminor: %5.2f'%(rminor/1.e2),'m')   





    ## --------------------------------------------------
    ## --------------------------------------------------    
    ## ---------- add points in the SOL region  ---------
    ## --------------------------------------------------
    ## --------------------------------------------------      
    ## ---------------------------------------------
    ## ----------- generate extended rho grid ------
    ## ---------------------------------------------
    nrho_sol=25
    rhot=XB.data[tindex,:]
    ## Lets add a position in the very center since this is important for extrapolating 
    ## the magnetic coordinate (s) down to zero.
    rhot=np.r_[0,rhot]
    nrr+=1
    
    
    drho=np.mean(np.diff(rhot))
    rho_sol=np.linspace(rhot[-1]+drho,rhot[-1]+nrho_sol*drho,num=nrho_sol)
    rhot_extended=np.r_[rhot,rho_sol]
    nrr_extended=nrr+nrho_sol
    ## Generate enlarged r_surf and z_surf array and fill it with the known data   
    r_surf_sol=np.zeros((nrr_extended,ntheta))
    z_surf_sol=np.zeros((nrr_extended,ntheta))  
    
    r_surf_sol[1:nrr,:]=r_surf[tindex,:,:]
    z_surf_sol[1:nrr,:]=z_surf[tindex,:,:]   
    ## Here, we populate the position in the very center since this is important for extrapolating 
    ## the magnetic coordinate (s) down to zero.   
    r_surf_sol[0,:] = RAXIS[tindex]
    z_surf_sol[0,:] = ZAXIS[tindex]

    ## -----------------------------------------------------
    ## --------- Interpolate darea grid (extrapolate) ------
    ## -----------------------------------------------------
    from scipy.interpolate import interp1d
    fa=interp1d(rhot,np.r_[0,DAREA.data[tindex, :]],kind='linear',fill_value="extrapolate")
    darea=fa(rhot_extended)
     
    ## ----------------------------------------------------------------------------------
    ## -Determine the SOL r_surf and z_surf contours that agree with the poloidal darea -
    ## ----------------------------------------------------------------------------------  
    ## calcualte vector into the radial direction (last two grid-points per theta angle) 
    p1 = np.array([r_surf[tindex, -2, :],z_surf[tindex, -2, :]])
    p2 = np.array([r_surf[tindex, -1, :],z_surf[tindex, -1, :]])
    vec=p2-p1
    for j in range(nrr,nrr_extended):
        for ii in range(3): ## use 3 interations to find the correct step size dl
            if ii ==0:
                dl=0.5
            r_surf_sol[j,:]=r_surf_sol[j-1,:] + vec[0,:]*dl
            z_surf_sol[j,:]=z_surf_sol[j-1,:] + vec[1,:]*dl   
            ## generate R and Z arrays that close by itself (add the first grid-point to the last position)
            RR=np.r_[r_surf_sol[j,:],r_surf_sol[j,0]]
            ZZ=np.r_[z_surf_sol[j,:],z_surf_sol[j,0]]
            area = np.abs(0.5 * np.sum(ZZ[:-1] * np.diff(RR) - RR[:-1] * np.diff(ZZ)))
            darea_test=area-np.sum(darea[0:j])
            factor=darea[j]/darea_test
            dl*=factor
    '''          
    plt.figure()
    plt.plot(r_surf_sol[nrr-1,:],z_surf_sol[nrr-1,:],'k')
    for i in range(ntheta):
        plt.plot(r_surf_sol[nrr::,i],z_surf_sol[nrr::,i],'.')
    ''' 
    ## there is something wrong with this formula 
    dvol=darea*2.*np.pi*np.mean(r_surf_sol,axis=1)
    ## so lets use the TRANSP definition in the confined region
    dvol[1:nrr]=DVOL.data[tindex,:]


    ## ------------------------------------------------------------------------------
    ## --------- Interpolate the fluxes onto the extended rhot grid -----------------
    ## ------------------------------------------------------------------------------   


    from scipy.interpolate import interp1d
    ## not here that we add a zero value in the center since this is important for extrapolating 
    ## the magnetic coordinate (s) down to zero.
    fp=interp1d(rhot,np.r_[0,PLFLX.data[tindex, :]],kind='linear',fill_value="extrapolate")
    ft=interp1d(rhot,np.r_[0,TRFLX.data[tindex, :]],kind='linear',fill_value="extrapolate")    
    fl=interp1d(rhot,np.r_[0,plcurtot.data[tindex, :]],kind='linear',fill_value="extrapolate")      
    pflx1d=fp(rhot_extended) ## Wb/rad
    tflx1d=ft(rhot_extended)
    fpol1d=fl(rhot_extended)
  
    #plt.figure()
    #plt.plot(rhot_extended,pflx1d)
    #plt.plot(rhot,PLFLX.data[tindex, :])
    #plt.xlabel('rhot')
    #plt.ylabel('Psi')   
    '''
    plt.figure()
    plt.plot(rhot_extended,tflx1d)
    plt.plot(rhot,np.r_[0,TRFLX.data[tindex, :]])   
    plt.xlabel('rhot')
    plt.ylabel('Phi')   
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    '''
    #plt.figure()
    #plt.plot(rhot_extended,fpol1d)
    #plt.plot(rhot,plcurtot.data[tindex, :])   
    #plt.xlabel('rhot')
    #plt.ylabel('fpol')    
  

    

    # ------------------------------------------------------------------------     
    # Expand the 1D poloidal flux and poloidal current on the nr,nthe positions
    # ------------------------------------------------------------------------    
    pflx = np.zeros((nrr_extended, ntheta))
    tflx = np.zeros((nrr_extended, ntheta))
    fpol = np.zeros((nrr_extended, ntheta))
    for i in range(ntheta):
        ## ploidal flux function
        pflx[:, i] = pflx1d  ## Wb/rad
        ## toroidal flux
        tflx[:, i] = tflx1d
        ## poloidal current density
        fpol[:, i] = fpol1d
           
    
    ##-------------------------------------------------------------------
    ## Define R and Z arrays to store the output Br,Bphi,Bz
    ##-------------------------------------------------------------------
    if not rmin:
        rmin=np.min(r_surf_sol[0:nrr])
    if not rmax:
        rmax=np.max(r_surf_sol) 
        
    zmin= np.min(z_surf_sol[0:nrr])
    zmax= np.max(z_surf_sol)
    
    
    # define number of points based on drz
    nr=int((rmax-rmin)/drz+0.5) 
    nz=int((zmax-zmin)/drz+0.5)  

    
    rarr = np.linspace(rmin,rmax, num=nr)
    zarr = np.linspace(zmin,zmax, num=nz)
    r2d, z2d = np.meshgrid(rarr, zarr)

    ## -----------------------------------------------------------------------------
    ## pflx, tflx and fpol are on a irregular grid. Now we need to map it onto the
    ## r2d,z2d grid
    ## -----------------------------------------------------------------------------
    rz_surf = np.transpose(np.array([r_surf_sol[ :, :].flatten(), z_surf_sol[ :, :].flatten()]))
    tflx_rz = griddata(rz_surf, tflx.flatten(), (r2d, z2d),method='cubic')
    pflx_rz = griddata(rz_surf, pflx.flatten(), (r2d, z2d),method='cubic') 
    ## get rhot on the 2d grid
    rhot2d = np.zeros((1,nz, nr))
    tflx_sep = tflx[nrr-1,0] ## any point on the tflx is the same. So take thefirst one. 
    rhot2d[0,:,:] = np.sqrt((tflx_rz) / tflx_sep)
    
    
    ## get the poloidal curent function
    fpol_rz = griddata(rz_surf, fpol.flatten(), (r2d, z2d),method='cubic')
    
    
    rhot2d=np.transpose(rhot2d)
    fpol_rz=np.transpose(fpol_rz)
    pflx_rz=np.transpose(pflx_rz)
    tflx_rz= np.transpose(tflx_rz)
    r2d=np.transpose(r2d)
    z2d=np.transpose(z2d)
    
    ## -----------------------------------------------------------------------
    ## -------- calcualte the corresponding B-fields -------------------------
    ## ----------------------------------------------------------------------- 
    ## Amperes law: \int B*dl = mu_0*Ip
    ## 2piR B = mu_0*Ip
    ## B= Ip*mu_0/(2pi)/R
    btor = np.zeros((nr, nz,1))
    btor[:, :,0] = (fpol_rz*const.mu_0/(2.*np.pi)*100. + BZXR.data[tindex]) / r2d
    

    dpsidz=np.zeros((nr, nz))
    for i in range(nr):
            dpsidz[i,:]=np.gradient(pflx_rz[i,:],zarr/100.)
    dpsidr=np.zeros((nr, nz))
    for j in range(nz):
        dpsidr[:,j]=np.gradient(pflx_rz[:,j],rarr/100.)    
        
    br = np.zeros((nr,nz,1))
    bz = np.zeros((nr,nz,1))
    for j in range(nz):
        br[:,j,0]=dpsidz[:,j]/(rarr/100.)
        bz[:,j,0]=-dpsidr[:,j]/(rarr/100.)
  
    ## accont for the direction of the plasma current      
    if Ip_sign == -1:
        br*=-1.
        bz*=-1.
    ## accont for the direction of the toroidal magnetic field           
    if Bt_sign == -1:
        btor*=-1.


    fields = {}
    fields['rminor']=rminor
    fields['rmajor']=rmajor
    # information on flux-surfaces: 1D volume and area
    ## not here that we introduced a position at rho=0 for the calculation of the magnetic field
    ## this is position is removed here becasue it would provide dvol=0, causing infinite densities. 
    fields['flux_s'] = rhot_extended[1::]
    fields['flux_ds'] = np.mean(np.diff(rhot[1::]))
    fields['flux_s_min'] = fields['flux_s'][0]-0.5*fields['flux_ds'] 
    fields['flux_ra'] = rhot_extended[1::]
    fields['flux_nr'] = int(nrr_extended-1)  
    fields['flux_ns'] = fields['flux_nr']
    fields['flux_dvol'] = dvol[1::]
    fields['flux_area'] = darea[1::]

    # 3D grid
    fields['nphi'] = np.int32(1)
    fields['nsym'] = 1
    fields['dphi_sym'] = np.float64(2. * np.pi)
    fields['nr'] = nr
    fields['nz'] = nz
    fields['s'] = rhot2d ## !! Attention, this is rho(r/a) and not s!
    print('Attention, for TRANSP simulations we use the s-grid s==r/a!')
    fields['R'] = rarr
    fields['Z'] = zarr
    fields['phi'] = np.array([np.float64((phi_ran[0] + phi_ran[1]) / 2.)])
    fields['dR'] = (fields['R'][1] - fields['R'][0])
    fields['dZ'] = fields['Z'][1] - fields['Z'][0]
    fields['dphi'] = np.float64(phi_ran[1] - phi_ran[0])
    fields['dvol'] = fields['dR'] * fields['dZ'] * (fields['R'] * fields['dphi'])

    ## rmin and rmax are exactly at the boundaryies as the s-grid interpolation needs it like that!
    fields['Rmax'] = np.max(fields['R'])
    fields['Rmin'] = np.min(fields['R'])
    fields['Zmax'] = np.max(fields['Z'])
    fields['Zmin'] = np.min(fields['Z'])
    fields['phimin'] = np.float64(phi_ran[0])
    fields['phimax'] = np.float64(phi_ran[1])

    # this is a rotation in case phi=0 is crossed in the initial phi_ran definition
    # this is possible only for the axi-symmetric case.
    fields['rotate_phi_grid'] = np.float64(0.)
    if fields['phimin'] < 0:
        if fields['phimax'] < 0:
            # both are negative. So we can add 2pi to both
            fields['rotate_phi_grid'] = 2. * np.pi
        else:
            # this is when we are crossing the 0-phi position
            fields['rotate_phi_grid'] = np.abs(fields['phimin'])

        fields['phimin'] += fields['rotate_phi_grid']
        fields['phimax'] += fields['rotate_phi_grid']

    # Flux-surface contour lines
    
    fields['ntheta'] = ntheta
    #fields['ra_surf'] = XB.data[tindex, :]
    fields['s_surf'] = fields['flux_s']
    
    fields['Rsurf']=np.zeros((fields['flux_ns'],1,fields['ntheta']))
    fields['Zsurf']=np.zeros((fields['flux_ns'],1,fields['ntheta']))    
    for i in range(0,fields['flux_ns']):
        fields['Rsurf'][i,0,:]=r_surf_sol[i+1,:] 
        fields['Zsurf'][i,0,:]=z_surf_sol[i+1,:]
        ## Note that +1 is needed since the rho=0 value has been removed for flux_s.
        
     
    fields['Rmean'] = 0.5 * (fields['Rmax'] + fields['Rmin'])

    # magnetic fields: Br, Bz, Bphi
    fields['calcb'] = True
    fields['Br'] =   br
    fields['Bz'] =   bz
    fields['Bphi'] = -btor
    fields['btipsign']=-1.
    
    fields['dl_max']=2.*np.sqrt(fields['Rmax']**2-fields['Rmin']**2)
    return(fields)


if __name__ == "__main__":
    plt.close('all')
    transp = {'time': 1.7, 'runid': '17985A05', 'directory': '../../examples/DIIID/Data/'}
    fields = generate_s_grid_transp(transp,drz=1.)#,rmin=120.,rmax=250.)
    


    ## plot flux surfaces    
    import mpl_toolkits.mplot3d.axes3d as p3
    fig1 = plt.figure(figsize=(8.,5.))
    ax1 = fig1.add_subplot(111, projection = '3d')
    cmap = plt.get_cmap("tab10")
    isurf=np.argmin(np.abs(fields['flux_s']-1.))
    for phi in np.arange(0.*np.pi,2.*np.pi,0.05):
        Xsurf = fields['Rsurf'][isurf,0,:] * np.cos(phi)
        Ysurf = fields['Rsurf'][isurf,0,:]  *np.sin(phi)
        Zsurf = fields['Zsurf'][isurf,0,:]
        ax1.plot3D(Xsurf,Ysurf,Zsurf,color = 'k', lw = 0.7, alpha = 0.2)
    ## set the axis of the plot equal
  
        

    plt.figure()
    plt.contourf(fields['R'], fields['Z'], np.transpose(fields['s'][:, :, 0]))
    cax = plt.colorbar()
    cax.set_label(r'$\rho_t$')
    plt.plot(fields['Rsurf'][isurf,0,:],fields['Zsurf'][isurf,0,:],color='r')
    plt.xlabel('R [cm]')
    plt.ylabel('Z [cm]')
    plt.axis('equal')
    plt.tight_layout()
    


    plt.figure()
    plt.contourf(fields['R'], fields['Z'], np.transpose(fields['Bphi'][:, :, 0]))
    cax = plt.colorbar()
    cax.set_label(r'$B_t$')
    plt.plot(fields['Rsurf'][isurf,0,:],fields['Zsurf'][isurf,0,:],color='r')
    plt.xlabel('R [cm]')
    plt.ylabel('Z [cm]')
    plt.axis('equal')
    plt.tight_layout()
    
    
    plt.figure()
    plt.contourf(fields['R'], fields['Z'], np.transpose(fields['Br'][:, :, 0]))
    cax = plt.colorbar()
    cax.set_label(r'$B_r$')
    plt.plot(fields['Rsurf'][isurf,0,:],fields['Zsurf'][isurf,0,:],color='r')
    plt.xlabel('R [cm]')
    plt.ylabel('Z [cm]')
    plt.axis('equal')
    plt.tight_layout()
    
    
    plt.figure()
    plt.contourf(fields['R'], fields['Z'], np.transpose(fields['Bz'][:, :, 0]))
    cax = plt.colorbar()
    cax.set_label(r'$B_z$')
    plt.plot(fields['Rsurf'][isurf,0,:],fields['Zsurf'][isurf,0,:],color='r')
    plt.xlabel('R [cm]')
    plt.ylabel('Z [cm]')
    plt.axis('equal')
    plt.tight_layout()


    plt.figure()
    zz=np.sqrt(fields['Bz'][:, :, 0]**2+fields['Br'][:, :, 0]**2)
    plt.contourf(fields['R'], fields['Z'], np.transpose(zz))
    cax = plt.colorbar()
    cax.set_label(r'$B_{pol}$')
    plt.plot(fields['Rsurf'][isurf,0,:],fields['Zsurf'][isurf,0,:],color='r')
    plt.xlabel('R [cm]')
    plt.ylabel('Z [cm]')
    plt.axis('equal')
    plt.tight_layout()