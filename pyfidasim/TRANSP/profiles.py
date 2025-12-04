import matplotlib.pyplot as plt
from scipy.io import netcdf
from scipy.interpolate import interp1d
import numpy as np
import copy as cp

def read_transp_profiles(transp, doplt=False,s_new=[], ne_decay=0.1,nimp_decay=0.1, te_decay=0.1,ti_decay=0.1,omega_decay=0.1, exponential=True):
    '''
    read profile data from
    '''
    file = transp['directory'] + transp['runid'] + '.CDF'
    f = netcdf.netcdf_file(file, mode='r', version=4)
    time = cp.deepcopy(f.variables['TIME3'].data).astype(float)
    idx = (np.abs(time - transp['time'])).argmin()
    X = cp.deepcopy(f.variables['X'].data[idx, :]).astype(float)
    te = cp.deepcopy(f.variables['TE'].data[idx, :]).astype(float) / 1.e3
    ti = cp.deepcopy(f.variables['TI'].data[idx, :]).astype(float) / 1.e3
    dene = cp.deepcopy(f.variables['NE'].data[idx, :]).astype(float)
    zeff = cp.deepcopy(f.variables['ZEFFI'].data[idx, :]).astype(float)
    omega = cp.deepcopy(f.variables['OMEGA'].data[idx, :]).astype(float)
    xzimp = cp.deepcopy(f.variables['XZIMP'].data[idx]).astype(float)
    taup = cp.deepcopy(f.variables['TAUPI'].data[idx,-1]).astype(float)
    taue = cp.deepcopy(f.variables['TAUE'].data[idx,-1]).astype(float)
    pcx = cp.deepcopy(f.variables['PCX'].data[idx,:]).astype(float) ## MW/m^3              CHARGE EXCHANGE LOSS  
    try:
        pcx_halo = cp.deepcopy(f.variables['PCXHALO'].data[idx,:]).astype(float) ## W/cm^3     CX POWER to halo NEUTRALS
    except:
        pcx_halo=np.copy(pcx)*0.
    pcx_recy = cp.deepcopy(f.variables['PCXSRC'].data[idx,:]).astype(float) ## W/cm^3      CX POWER to recyling NEUTRALS  
    
    
    pisrc= cp.deepcopy(f.variables['PISRC'].data[idx,:]).astype(float)  ## W/cm^3             recyc neutral ionization POWER
    
    
    dvol = cp.deepcopy(f.variables['DVOL'].data[idx,:]).astype(float) ## cm^3
    n_imp = cp.deepcopy(f.variables['NIMP'].data[idx,:])
    try:
        dn0wd = cp.deepcopy(f.variables['DN0WD'].data[idx, :]).astype(float)
        dn0vd = cp.deepcopy(f.variables['DN0WD'].data[idx, :]).astype(float)
    except BaseException:
        dn0wd = 0
        dn0vd = 0
    try:
        dn0wh = cp.deepcopy(f.variables['DN0WH'].data[idx, :]).astype(float)
        dn0vh = cp.deepcopy(f.variables['DN0WH'].data[idx, :]).astype(float)
    except BaseException:
        dn0wh = 0
        dn0vh = 0
    f.close()
    n0 = dn0wh + dn0vh + dn0wd + dn0vd
    

    profiles = {'ra':X,'s': X, 'dene': dene, 'te': te, 'ti': ti,'omega': omega, 'zeff': zeff,\
                'n0': n0, 'xzimp': xzimp,'n_imp':n_imp, 'taup':taup, 'taue':taue,\
                'pcx':pcx,'pcx_tot':np.sum(pcx*dvol), 'pcx_halo':pcx_halo,'pcx_recy':pcx_recy,'pisrc':pisrc}
    
    ## add exponential decay to the profiles
    if exponential:
        # add exponential decay
        if len(s_new)==0:
            s_new = np.linspace(0, 1.2, 100)
        index = s_new > profiles['s'][-1]
        names = ['dene', 'n_imp','te', 'ti', 'omega', 'zeff','n0','pcx','pcx_halo','pcx_recy','pisrc']
        decay = [ne_decay,nimp_decay,te_decay,ti_decay,omega_decay, 1000.,1000.,0.0001,0.0001,0.0001,0.0001]
        for i, name in enumerate(names):
            #f=interpolate.interp1d(profiles['s'],profiles[name],kind='cubic',fill_value='extrapolate')
            interper = interp1d(profiles['s'],profiles[name],fill_value='extrapolate', bounds_error=False)
            last_val = profiles[name][-1]
            profiles[name] = interper(s_new)
            #profiles[name] = np.interp(s_new, profiles['s'], profiles[name])
            profiles[name][index] = last_val * np.exp((1 - s_new[index]) / decay[i])
        profiles['s'] = s_new
    else:
        if len(s_new)==0:
            s_new = np.linspace(0, 1.0, 100)
        names = ['dene', 'n_imp','te', 'ti', 'omega', 'zeff','n0','pcx','pcx_halo','pcx_recy','pisrc']
        for i, name in enumerate(names):
            interper = interp1d(profiles['s'],profiles[name],fill_value='extrapolate', bounds_error=False)
            profiles[name] = interper(s_new)
            #profiles[name] = np.interp(s_new, profiles['s'], profiles[name])
        profiles['s'] = s_new


    nr=len(profiles['s'])
    profiles['denimp']=np.zeros((1,nr))
    profiles['denimp'][0,:]=profiles['n_imp']
    profiles['denp']=profiles['dene'] - profiles['denimp'][0,:]*xzimp
    profiles['ra']=profiles['s']
    profiles['nr']=len(profiles['s'])
    if doplt:
        fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True,figsize=([8., 12.]))
        plt.subplots_adjust(hspace=.1)
        ax1.grid()
        ax2.grid()
        ax3.grid()
        # Question: Is the s matrix rho or rho squared? Because it is labeled
        # as rho but seems to be rho squared...
        ax1.plot(profiles['s'], profiles['dene'] / 1.e13, label=r'n${_e}$')
        ax1.plot(profiles['s'], profiles['denp'] / 1.e13, label=r'n${_i}$')  
        ax1.plot(profiles['s'], profiles['denimp'][0,:] / 1.e13, label =r'n$_{imp}$')     
        ax2.plot(profiles['s'], profiles['te'], label='T${_e}$')
        ax2.plot(profiles['s'], profiles['ti'], label='T${_i}$')
        ax2.legend(loc="lower left")
        ax3.plot(profiles['s'], profiles['omega'] / 1.e3)    
        ax1.set_ylabel(r'n [10$^{13}$cm$^{-3}$]')
        ax2.set_ylabel(r'$T$ [keV]')
        ax3.set_ylabel(r'$v_{rot}$ [krad/s]')
        ax3.set_xlabel("r/a")
        ax3.set_xlim([0., 1.2])
        
        ax1.legend(loc="lower left")
        plt.savefig('TRANSP_input_kinetic_profiles.png')
    return(profiles)

