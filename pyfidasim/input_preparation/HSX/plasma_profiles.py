import numpy as np

def get_plasma_profiles(f0_ne=2.0,offset_ne=1.5,mu_ne=0.5,f0_Te=1.,offset_Te=0.2,mu_Te=2.,Te_exp=False,f0_Ti=0.04,offset_Ti=0.04,mu_Ti=0.1,zimp=6,zeff=1.5):
    from pyfidasim.toolbox import radial_profile
    profiles = {}
    profiles['s'] = np.linspace(0, 1.2, 221)
    profiles['ra'] =np.sqrt(profiles['s'])
    profiles['te'] = radial_profile(profiles['ra'], f_0=f0_Te, mu=mu_Te, offset=offset_Te,sol_decay=0.1,exp=Te_exp)
    
    profiles['ti'] = radial_profile(profiles['ra'], f_0=f0_Ti, mu=mu_Ti, offset=offset_Ti,sol_decay=0.1)
    
    
    profiles['dene'] = radial_profile(profiles['ra'], f_0=f0_ne, mu=mu_ne,offset=offset_ne,sol_decay=1.) * 1.e13
    profiles['omega'] = radial_profile(profiles['ra'], f_0=0.0, mu=0.2, offset=0.0,sol_decay=1.) * 1.e3  # [rad/s]
    profiles['zeff'] = profiles['ra'] * 0. + zeff
    profiles['n0']= np.power(10,radial_profile(profiles['ra'],f_0=-4.,mu=2.,offset=10.,sol_decay=1.e3)) # [cm^-3]
    
    zimp=6.
    profiles['zimp']=zimp
    profiles['denimp'] = np.zeros((1,len(profiles['s'])))
    profiles['denimp'][0,:]=profiles['dene'] * (profiles['zeff'] - 1) / (zimp**2 - zimp)
    profiles['denp']=profiles['dene'] - profiles['denimp'][0,:]*zimp
    
    return(profiles)
    
    
    
if __name__ == "__main__":  
    import sys
    sys.path.insert(1, '../../../')

    import matplotlib.pyplot as plt
    plt.close('all')

    profiles=get_plasma_profiles()
    
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, sharex=True,figsize=[9,9])
    plt.subplots_adjust(hspace=.1)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    #plot density profile
    ax1.plot(profiles['s'],profiles['dene']/1.e13, label='density')
    ax1.legend(loc=1,fontsize=14)
    ax1.set_ylabel(r'$10^{19}\mathrm{m}^{-3}$')
    ax1.set_ylim(bottom=0)
    #plot temperature profile
    ax2.plot(profiles['s'],profiles['te'],label='Te')
    ax2.plot(profiles['s'],profiles['ti'],label='Ti') 
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel('keV')
    ax2.legend(loc=1,fontsize=14)  
    
    ## plot neutral density
    ax3.semilogy(profiles['s'],profiles['n0'],label='n0')     
    ax3.set_ylabel(r'$n_0$ [cm$^{-3}$]')    
    ax3.legend(loc=1,fontsize=14)   
    ax3.set_xlabel('s')