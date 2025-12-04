import numpy as np
from pyfidasim.toolbox import radial_profile, load_dict

def plasma_profiles_from_pkl(file='Data/W7Xprofiles.pkl', impurities = ['Carbon'],zimps=[6]):
     
    profiles = load_dict(file)
    profiles['omega'] = radial_profile(
        profiles['s'], f_0=0.5, mu=0.2, offset=0.1) * 1.e3  # [rad/s]
    profiles['ai'] = 1.
    profiles['taup'] = 0.1  # [s] particle confinment time
    
    if max(profiles['s']) < 1.01:
        # add exponential decay
        s_new = np.linspace(0, 1.2, 50)
        index = s_new > 1
        names = ['dene', 'te', 'ti', 'omega', 'zeff']
        decay = [0.1, 0.1, 0.1, 0.1, 1000.]
        for i, name in enumerate(names):
            profiles[name] = np.interp(s_new, profiles['s'], profiles[name])
            profiles[name][index] = profiles[name][-1] * \
                np.exp((1 - s_new[index]) / decay[i])
        # also to the impurity species if those are available
        try:
            for impurity in impurities:
                decay = 0.1
                profiles[impurity]        = np.interp(s_new, profiles['s'], profiles[impurity])
                profiles[impurity][index] = profiles[impurity][-1] * np.exp((1 - s_new[index]) / decay)
        except KeyError:
            pass
        profiles['s'] = s_new
        
    # in case that species resolved profiles are passed use those
    profiles['denimp'] = np.zeros((len(impurities),len(profiles['s'])))
    profiles['denp']=profiles['dene'] - profiles['denimp'][0,:]*zimps[0]
    try:
        for impurity in impurities:
            profiles['denimp'][impurity]=profiles[impurity]
            
        subtractor = np.zeros((len(profiles['s'])))
        for impurity in impurities:
            subtractor     += profiles[impurity] * zimps[impurity]
        # enforce quasi neutrality
        profiles['denp']=profiles['dene'] - subtractor
            
    # in case that only Z_eff is passed, use this and calculate the impurity density assuming everything is Carbon
    except KeyError:
        profiles['denimp'][0,:]=profiles['dene'] * (profiles['zeff'] - 1) / (zimps[0]**2 - zimps[0])
        profiles['denp']=profiles['dene'] - profiles['denimp'][0,:]*zimps[0]
        
    profiles['denimp'] = np.array(profiles['denimp'])
    profiles['ra']=np.sqrt(profiles['s'])
    
    return(profiles)