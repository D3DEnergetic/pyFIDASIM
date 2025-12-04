"""
Preparation of plasma profiles for pyfidasim. Following structure:
    profiles = {}
    profiles['s'] : Vmec s=(r/a)**2  
    profiles['te'] : Electron temp [keV]     
    profiles['ti'] : Ion temp [keV]            
    profiles['dene'] : Electron density [cm-3] 
    profiles['omega'] : Plasma rotation     
    profiles['zeff'] : Effective charge
    profiles['taup'] : Particle confinement time
    profiles['ai] : Mass of ions in plasma


For retrieving Thomson data w7xdia and w7xspec are needed:
https://git.ipp-hgw.mpg.de/boz/w7xdia
https://git.ipp-hgw.mpg.de/olfo/w7xspec

For getting XICS data (Ti profile) the MDSplus python package is needed (do not build from source!): 
https://www.mdsplus.org/index.php/Introduction
"""
import numpy as np
from scipy.interpolate import interp1d
import os
import warnings

from pyfidasim.toolbox import radial_profile, load_dict, save_dict
from pyfidasim.input_preparation.W7X import equilibrium

try:
    from w7xspec.oliford import TS_get_all
    import w7xdia as w7xdia
except (ModuleNotFoundError, ImportError):
    print("Import error")
    print("If not using local profile data clone and install w7xspec and w7xdia from\n \
          https://git.ipp-hgw.mpg.de/boz/w7xdia\n https://git.ipp-hgw.mpg.de/olfo/w7xspec")



# decorator to cache plasma profiles locally
def cache_profiles(function_to_cache):
    def wrapper(*args, **kwargs):
        try: 
            if kwargs['use_cache'] == False: use_cache = False
        except Exception:
            use_cache = True
            
        if use_cache:
            file_id = args[0] + '_' +str( args[1] )
            if not os.path.isdir('./profiles_cache'): os.mkdir('./profiles_cache')
            file = './profiles_cache/' + file_id + '.hdf5'
            if os.path.isfile(file):
                #print('Load cached profile data from ' + file[0:-5])
                return load_dict(file)
            else:
                profiles = function_to_cache(*args, **kwargs)
                save_dict(profiles, file)
                #print("Saved profiles to local cache at:" + os.getcwd() + "/profiles_cache")
        else:
            profiles = function_to_cache(*args, **kwargs)
        return profiles
    return wrapper


@cache_profiles
def get_plasma_profiles(progID, time, ti_diagnostic='XICS', const_zeff=None, use_cache=True):
    """
    Loads Thomson, XICS and CXRS data to fit Te, Ti and ne profiles. Fitting method lowess provided
    the w7xdia package. To get XICS data MDSplus needs to be installed. If zeff is None it will load
    line integrated data from the archive and use that value as a flat profile.

    Args:
        progID (string): yyyymmdd.shot
        time (double): shot time in ms
        ti_diagnostic (str, optional): Option to use 'CXRS'. Defaults to 'XICS'.
        const_zeff (float, optional): Constant zeff profile. Defaults to None.
        use_cache (bool, optional): Cache fitted profiles. Defaults to True.

    Returns:
        dict: Profiles ready to use with pyfidasim
    """
    warnings.filterwarnings("ignore")
    print("All warnings suppressed")
    
    # Get vmec run and minor radius
    # vmecRefVac, vmecRefBeta = w7xspec.oliford.getVmecIDs(progID) 
    # vmecRef = vmecRefBeta
    # print("Using VMEC ID: %s" % vmecRef)
    vmecID, b0_scaling = equilibrium.get_reference_equilibrium(progID)
    a_minor = equilibrium.get_minor_radius(vmecID)
    
    # vmec s coordinate defined as s = (reff/a)^2 where a is the minor radius
    s = np.linspace(0, 1., 60)
    reff = np.sqrt(s) * a_minor
    #? Can we define reff and then have not evenly spaced s coordinate?
    # reff = np.linspace(0, 0.53, 60)
    # s = (reff / 0.53)**2
    
    # ne and Te profiles from Thomson measurements -------------------------------------------------
    
    # get thomson data
    thomson_data = TS_get_all(progID)

    # fit profiles
    fit = fit_thomson_profiles(thomson_data, 2*time/1000, fitting_window=0.05, n_edge=0)
    reff_fit, ne, Te = fit['rEff'], fit['ne'], fit['Te']
    
    # select time point of interest
    index = np.argmin(np.abs(fit['time'] - time/1000) )
    
    ne = interp1d(reff_fit, ne[index,:])( reff ) * 1e13  # 1/cm^3
    Te = interp1d(reff_fit, Te[index,:])( reff )         # keV
    
    
    # Ti data + fit from CXRS or XICS measurements -------------------------------------------------
    if ti_diagnostic == 'CXRS':
        # head=None -> All lines of sight. Other analysis branch: BGSubtract
        spec, head = "ILS_Green", None
        cx_data = w7xdia.cxrs.get_all(progID, spectrometer=spec, head=head, 
                               analysisBranch="DualGauss", getConfig=True)
        if cx_data['versionDescription'] is not None:
            print("CXRS Data fit comment: '%s'" % cx_data['versionDescription'])

        #Get REff values from VMEC, extrapolated outside LCFS
        w7xdia.cxrs.get_rEff(cx_data)
        
        # fit cx data and interpolate on reff used in pyfidasim
        reff_fit, Ti = fit_cxrs_ti_profile(cx_data, time/1000)
        Ti = interp1d(reff_fit, Ti)( reff ) 
    
    elif ti_diagnostic == 'XICS':
        try:
            import w7xdia.xics
        except Exception as e:
            print(e)
            print("Install MDSplus from: https://www.mdsplus.org/index.php/Introduction")
            import sys
            raise Exception("Error")
        
        # load data
        t, reff_data, Ti_data, sigma, mask = w7xdia.xics.get_inverted_Ti(progID)
        # select time point
        j = np.argmin(np.abs(t - time/1000))
        reff_data, Ti_data, sigma, mask = reff_data[:,j], Ti_data[:,j], sigma[:,j], mask[:,j]
        # unreliable profile points are used anyway but print the range from which on unreliable
        unreliable = np.where(mask==0)[0]
        print(" Unreliable XICS data points for reff > ", reff_data[unreliable[0]])
        # interpolate xics data
        Ti = np.interp(reff, reff_data, Ti_data)
        
    
    # Flat line integrated zeff profile ------------------------------------------------------------
    if const_zeff is None:
        t, zeff_data = w7xdia.zeff.get_zeff_signal(progID, return_errors=False, returnVersion=False)
        flat_zeff = interp1d(t, zeff_data)(time/1000)
        print("Flat Zeff profile with line integrated data from archive. Value: " + str(flat_zeff))
    else:
        flat_zeff = const_zeff
        print("Flat zeff profile set to : " + str(flat_zeff))
    zeff = np.repeat(flat_zeff, len(s))
    
    
    # Plasma rot omega, particle confinement time, bulk ion mass -----------------------------------
    omega = np.zeros(len(s))
    print("Plasma rotation set to 0")
    tau_p = 0.1  # s, only important for neutral den calcs
    ai = 1.  # amu

    # fill dict
    profiles = fill_dict(s, ne, Te, Ti, zeff, omega, tau_p, ai)
    warnings.filterwarnings("default")
    
    return profiles


def fit_thomson_profiles(ts_data, maxTime, fitting_window, n_edge):
    tsFit = dict()
    # use TS time point
    tsFit['time'] = ts_data['time'][(ts_data['time'] > 0) & (ts_data['time'] < maxTime)] 

    tsFit['rEff'] = np.linspace(0, 0.7, 100)
    for param in ['ne', 'Te']: 
        tsFit[param] =  np.nan * np.ones((len(tsFit['time']), len(tsFit['rEff'])))
    tsFit['inFit'] = np.zeros((len(tsFit['time']), len(ts_data['rEff']))) > 0
    tsFit['int'] = None
    tsFit['reNorm'] = None
    tsFit['iTS'] = -1 * np.ones((len(tsFit['time'])), dtype='int')

    for iTF in range(0, len(tsFit['time'])) :
        #nearest TS data time point, (might be exact)
        iTTS = np.argmin((ts_data['time'] - tsFit['time'][iTF])**2)

        for k, param in enumerate(['ne', 'Te']): 

            #local vars
            x = ts_data['rEff'].copy()
            y = ts_data[param][iTTS, :]
            yErr = ts_data[param + '_sigma'][iTTS, :]
            l = ts_data['l']
            R = ts_data['R']

            #remove invalid points
            i = np.isfinite(x) & np.isfinite(y) 
            i = (i) & ((R < 5.29) | (R > 5.33))
            if max(i) == False: continue  # no valid points
            
            x = x[i]; y = y[i]; yErr = yErr[i]; l = l[i]; R = R[i]
            tsFit['inFit'][iTF, :] = i

            if n_edge > 0 :
                iE = np.arange(0,n_edge);
                x = np.concatenate([x, 0.55 + 0.01 * iE])
                y = np.concatenate([y, iE * 0]);
                yErr = np.concatenate([yErr, iE*0 + 0.01])
                l = np.concatenate([l, 2.00 + iE * 0.01])
                R = np.concatenate([R, 7.00 + iE * 0.01])

            yErr[yErr <= 0] = 0.1;

            #Use w7xdia fitting to get a nice fit with ~5cm resolution
            xx = tsFit['rEff']
            yy = w7xdia.fits.fit_profile_lowess(xx, x, y, yErr, window=fitting_window, power=1)
            yy[yy<0.] = 0.

            tsFit[param][iTF, :] = yy
            tsFit['iTS'][iTF] = iTTS
    return tsFit


def fit_cxrs_ti_profile(cx_data, time):
    import w7xdia
    # index j for valid data points
    j = np.where(np.isfinite(cx_data['nominalREff']))[0]
    # reorder channels by average valid R
    j=j[np.argsort(cx_data['nominalREff'][j])] 
    # filter out points outside R=6.0 that didn't measure anything
    j=j[cx_data['nominalR'][j] < 6.0]

    iT = np.argmin((cx_data['time'] - time)**2)
    
    x = cx_data['nominalREff'][j]
    y = cx_data['Ti'][:,j][iT,:]
    yerr = cx_data['TiErr'][:,j][iT,:]
    
    #Use w7xdia fitting to get a nice fit with ~5cm resolution
    xx = np.linspace(0, 0.6, 100);
    yy = w7xdia.fits.fit_profile_lowess(xx, x, y, yerr, window=0.05)
    return xx, yy


def load_cooker_profile(JSON_file, const_zeff):
    """
    Load a JSON file generated by the W7X profile cooker. In this file fits of the
    Thomson ne and Te profile and a fit of the XICS or CXRS Ti profile must be included.
    Profile cooker: https://w7x-profiles.ipp-hgw.mpg.de/
    To get the json file from the webservice use the 'get fits' button!

    Args:
        JSON_file (json): Fit data from profile cooker
        const_zeff (float): Set to use flat zeff profile.

    Returns:
        dict: Profiles ready to use with pyfidasim
    """
    import json
    with open(JSON_file, 'r') as myfile:
        data = myfile.read()
        data = json.loads(data)
    
    try:
        vmecID = data['thomson_Te']['settings']['vmecid']
        a_minor = equilibrium.get_minor_radius(vmecID)  # m
    except Exception:
        print('Could not retrieve minor radius from vmec id! Set to 0.53')
        a_minor = 0.53
        
    # pyfidasim profiles prepared over vmec s ( =(r/a)**2 )
    s = np.linspace(0, 1., 80) 
    reff = np.sqrt(s) * a_minor
    
    import sys
    try:
        # electron temp from thomson
        Te = data['thomson_Te']['value']
        reff_te = data['thomson_Te']['reff']
        Te = interp1d(reff_te, Te)
        # electron den from thomson
        ne = np.array(data['thomson_ne']['value']) * 1e13 # cm-3
        reff_ne = data['thomson_ne']['reff']
        ne = interp1d(reff_ne, ne)
    except KeyError:
        print("Thomson data not found. Make sure to use 'get fits' and not 'get data' in the profile cooker webservice!")
        raise Exception("Error")
    try:
        # ion temp from xics or cxrs
        if 'xics_Ti' in data:
            Ti = data['xics_Ti']['value']
            reff_ti = data['xics_Ti']['reff']
        elif 'cxrs_Ti' in data:
            Ti = data['cxrs_Ti']['value']
            reff_ti = data['cxrs_Ti']['reff']
        Ti = interp1d(reff_ti, Ti)
    except Exception:
        print("Ti data not found. Make sure to use 'get fits' and not 'get data' in the profile cooker webservice!")
        raise Exception("Error")
        
    # as far as I see there is no zeff inclusion in profile cooker so we set it here manually to a
    # flat profile
    zeff = np.repeat(const_zeff, len(s))
    
    omega = np.repeat(0., len(s))
    tau_p = 0.1
    ai = 1.
    print("Plasma rotation set to 0 \nParticle confinement time set to 0.1s")
    
    # interpolate profiles at reff = sqrt(s) * a_minor
    ne, Te, Ti = ne(reff), Te(reff),Ti(reff)
    
    profiles = fill_dict(s, ne, Te, Ti, zeff, omega, tau_p, ai)
    print("Loaded Te, Ti, and ne profiles from: " + JSON_file)
    return profiles


def fill_dict(s, dene, te, ti, zeff, omega, tau_p, ai=1.):
    """
    Fills a dictionary in the form required by pyfidasim. It also adds exponential decays of the
    profiles at s > 1.01.
    """
    length = len(s)
    assert(all(array.ndim == 1 for array in [dene, te, ti, zeff, omega]))
    assert(all(len(array) == length for array in [dene, te, ti, zeff, omega]))
    
    # in some profile fits negative/zero temperatures or dens show up, set these to small value
    ti[ti<=0.0] = 1e-10
    te[te<=0.0] = 1e-10
    dene[dene<=0.0] = 1e-10
    
    profiles = {}
    profiles['s'] = s  
    profiles['dene'] = dene
    profiles['te'] = te
    profiles['ti'] = ti
    profiles['zeff'] = zeff
    profiles['omega'] = omega
    profiles['ai'] = ai  
    profiles['taup'] = tau_p # [s] particle confinement time
    
    if max(profiles['s']) < 1.01:
        # add exponential decay of profiles at plasma edge
        s_new = np.linspace(0, 1.2, 50)
        index = s_new > 1
        names = ['dene', 'te', 'ti', 'omega', 'zeff']
        decay = [0.1, 0.1, 0.1, 0.1, 1000.]
        for i, name in enumerate(names):
            profiles[name] = np.interp(s_new, profiles['s'], profiles[name])
            profiles[name][index] = profiles[name][-1] * np.exp((1 - s_new[index]) / decay[i])
        profiles['s'] = s_new
        
    profiles['ra'] = np.sqrt(s)  
    return profiles


def load_plasma_profile_from_hdf5(file='Data/W7Xprofiles.h5'):
    '''
    Load plasma profiles from a hdf5 file
    '''
    return load_dict(file)


def create_arbitrary_profiles(s=None, Te=None, Ti=None, ne=None, zeff=None, ai=1., tau_p=0.1):
    
    profiles = {}
    if s is None:
        profiles['s'] = np.linspace(0,1.,41)
        profiles['ra'] = np.sqrt(profiles['s']) 
        profiles['te'] =  radial_profile(profiles['ra'],f_0=3.5,mu=1.,offset=0.1)
        profiles['ti'] =  radial_profile(profiles['ra'],f_0=1.5,mu=1.,offset=0.1)
        profiles['dene'] = radial_profile(profiles['ra'],f_0=0.5,mu=0.2,offset=0.1)*1.e14
        profiles['zeff'] =  np.repeat(1.5, len(profiles['s']))
    else:
        assert(len(Te)==len(s) and len(Ti)==len(s) and len(ne)==len(s))
        profiles['s'] = s
        profiles['ra'] = np.sqrt(profiles['s']) 
        profiles['te'] = Te
        profiles['ti'] = Ti 
        profiles['dene'] = ne 
        profiles['zeff'] = zeff

    print("Plasma rotation set to 0 \n Plasma ion mass set to 1 u \n Particle conf time set to 0.1s")
    profiles['taup'] = tau_p
    profiles['omega']=  np.zeros(len(profiles['s']))
    profiles['ai'] = ai  
    
    
    zimp=6.
    profiles['denimp'] = np.zeros((1,len(profiles['s'])))
    profiles['denimp'][0,:]=profiles['dene'] * (profiles['zeff'] - 1) / (zimp**2 - zimp)
    profiles['denp']=profiles['dene'] - profiles['denimp'][0,:]*zimp
    

    return profiles


def plot_profiles(profiles_dict, savefig=False):
    from pyfidasim.toolbox import plot_profiles
    plot_profiles(profiles_dict, savefig=savefig)
    