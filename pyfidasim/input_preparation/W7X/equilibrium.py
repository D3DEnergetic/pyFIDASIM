import numpy as np
from pyfidasim.toolbox import load_dict, save_dict

def get_reference_equilibrium(program = '20180823.035', time = [1.,2.]):
    import archivedb as db
    import requests as rq
    t = db.get_program_from_to(program)
    verURL1 = 'http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/views/Minerva/Equilibrium/ReferenceEquilibrium/?'
    t1 = 'filterstart=' + str(np.int64(t[0]))
    t2 = '&filterstop=' + str(np.int64(t[1]))
    url = verURL1 + t1 + t2
    ans=rq.get(verURL1 + t1 + t2, headers={"Accept": "application/json"})
    if 'time_intervals' in ans.json().keys():    
        ver = (ans.json())['time_intervals'][0]['href'].split('/')[8]
        print('VMEC EQ version:', ver)
        
        genURL1 = '/ArchiveDB/raw/W7XAnalysis/Equilibrium/RefEq_PARLOG/'
        genURL2 = '/parms/equilibriumID/' 
        genURL3 = '/parms/scalingFactorB0/'
        filter_query = '?from={:18.0f}&upto={:18.0f}'.format(t[0], t[1])
        
        url = genURL1+ver+genURL2
        url = 'http://archive-webapi.ipp-hgw.mpg.de'+url+ '_signal.json'+filter_query
        req = rq.get(url, headers={'Accept':'application/json'})
        vmecID = (req.json())['values'][0]
        
        url = genURL1+ver+genURL3
        url = 'http://archive-webapi.ipp-hgw.mpg.de'+url+ '_signal.json'+filter_query
        req = rq.get(url=url, headers={'Accept':'application/json'})
        b0scaling = (req.json())['values'][0]
    else:
        print('no discharge found')
        vmecID = ''
        b0scaling = 0
    
    return vmecID, b0scaling

def get_wout(vmecID = 'w7x_ref_66', savenc = True):
    from osa import Client
    vmec = Client('http://esb.ipp-hgw.mpg.de:8280/services/vmec_v5?wsdl')
    wout_netcdf = vmec.service.getVmecOutputNetcdf(vmecID)
    if savenc:
        file = open('./Data/wout_'+vmecID+'.nc', 'wb')
        file.write(wout_netcdf)
        file.close()
    woutpath = './Data/wout_'+vmecID+'.nc'
    return wout_netcdf, woutpath

def get_minor_radius(vmecID = 'w7x_ref_66'):
    """
    Get minor plasma radius from vmec reference ID.

    Args:
        vmecID (str, optional):Defaults to 'w7x_ref_66'.

    Returns:
        double: Minor radius in [m]
    """
    from osa import Client
    vmec = Client('http://esb.ipp-hgw.mpg.de:8280/services/vmec_v5?wsdl')
    aminor = max(vmec.service.getReffProfile(vmecID))
    print('Minor radius: '+'{:.2f}'.format(aminor*100.)+' cm')
    return aminor

def equilibrium(progID='', timerange=[], woutpath = '', vmecID = '', 
                extended_vmec_factor = 1., b0_factor = 1.,
                drz=2.,phi_ran=None):
    '''
    Default:
    progID = '20180823.037'
    - investigated discharge number
    Optional:
    timerange = not yet in use
    - time range investigated
    woutpath = './Data/wout_w7x_ref_11.nc'
    - path of the already existing wout file to read from
    vmecID = 'w7x_ref_66'
    - reference vmec equilibrium ID for restoring field
    extended_vmec_factor = 1.
    - it is possible to scale the flux labeling with this factor
    b0_factor = 1.
    - it is possible to scale the magnetic field    
    '''

    print('-=== Generating equilibrium ===-')
    nsym=5 ## define the symmetry of W7X
    
    if progID != '':
        vmecID, b0_factor = get_reference_equilibrium(progID)
        print(progID+' - '+vmecID)
        wout, woutpath = get_wout(vmecID)
    elif vmecID != '':
        print(vmecID)
        wout, woutpath = get_wout(vmecID)
    elif woutpath != '':
        # vmecID = 'userDefinedWout'
        vmecID = woutpath
        ind = vmecID.find('/')
        while ind != -1:
            vmecID = vmecID[ind+1:]
            ind = vmecID.find('/')   
        vmecID = '_' + vmecID[0:vmecID.find('.nc')]
        woutpath = woutpath
    else:
        print('No info found for equilibrium generation.')
        print('Use standard wout file (standard config)')
        woutpath = './Data/wout_W7X_standard.nc' #default EQ
        
    print('B0 scaling factor:', b0_factor)
    print('Flux expansion factor', extended_vmec_factor)
 
    sdict = {'calc_brzphi':True,
             'extended_vmec_factor':extended_vmec_factor,
             'bfield_exp_factor':b0_factor,
             'drz': drz}
    
    if phi_ran:
        phi_ran_string='_phi_'+str(np.round(phi_ran[0]/np.pi,2))+'-'+str(np.round(phi_ran[1]/np.pi,2))+'pi'
    else:
        phi_ran_string='_phi_full'
    fields_file = 'Data/fields_id'+vmecID+'_drz_'+str(drz)+phi_ran_string+'_'+str(b0_factor)+'.pkl'


    ### See whether an sdict 3D equilibrium already exists
    ## if yes, load it
    try:
        fields = load_dict(fields_file)
        for i in sdict:
            if isinstance(sdict[i], (list, np.ndarray)): ## if sdict is a list or array
                if not isinstance(fields[i], (list, np.ndarray)): ## if test dict is not a list or array, return
                    raise ValueError
                if not np.array_equal(sdict[i], fields[i]): ## if values are not the same, return
                    raise ValueError
            else:
                if isinstance(fields[i], (list, np.ndarray)): ## if sdict_test is not a single argument, return
                    raise ValueError
                if sdict[i] != fields[i]: ## if the two are not the same, return
                    raise ValueError
    ### if not, we need to generate it!
    except (ValueError, FileNotFoundError):
        print("Recreating fields, may take some time")
        from pyfidasim._fields import generate_fields
        fields = generate_fields(woutpath, 
                               nsym, phi_ran=phi_ran,
                               drz=sdict['drz'],
                               calc_brzphi=sdict['calc_brzphi'],
                               extended_vmec_factor=sdict['extended_vmec_factor'],
                               bfield_exp_factor=sdict['bfield_exp_factor'])
        save_dict(fields, fields_file)
    return fields


if __name__ == '__main__':
    get_minor_radius(vmecID = 'w7x_ref_416')
