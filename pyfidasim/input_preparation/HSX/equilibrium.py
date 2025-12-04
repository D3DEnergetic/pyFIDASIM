import numpy as np

def equilibrium(woutfile='',extended_vmec_factor=1.,drz=2.,phi_ran=None,calc_brzphi=True,directory='',vmecID=''):
    from pyfidasim.hdf5 import save_dict, load_dict
    nsym=4
    if woutfile =='':
        woutfile = 'Data/wout_HSX_qhsExtend.nc'
        extended_vmec_factor = 1.5
        #woutfile = directory+'wout_HSX_main_opt0.nc'
        #extended_vmec_factor = 1.

 
    sdict = {'calc_brzphi':True,'extended_vmec_factor':extended_vmec_factor,
             'drz': drz}

    if phi_ran:
        phi_ran_string='_phi_'+str(np.round(phi_ran[0]/np.pi,2))+'-'+str(np.round(phi_ran[1]/np.pi,2))+'pi'
    else:
        phi_ran_string='_phi_full'
    fields_file = 'Data/fields_id'+vmecID+'_drz_'+str(drz)+phi_ran_string+'.pkl'

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
        from pyfidasim._fields import _generate_fields
        fields = _generate_fields(directory+woutfile,nsym,phi_ran=phi_ran,drz=sdict['drz'],
                               calc_brzphi=sdict['calc_brzphi'],
                               extended_vmec_factor=sdict['extended_vmec_factor'])
        save_dict(fields, fields_file)
    return fields
    