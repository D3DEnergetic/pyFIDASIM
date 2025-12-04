import numpy as np

def nbi_geometry(kind='DNB'):
    """Fills a blank dictionary with nbi geometry related parameters.

    Returns
    ----------
    nbi_geometry : dictionary, nbi geometry parameters
    """

    S1_geometry = {}
    S1_geometry['ID'] = 'S1'  # source's name
    
    
    S1_geometry['aperture_1_distance'] = 0.1
    S1_geometry['aperture_2_distance'] = 0.1
    S1_geometry['aperture_1_size'] = np.array([50., 0.])  # horizontal & vertical direction
    S1_geometry['aperture_2_size'] = np.array([50., 0.])

    S1_geometry['ion_source_size'] = np.array([8., 8.])  # rectangular ion-source's size [cm] -- actually, that one should be circular!
    S1_geometry['focal_length'] = np.array([130., 130.])  # focal length [cm]
    S1_geometry['divergence'] = np.array([0.016, 0.016])  # beam divergence in [radian]
 
    S1_geometry['aperture_1_offset'] = np.zeros(2)  # aperture offset //not yet in use
    S1_geometry['aperture_1_rectangular']= False
    S1_geometry['aperture_2_offset'] = np.zeros(2)
    S1_geometry['aperture_2_rectangular']= False
    
    
    if kind=='tangential':
        #heating beam injection
        S1_geometry['source_position'] = np.array([-8.1, 154.6, 6.3]) #source's position [cm]
        S1_geometry['direction'] = np.array([0.62, -0.455, -.4]) #NBI's los direction
        S1_geometry['direction'] = S1_geometry['direction'] / np.linalg.norm(S1_geometry['direction'])
    elif kind == 'radial':
        ## radial injection
        S1_geometry['source_position'] = np.array([100.,   40.,   50.]) #source's position [cm]        
        S1_geometry['direction'] = np.array([0.6,  0, -0.9]) #NBI's los direction
        S1_geometry['direction'] = S1_geometry['direction'] / np.linalg.norm(S1_geometry['direction'])
        

    elif kind== 'DNB':
        # DNB injection geometry
        S1_geometry['source_position'] = np.array([13.01, 144.42, 23.65])  # source's position [cm]
        S1_geometry['direction'] = np.array([-0.482, 0., -0.875])  # NBI's los direction
        S1_geometry['direction'] = S1_geometry['direction'] / np.linalg.norm(S1_geometry['direction'])
        ## move the source position such that is it 1.4 m above the Z=0 plane (according to the drawings from Santhosh)
        S1_geometry['source_position']-=S1_geometry['direction']*105.
        S1_geometry['aperture_1_distance'] = 23.
        S1_geometry['aperture_2_distance'] = 104.
        S1_geometry['aperture_1_size'] = np.array([7.5, 0.])  # horizontal & vertical direction
        S1_geometry['aperture_2_size'] = np.array([3.2, 0.])
        S1_geometry['ion_source_size'] = np.array([8., 8.])  # rectangular ion-source's size [cm] -- actually, that one should be circular!
        S1_geometry['focal_length'] = np.array([160., 160.])  # focal length [cm]
        S1_geometry['divergence'] = np.array([0.008, 0.008]) # beam divergence in [radian]
        print(S1_geometry['divergence'])
        
    elif kind == 'box_port':
        ## position along beam provided by Kevin:
        pos=np.array([84.32,84.32,0])
        S1_geometry['direction'] = np.array([+1., -1., 0.]) #NBI's los direction
        S1_geometry['direction'] = S1_geometry['direction'] / np.linalg.norm(S1_geometry['direction'])
        S1_geometry['source_position'] = pos-150*S1_geometry['direction']
        
    
    #---------------------------------------------------
    # define Euler angles and rotational matrices
    #---------------------------------------------------
    y = S1_geometry['direction'][2]
    x = np.sqrt(S1_geometry['direction'][0]**2+S1_geometry['direction'][1]**2)
    b = np.arctan2(y, x)
    Arot = np.array([[np.cos(b), 0., np.sin(b)],
                     [0., 1., 0.],
                     [-np.sin(b), 0., np.cos(b)]])
    y = S1_geometry['direction'][1]
    x = S1_geometry['direction'][0]
    a = np.arctan2(y, x)
    Brot = np.array([[np.cos(a), -np.sin(a), 0.],
                     [np.sin(a),  np.cos(a), 0.],
                     [0., 0., 1.]])
    
    S1_geometry['Arot'] = Arot
    S1_geometry['Brot'] = Brot
    nbi_geometry = {}
    nbi_geometry['sources'] = ['S1']
    nbi_geometry['S1'] = S1_geometry
    return nbi_geometry

def nbi_parameters():
    """Fills a dictionary with nbi parameters.
    Returns
    ----------
    nbi_params : dictionary, nbi parameters
    """
    nbi_params = {}
    nbi_params['sources'] = ['S1']
    params = {}
    params['voltage'] = 28.
    params['power'] = 4.* params['voltage']*1.e3/1.e6 # MW
    params['fraction'] = [0.94, 0.06, 0]
    params['type'] = 'H'
    nbi_params['S1'] = params
    return nbi_params

def HSX_nbi(kind='DNB'):
    nbigeom = nbi_geometry(kind)
    nbiparams = nbi_parameters()
    
    nbi = {}
    nbi['sources'] = nbiparams['sources']
    for src in nbiparams['sources']:
        #Combine Arot & Brot into one rotation matrix
        nbi[src] = {}
        nbi[src]['uvw_xyz_rot'] = nbigeom[src]['uvw_xyz_rot']
        nbi[src]['aperture_1_distance'] = nbigeom[src]['aperture_1_distance']
        nbi[src]['aperture_1_offset'] = nbigeom[src]['aperture_1_offset']
        nbi[src]['aperture_1_rectangular'] = nbigeom[src]['aperture_1_rectangular']
        nbi[src]['aperture_1_size'] = nbigeom[src]['aperture_1_size']
        nbi[src]['aperture_2_distance'] = nbigeom[src]['aperture_2_distance']
        nbi[src]['aperture_2_offset'] = nbigeom[src]['aperture_2_offset']
        nbi[src]['aperture_2_rectangular'] = nbigeom[src]['aperture_2_rectangular']
        nbi[src]['aperture_2_size'] = nbigeom[src]['aperture_2_size']
        nbi[src]['direction'] = nbigeom[src]['direction']
        nbi[src]['divergence'] = nbigeom[src]['divergence']
        nbi[src]['focal_length'] = nbigeom[src]['focal_length']
        nbi[src]['ion_source_size'] = nbigeom[src]['ion_source_size']
        nbi[src]['source_position'] = nbigeom[src]['source_position']
        
        nbi[src]['current_fractions'] = nbiparams[src]['fraction']
        nbi[src]['power'] = nbiparams[src]['power']
        nbi[src]['voltage'] = nbiparams[src]['voltage']
    
    return nbi