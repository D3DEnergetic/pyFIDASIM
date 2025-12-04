import numpy as np
import matplotlib.pyplot as plt


def ssplit(ll):

    tmp = ll.replace('-', ' -')
    tmp = tmp.replace('e -', 'e-')
    tmp = tmp.replace('E -', 'E-')
    slist = tmp.split()
    a = [float(i) for i in slist]
    return(a)


def read_ufile(file):
    # === Open the woutfile
    f = open(file, 'r')
    lines = f.readlines()
    label = {}
    values = {}
    n = {}
    varss = []
    nvar = 0
    var_arr = ['X', 'Y', 'Z']
    # first check for the independent variables
    for lin in lines:
        if 'INDEPENDENT VARIABLE' in lin.upper():
            nvar += 1
            try:
                var = ((lin.upper().split('INDEPENDENT VARIABLE')
                        [1]).split(':')[1]).strip()
                var = var.split('-')[0]
            except BaseException:
                var = var_arr[nvar - 1]
            varss.append(var)
            label[var] = lin.split(';-')[0][1:]

    # now read further details of the header
    for lin in lines:
        a = lin.split()
        if '-SHOT #' in lin or '; Shot #' in lin:
            #shot = int(a[0][:5])
            dim = int(a[1])
        for var in varss:
            svar = '-# OF %s PTS-' % var
            if svar in lin:
                n[var] = int(a[0])
            svar = ';-# of radial pts  %s' % var
            if svar in lin:
                n[var] = int(a[0])
        # if '-DEPENDENT VARIABLE LABEL-' in lin:
            #flabel = lin.split(';-')[0][1: ]

    if dim != nvar:
        print('Inconsistency in the number of independent variables dim=%i nvar=%i' % (
            dim, nvar))

    list_var = varss[:nvar]
    jstart = 3 + 2 * dim + 2
    while True:  # skip over lines until we find first true data line
        try:
            temp = np.array(lines[jstart].split(), dtype=float)
            break
        except BaseException:
            jstart += 1

    for var in list_var:
        vtmp = []
        for jlin, lin in enumerate(lines[jstart:]):
            vtmp += ssplit(lin)
            if len(vtmp) == int(n[var]):
                jstart += jlin + 1
                break
        values[var] = np.array(vtmp, dtype=np.float64)

    jdata = jstart
    ftmp = []
    for lin in lines[jdata:]:
        if 'END-OF-DATA' in lin:
            break
        ftmp += ssplit(lin)

    farr = np.array(ftmp, dtype=np.float64)
    if dim == 1:
        fvalues = farr
    elif dim == 2:
        fvalues = farr.reshape(int(n[varss[1]]), int(n[varss[0]])).T
    elif dim == 3:
        fvalues = farr.reshape(int(n[varss[2]]), int(
            n[varss[1]]), int(n[varss[0]])).T
    return(label, values, fvalues)


def nbi_parameters_transp(transp, doplt=False):
    file = transp['directory'] + transp['runid'] + 'TR.DAT'
    nml = {'divra': [], 'divza': [], 'foclra': [], 'foclza': [], 'abeama': []}
    names = list(nml.keys())

    f = open(file, 'r')  # We need to re-open the woutfile
    for line in f:
        if line[0] == '!':
            continue
        if 'nbeam' in line.lower():
            nml['nbeam'] = int(line.split('=')[1])
        if 'nshot' in line.lower():
            nml['nshot'] = int(line.split('=')[1])

        if 'prenb2' in line.lower():
            prenb2 = str((line.split('=')[1]).split("'")[1])
        if 'extnb2' in line.lower():
            extnb2 = str((line.split('=')[1]).split("'")[1])

        # loop over the keys in nml
        for name in names:
            if name in line.lower():
                try:
                    nml[name].append(float(line.split('=')[1]))
                except BaseException:
                    nml[name].append(str(line.split('=')[1]))
    f.close()

    # define the UFILE name
    file = transp['directory'] + \
        ''.join(prenb2.strip() + str(nml['nshot']) + '.' + extnb2.strip())
    # read the ufile
    label, xvalues, data = read_ufile(file)

    # Search for the label containing the time
    names = list(label.keys())
    for ii in range(len(names)):
        if label[names[ii]].upper().find('TIME') != -1:
            time_ufile = xvalues[names[ii]]
            break

    idx = (np.abs(time_ufile - transp['time'])).argmin()
    nbi_param = {}
    for ii in range(nml['nbeam']):
        if doplt:
            fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True,
                                                 figsize=([8., 12.]))  # frameon=False removes frames
            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax1.plot(time_ufile, data[:, ii] / 1.e6, label=str(ii + 1))
            ax2.plot(time_ufile, data[:, ii + nml['nbeam']] / 1.e3)
            ax3.plot(time_ufile, data[:, ii + 2 * nml['nbeam']])
            ax3.plot(time_ufile, data[:, ii + 3 * nml['nbeam']])
            ax1.legend()

        params = {}
        params['power'] = data[idx, ii] / 1.e6
        params['voltage'] = data[idx, ii + nml['nbeam']] / 1.e3

        full = data[idx, ii + 2 * nml['nbeam']]
        half = data[idx, ii + 3 * nml['nbeam']]
        third = 1 - full - half
        params['fraction'] = [full, half, third]
        if nml['abeama'] == 2:
            params['type'] = 'D'
        if nml['abeama'] == 1:
            params['type'] = 'H'
        nbi_param[str(ii + 1)] = params

    return(nbi_param)

def nbi_geometry_transp(transp):
    ''' 
    code to read the NBI geometry from a TRANSP namelist
    '''
    ##---------------------------------------------------------
    ## ----- define the namelist variable to be considered ----
    ##---------------------------------------------------------    
    file = transp['directory'] + transp['runid'] + 'TR.DAT'
    nml = {'rtcena': [], 'xlbtna': [], 'xybsca': [], 'xbzeta': [], 'xlbapa': [], 'xybapa': [], 'nlco': [],
           'bmwidr': [], 'bmwidz': [], 'divr': [], 'divz': [], 'foclr': [], 'foclz': [], 'abeama': [],
           'nbapsha': [], 'rapedga': [], 'xzpedga': [], 'xrapoffa': [], 'xzapoffa': [],
           'nbapsh2': [], 'rapedg2': [], 'xzpedg2': [], 'xrapoff2': [], 'xzapoff2': [], 'xlbapa2': []}
    ##---------------------------------------------------------
    ##-----  open namelist and search for the variables -------
    ##---------------------------------------------------------    
    names = list(nml.keys())
    f = open(file, 'r')  # We need to re-open the woutfile
    for line in f:
        if line[0] == '!':
            continue

        if 'nbeam' in line.lower():
            nml['nbeam'] = int(line.split('=')[1])

        # current direction
        if 'nljccw' in line.lower():
            nml['nljccw'] = line.split('=')[1]
        # loop over the keys in nml
        for name in names:
            if name in line.lower():
                try:
                    nml[name].append(float(line.split('=')[1]))
                except BaseException:
                    nml[name].append(str(line.split('=')[1]))
    f.close()


    ## there is a difficulty when the aperture is circular because in 
    ## that case, xzpedga is not defined!
    xzpedga = np.zeros(nml['nbeam'])
    index = np.array(nml['nbapsha']) == 1.0
    xzpedga[index] = nml['xzpedga']
    nml['xzpedga'] = xzpedga
    
    xzpedg2 = np.zeros(nml['nbeam'])
    index = np.array(nml['nbapsh2']) == 1.0
    if sum(index) == len(nml['xzpedg2']):
        xzpedg2[index] = nml['xzpedg2']
    else:
        xzpedg2[index] = np.array(nml['xzpedg2'])[index]
    nml['xzpedg2'] = xzpedg2

    # define source names
    sources = []
    for ii in range(nml['nbeam']):
        sources.append(str(ii + 1))
        
    if np.all(nml['xrapoffa']==0) and np.all(nml['xzapoffa']==0):
        nml['xrapoffa']=list(np.zeros(nml['nbeam']))
        nml['xzapoffa']=list(np.zeros(nml['nbeam']))
    if np.all(nml['xrapoff2']==0) and np.all(nml['xzapoff2']==0) and np.all(nml['xlbapa2']==0):       
        nml['xrapoff2']=list(np.zeros(nml['nbeam']))
        nml['xzapoff2']=list(np.zeros(nml['nbeam']))
        nml['rapedg2']=list(np.ones(nml['nbeam'])*100.)
        nml['xzpedg2']=list(np.ones(nml['nbeam'])*100.)
        nml['xlbapa2']=nml['xlbapa']
        nml['nbapsh2']=list(np.ones(nml['nbeam']))
        
        
    
    ## ------------------------------------------------------
    ## transform the TRANSP beam geometry to the FIDASIM one
    ## ------------------------------------------------------
    nbi_geometry = {'sources': sources}
    for ii in range(nml['nbeam']):
        xyz_src = np.zeros(3)
        xyz_rt = np.zeros(3)
        xyz_vec = np.zeros(3)
        xyz_aper = np.zeros(3)
        xyz_src[2] = nml['xybsca'][ii]  # elevation of beam source
        xyz_rt[2] = xyz_src[2] + nml['xlbtna'][ii] * \
            (+nml['xybapa'][ii] - nml['xybsca'][ii]) / nml['xlbapa'][ii]

        # distance source to tangency point in the horizontal plane:
        dist_src_rt = np.sqrt(nml['xlbtna'][ii]**2 -
                              (xyz_src[2] - xyz_rt[2])**2)
        xyz_src[0] = np.sqrt(nml['rtcena'][ii]**2 + dist_src_rt**2) * \
            np.cos(nml['xbzeta'][ii] * np.pi / 180.)
        xyz_src[1] = np.sqrt(nml['rtcena'][ii]**2 + dist_src_rt**2) * \
            np.sin(nml['xbzeta'][ii] * np.pi / 180.)

        gamma = np.arctan2(dist_src_rt, nml['rtcena'][ii])

        # get the direction of the beam
        if ('T' in nml['nlco'][ii]) & ('T' in nml['nljccw']):
            #print('beam and current are counterclockwise')
            gamma *= 1.  # beam direction is counter clockwise
        if ('T' in nml['nlco'][ii]) & ('F' in nml['nljccw']):
            gamma *= -1.  # beam direction is clockwise
        if ('F' in nml['nlco'][ii]) & ('T' in nml['nljccw']):
            gamma *= -1.  # beam direction is clockwise
        if ('F' in nml['nlco'][ii]) & ('F' in nml['nljccw']):
            gamma *= 1.  # beam direction is counter clockwise

        xyz_rt[0] = nml['rtcena'][ii] * \
            np.cos(gamma + nml['xbzeta'][ii] * np.pi / 180.)
        xyz_rt[1] = nml['rtcena'][ii] * \
            np.sin(gamma + nml['xbzeta'][ii] * np.pi / 180.)

        vec = xyz_rt[:] - xyz_src[:]
        xyz_vec[:] = vec / np.linalg.norm(vec)

        # position of the aperture
        xyz_aper[:] = xyz_src[:] + xyz_vec[:] * nml['xlbapa'][ii]
        
        def calc_arot_brot(direction):
            assert(direction.size == 3)
            y = direction[2]
            x = np.sqrt(np.sum(direction[:]**2))
            b = np.arctan2(y, x)
            Arot = np.array([[np.cos(b), 0., np.sin(b)],
                             [0., 1., 0.],
                             [-np.sin(b), 0., np.cos(b)]])
            y = direction[1]
            x = direction[0]
            a = np.arctan2(y, x)
            Brot = np.array([[np.cos(a), -np.sin(a), 0.],
                             [np.sin(a),  np.cos(a), 0.],
                             [0., 0., 1.]])
            return Arot, Brot
        
        Arot, Brot = calc_arot_brot(xyz_vec)
        
        # define a dictionary with the FIDASIM geometry
        geom = {}
        geom['ID'] = ''
        geom['source_position'] = xyz_src
        geom['direction'] = xyz_vec
        geom['ion_source_size'] = np.array(
            [nml['bmwidr'][0], nml['bmwidz'][0]]) * 2.
        geom['focal_length'] = np.array([nml['foclr'][0], nml['foclz'][0]])
        geom['divergence'] = np.array([nml['divr'][0], nml['divz'][0]])
        geom['uvw_xyz_rot'] = Brot @ Arot
        
        ## aperture 1:
        geom['aperture_1_size'] = np.array(
            [nml['rapedga'][ii], nml['xzpedga'][ii]]) * 2
        geom['aperture_1_distance'] = nml['xlbapa'][ii]
        geom['aperture_1_offset'] = np.array(
            [nml['xrapoffa'][ii], nml['xzapoffa'][ii]])
        if int(nml['nbapsha'][ii]) == 1:
            geom['aperture_1_rectangular'] = True
        else:
            geom['aperture_1_rectangular'] = False
        ## aperture 2:
        geom['aperture_2_size'] = np.array(
            [nml['rapedg2'][ii], nml['xzpedg2'][ii]]) * 2
        geom['aperture_2_distance'] = nml['xlbapa2'][ii]
        geom['aperture_2_offset'] = np.array(
            [nml['xrapoff2'][ii], nml['xzapoff2'][ii]])
        if int(nml['nbapsh2'][ii]) == 1:
            geom['aperture_2_rectangular'] = True
        else:
            geom['aperture_2_rectangular'] = False



        nbi_geometry[str(ii + 1)] = geom
    return(nbi_geometry)

def transp_nbi(transp, doplt=False):
    nbigeom = nbi_geometry_transp(transp)
    nbiparams = nbi_parameters_transp(transp, doplt)
    
    nbi = {}
    nbi['sources'] = nbigeom['sources']
    for src in nbigeom['sources']:
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