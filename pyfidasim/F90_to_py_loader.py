from pathlib import Path
import h5py
import re
import numpy as np
from copy import deepcopy
from pyfidasim.TRANSP.transp_fbeam import transp_fbeam
import os

def file_numbers(fp):
    """Generator to get numbers from a text file"""
    toklist = []
    while True:
        line = fp.readline()
        if not line: break
        # Match numbers in the line using regular expression
        pattern = r'[+-]?\d*[\.]?\d+(?:[Ee][+-]?\d+)?'
        toklist = re.findall(pattern, line)
        for tok in toklist:
            yield tok

def read(f):
    """ Reads a G-EQDSK file

    Parameters
    ----------
    f = Input file. Can either be a file-like object,
        or a string. If a string, then treated as a file name
        and opened.
    Returns
    -------
    """
    
    if isinstance(f, str):
        # If the input is a string, treat as file name
        with open(f) as fh: # Ensure file is closed
            return read(fh) # Call again with file object
    # Read the first line, which should contain the mesh sizes
    desc = f.readline()
    if not desc:
        raise IOError("Cannot read from input file")

    s = desc.split() # Split by whitespace
    if len(s) < 3:
        raise IOError("First line must contain at least 3 numbers")
    idum = int(s[-3])
    nxefit = int(s[-2])
    nyefit = int(s[-1])
    # Use a generator to read numbers
    token = file_numbers(f)
    try:
        xdim   = float(token.next())
        zdim   = float(token.next())
        rcentr = float(token.next())
        rgrid1 = float(token.next())
        zmid   = float(token.next())
        rmagx  = float(token.next())
        zmagx  = float(token.next())
        simagx = float(token.next())
        sibdry = float(token.next())
        bcentr = float(token.next())
        cpasma = float(token.next())
        simagx = float(token.next())
        xdum   = float(token.next())
        rmagx  = float(token.next())
        xdum   = float(token.next())
        zmagx  = float(token.next())
        xdum   = float(token.next())
        sibdry = float(token.next())
        xdum   = float(token.next())
        xdum   = float(token.next())
    except:
        xdim = float(next(token))
        zdim = float(next(token))
        rcentr = float(next(token))
        rgrid1 = float(next(token))
        zmid = float(next(token))
        rmagx = float(next(token))
        zmagx = float(next(token))
        simagx = float(next(token))
        sibdry = float(next(token))
        bcentr = float(next(token))
        cpasma = float(next(token))
        simagx = float(next(token))
        xdum = float(next(token))
        rmagx = float(next(token))
        xdum = float(next(token))
        zmagx = float(next(token))
        xdum = float(next(token))
        sibdry = float(next(token))
        xdum = float(next(token))
        xdum = float(next(token))
    # Read arrays
    def read_array(n, name="Unknown"):
        data = np.zeros([n])
        try:
            for i in np.arange(n):
                try:
                    data[i] = float(token.next())
                except:
                    data[i] = float(next(token))
        except:
            raise IOError("Failed reading array '"+name+"' of size ", n)
        return data
    def read_2d(nx, ny, name="Unknown"):
        data = np.zeros([nx, ny])
        for i in np.arange(nx):
            data[i,:] = read_array(ny, name+"["+str(i)+"]")
        return data


    fpol   = read_array(nxefit, "fpol")
    pres   = read_array(nxefit, "pres")
    workk1 = read_array(nxefit, "workk1")
    workk2 = read_array(nxefit, "workk2")
    psi    = read_2d(nxefit, nyefit, "psi")
    qpsi   = read_array(nxefit, "qpsi")
    # Read boundary and limiters, if present
    try:
        nbdry = int(token.next())
        nlim  = int(token.next())
    except:
        nbdry = int(next(token))
        nlim = int(next(token))
    if nbdry > 0:
        rbdry = np.zeros([nbdry])
        zbdry = np.zeros([nbdry])
        for i in range(nbdry):
            try:
                rbdry[i] = float(token.next())
                zbdry[i] = float(token.next())
            except:
                rbdry[i] = float(next(token))
                zbdry[i] = float(next(token))
    else:
        rbdry = [0]
        zbdry = [0]

    if nlim > 0:
        xlim = np.zeros([nlim])
        ylim = np.zeros([nlim])
        for i in range(nlim):
            try:
                xlim[i] = float(token.next())
                ylim[i] = float(token.next())
            except:
                xlim[i] = float(next(token))
                ylim[i] = float(next(token))
    else:
        xlim = [0]
        ylim = [0]
    # Construct R-Z mesh
    r = np.zeros([nxefit, nyefit])
    z = r.copy()
    for i in range(nxefit):
        r[i,:] = rgrid1 + xdim*i/float(nxefit-1)
    for j in range(nyefit):
        z[:,j] = (zmid-0.5*zdim) + zdim*j/float(nyefit-1)
    # Create dictionary of values to return
    result = {'nx': nxefit, 'ny':nyefit,        # Number of horizontal and vertical points
              'r':r, 'z':z,                     # Location of the grid-points
              'rdim':xdim, 'zdim':zdim,         # Size of the domain in meters
              'rcentr':rcentr, 'bcentr':bcentr, # Reference vacuum toroidal field (m, T)
              'rgrid1':rgrid1,                  # R of left side of domain
              'zmid':zmid,                      # Z at the middle of the domain
              'rmagx':rmagx, 'zmagx':zmagx,     # Location of magnetic axis
              'simagx':simagx, # Poloidal flux at the axis (Weber / rad)
              'sibdry':sibdry, # Poloidal flux at plasma boundary (Weber / rad)
              'cpasma':cpasma,
              'psi':psi,    # Poloidal flux in Weber/rad on grid points
              'fpol':fpol,  # Poloidal current function on uniform flux grid
              'pressure':pres,  # Plasma pressure in nt/m^2 on uniform flux grid
              'qpsi':qpsi,  # q values on uniform flux grid
              'nbdry':nbdry, 'rbdry':rbdry, 'zbdry':zbdry, # Plasma boundary
              'nlim':nlim, 'xlim':xlim, 'ylim':ylim} # Wall boundary
    return result


def generate_sgrid_fidasim(file, verbose=False):
    """

    :param file: if str (file location) or file, will read the raw GEQDSK file represented. if int, will use as a shotnumber for MDSplus.
    :param float time: REQUIRED FOR MDSplus PULLING :: time in ms at which to pull from MDSplus EFITs, rounded to the nearest EFIT.
    :return:
    """
    data_full={'time': np.array([0.]), 'dict_arr': [read(file)]}

    sgrid_arr = []

    for i in range(len(data_full['time'])):
        data = data_full['dict_arr'][i]

        ## transpose poloidal flux function
        data['psi']=np.transpose(data['psi'])
        if verbose:
            print('psi = ', data['psi'])


        ## define 1D psi grid on which 1D functions are being mapped
        npsi1d=len(data['qpsi'])
        psi1d=np.linspace(data['simagx'], data['sibdry'], num=npsi1d)
        if verbose:
            print('psi1d = ', psi1d)

        ## define the 2D poloidal current function
        fpol2d=np.interp(data['psi'],psi1d,data['fpol'])

        if verbose:
            print('B0: %.2f'%data['bcentr']+' T')
            print('R0: %.2f'%data['rcentr']+' m')

        ## calcualte the 2D toroidal magnetic field
        nr=data['nx']
        nz=data['ny']
        r2d=data['r']
        z2d=data['z']
        btor = np.zeros((nr, nz,1))
        btor[:, :,0] = fpol2d / r2d #* np.sign(data['bcentr'])  # (fpol2d + data['bcentr']*data['rcentr']) / r2d
        if verbose:
            print('btor = ', btor)

        ## calcualte the radial and vertical fields
        zarr=z2d[0,:]*100
        rarr=r2d[:,0]*100
        dpsidz=np.zeros((nr, nz))
        for ii in range(nr):
            dpsidz[ii,:]=np.gradient(data['psi'][ii,:],zarr/100.)
        dpsidr=np.zeros((nr, nz))
        for j in range(nz):
            dpsidr[:,j]=np.gradient(data['psi'][:,j],rarr/100.)

        br = np.zeros((nr,nz,1))
        bz = np.zeros((nr,nz,1))
        for j in range(nz):
            br[:,j,0]=dpsidz[:,j]/(rarr/100.)
            bz[:,j,0]=-dpsidr[:,j]/(rarr/100.)

        ## accont for the direction of the plasma current
        Ip_sign = np.sign(data['bcentr'] * data['cpasma'])
        # Ip_sign = np.sign(data['cpasma'])
        if Ip_sign == -1:
            br*=-1.
            bz*=-1.

        ## Determine the rho poloidal flux label
        rhot= np.sqrt((psi1d - data['simagx']) / (data['sibdry'] - data['simagx']))
        rhot2d=np.zeros((nr,nz,1))
        rhot2d[:,:,0] = np.sqrt((data['psi'] - data['simagx'])/(data['sibdry'] - data['simagx']))

        ## ------------------------------------------------------------------------
        ## generate the output dictionary
        ## ------------------------------------------------------------------------

        sgrid = {}
        #sgrid['rminor']=rminor
        #sgrid['rmajor']=rmajor
        sgrid['flux_s'] = rhot
        sgrid['flux_ds'] = np.mean(np.diff(rhot[1::]))
        sgrid['flux_s_min'] = sgrid['flux_s'][0]-0.5*sgrid['flux_ds']
        sgrid['flux_ra'] = rhot
        sgrid['flux_nr'] = len(rhot)
        sgrid['flux_ns'] = sgrid['flux_nr']
        #sgrid['flux_dvol'] = dvol[1::]
        #sgrid['flux_area'] = darea[1::]
        # 3D grid
        sgrid['nphi'] = np.int32(1)
        sgrid['nsym'] = 1
        sgrid['dphi_sym'] = np.float64(2. * np.pi)
        sgrid['nr'] = nr
        sgrid['nz'] = nz
        sgrid['s'] = rhot2d ## !! Attention, this is rho(r/a) and not s!
        if verbose:
            print('Attention, for TRANSP simulations we use the s-grid s==r/a!')
        sgrid['R'] = rarr
        sgrid['Z'] = zarr
        phi_ran=[0,2.*np.pi]
        sgrid['phi'] = np.array([np.float64((phi_ran[0] + phi_ran[1]) / 2.)])
        sgrid['dR'] = (sgrid['R'][1] - sgrid['R'][0])
        sgrid['dZ'] = sgrid['Z'][1] - sgrid['Z'][0]
        sgrid['dphi'] = np.float64(phi_ran[1] - phi_ran[0])
        sgrid['dvol'] = sgrid['dR'] * sgrid['dZ'] * (sgrid['R'] * sgrid['dphi'])

        ## rmin and rmax are exactly at the boundaryies as the s-grid interpolation needs it like that!
        sgrid['Rmax'] = np.max(sgrid['R'])
        sgrid['Rmin'] = np.min(sgrid['R'])
        sgrid['Zmax'] = np.max(sgrid['Z'])
        sgrid['Zmin'] = np.min(sgrid['Z'])
        sgrid['phimin'] = np.float64(phi_ran[0])
        sgrid['phimax'] = np.float64(phi_ran[1])

        # this is a rotation in case phi=0 is crossed in the initial phi_ran definition
        # this is possible only for the axi-symmetric case.
        sgrid['rotate_phi_grid'] = np.float64(0.)
        if sgrid['phimin'] < 0:
            if sgrid['phimax'] < 0:
                # both are negative. So we can add 2pi to both
                sgrid['rotate_phi_grid'] = 2. * np.pi
            else:
                # this is when we are crossing the 0-phi position
                sgrid['rotate_phi_grid'] = np.abs(sgrid['phimin'])

            sgrid['phimin'] += sgrid['rotate_phi_grid']
            sgrid['phimax'] += sgrid['rotate_phi_grid']

        # Flux-surface contour lines
        sgrid['ntheta'] = data['nbdry']
        sgrid['s_surf'] = np.array([1])
        sgrid['Rsurf']=np.zeros((1,1,sgrid['ntheta']))
        sgrid['Zsurf']=np.zeros((1,1,sgrid['ntheta']))
        sgrid['Rsurf'][0,0,:]=data['rbdry']*100
        sgrid['Zsurf'][0,0,:]=data['zbdry']*100
        sgrid['Rmean'] = 0.5 * (sgrid['Rmax'] + sgrid['Rmin'])

        # magnetic fields: Br, Bz, Bphi
        sgrid['calcb'] = True
        sgrid['Br'] =   br
        sgrid['Bz'] =   bz
        sgrid['Bphi'] = -btor
        sgrid['btipsign']=Ip_sign
        sgrid['dl_max']=2.*np.sqrt(sgrid['Rmax']**2-sgrid['Rmin']**2)
        sgrid_arr.append(deepcopy(sgrid))

    sgrid_final = {'time': data_full['time'], 'sgrid_arr': sgrid_arr}
    if len(sgrid_arr) == 1:
        return sgrid_arr[0]

    return sgrid_final

def load_dict(filename):
    if not isinstance(filename, Path):
        filename = Path(filename)
    filename = filename.resolve()
    if not filename.exists():
        print(filename.as_posix())
        raise FileNotFoundError(f'{filename.as_posix()} does not exist')
    print(f'Loading {filename.as_posix()}')
    with h5py.File(filename.as_posix(), 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            if isinstance(item[()], bytes):
                ans[key] = item[()].decode('utf8')
            else:
                ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans

def extract_variables(file_path, target_vars):
    """
    Extracts float values for specified variables from a given file.
    
    Parameters:
    - file_path (str): Path to the input file.
    - target_vars (list of str): List of variable names to extract.
    
    Returns:
    - dict: A dictionary with variable names as keys and their float values.
    """
    # Initialize a dictionary to store the results
    extracted_values = {}
    
    # Compile a regex pattern for efficiency
    # This pattern matches lines like: var = value !! comment
    pattern = re.compile(
        r'^\s*(' + '|'.join(map(re.escape, target_vars)) + r')\s*=\s*([-+]?\d*\.\d+|\d+)\s*(?:!!.*)?$'
    )
    
    try:
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                # Strip leading/trailing whitespace
                stripped_line = line.strip()
    
                # Skip empty lines or lines that start with '!!' (comments)
                if not stripped_line or stripped_line.startswith('!!'):
                    continue
    
                # Attempt to match the pattern
                match = pattern.match(stripped_line)
                if match:
                    var_name, var_value = match.groups()
                    try:
                        # Convert the extracted value to float
                        extracted_values[var_name] = float(var_value)
                    except ValueError:
                        print(f"Warning: Unable to convert value '{var_value}' for variable '{var_name}' to float (Line {line_number}).")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except IOError as e:
        print(f"IOError while reading the file: {e}")
    
    return extracted_values

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

def f90_to_py(path,runid,geqdsk,fbm_file=None, emin=0, emax=100, pmin=-1, pmax=1):
    target_variables = ['ab','pinj','einj','current_fractions(1)', 'current_fractions(2)', 
                        'current_fractions(3)','nlambda','lambdamin','lambdamax']
                        #'nx','ny','nz','xmin','xmax',
                        #'ymin','ymax','zmin','zmax','alpha','beta','gamma','origin(1)','origin(2)'
                        #'origin(3)',]

    fsim_vals = extract_variables(path+'/'+runid+'_inputs.dat', target_variables)
    geom = load_dict(path+'/'+runid+'_geometry.h5')
    equil = load_dict(path+'/'+runid+'_equilibrium.h5')
    spectra = load_dict(path+'/'+runid+'_spectra.h5')
    
    nbi2 = {}
    nbi_name = geom['nbi']['name']
    nbi2[nbi_name] = {}
    nbi2[nbi_name]['aperture_1_distance'] = geom['nbi']['adist'][0]
    nbi2[nbi_name]['aperture_1_offset'] = np.array([geom['nbi']['aoffy'][0], geom['nbi']['aoffz'][0]])
    nbi2[nbi_name]['aperture_1_rectangular'] = True if geom['nbi']['ashape'][0] == 1 else False
    nbi2[nbi_name]['aperture_1_size'] = np.array([geom['nbi']['awidy'][0]*2., geom['nbi']['awidz'][0]*2.])
    if np.shape(geom['nbi']['adist'])[0] > 1:
        nbi2[nbi_name]['aperture_2_distance'] = geom['nbi']['adist'][1]
        nbi2[nbi_name]['aperture_2_offset'] = np.array([geom['nbi']['aoffy'][1], geom['nbi']['aoffz'][1]])
        nbi2[nbi_name]['aperture_2_rectangular'] = True if geom['nbi']['ashape'][1] == 1 else False
        nbi2[nbi_name]['aperture_2_size'] = np.array([geom['nbi']['awidy'][1]*2., geom['nbi']['awidz'][1]*2.])
    else:
        nbi2[nbi_name]['aperture_2_distance'] = nbi2[nbi_name]['aperture_1_distance']
        nbi2[nbi_name]['aperture_2_offset'] = nbi2[nbi_name]['aperture_1_offset']
        nbi2[nbi_name]['aperture_2_rectangular'] = nbi2[nbi_name]['aperture_1_rectangular']
        nbi2[nbi_name]['aperture_2_size'] = nbi2[nbi_name]['aperture_1_size']
    nbi2[nbi_name]['direction'] = geom['nbi']['axis']
    if not np.all(np.isclose(geom['nbi']['divy'], geom['nbi']['divy'][0])) and not np.all(np.isclose(geom['nbi']['divz'], geom['nbi']['divz'][0])):
        print("Warning: pyFIDASIM assumes divergence of all beam components are the same")
        print("but the values provided by the FIDASIM input are not")
    nbi2[nbi_name]['divergence'] = np.array([np.average(geom['nbi']['divy']), np.average(geom['nbi']['divz'])])
    nbi2[nbi_name]['focal_length'] = np.array([geom['nbi']['focy'], geom['nbi']['focz']])
    nbi2[nbi_name]['ID'] = nbi_name = geom['nbi']['name']
    nbi2[nbi_name]['ion_source_size'] =  np.array([geom['nbi']['widy']*2., geom['nbi']['widz']*2.])
    nbi2[nbi_name]['source_position'] = np.array(geom['nbi']['src'])
    Arot, Brot = calc_arot_brot(nbi2[nbi_name]['direction'])
    nbi2[nbi_name]['uvw_xyz_rot'] = Brot @ Arot 
    nbi2['sources'] = [nbi2[nbi_name]['ID']]
    
    nbi2['ab'] = fsim_vals['ab']
    nbi2[nbi_name]['current_fractions'] = np.array([fsim_vals['current_fractions(1)'], 
                                                 fsim_vals['current_fractions(2)'], 
                                                 fsim_vals['current_fractions(3)']])
    nbi2[nbi_name]['power'] = fsim_vals['pinj']
    #nbi2[nbi_name]['type'] = 'H' if fsim_vals['ab'] <= 2.0 else 'D'
    nbi2[nbi_name]['voltage'] = fsim_vals['einj']
    
    def extract_extreme_positions(arr):
        ones_indices = np.where(arr == 1)[1]  # Get only column indices directly
        if ones_indices.size == 0:
            return -1, -1  # No 1s found
        return ones_indices.max()
    
    fields = generate_sgrid_fidasim(path+'/'+geqdsk)

    fbm = None
    if fbm_file is not None:
        if os.path.dirname(fbm_file) == '':
            full_fbm_path = os.path.join(path, fbm_file)
        else:
            full_fbm_path = fbm_file
        
        print(f"Loading FBM from: {full_fbm_path}")
        fbm = transp_fbeam(full_fbm_path, fields, emin=emin, emax=emax, pmin=pmin, pmax=pmax)
    
    from ._fields import interp_fields
    rightmost = extract_extreme_positions(equil['plasma']['mask'])
    change = (equil['plasma']['r'][1]-equil['plasma']['r'][0])/2.
    s_max = interp_fields(np.array([equil['plasma']['r'][rightmost]+change]), np.array([0]), np.array([0]), 
                      fields['s'], fields['R'], fields['Rmin'], fields['dR'], 
                      fields['nr'], fields['Z'], fields['Zmin'], fields['dZ'], 
                      fields['nz'], fields['phi'], fields['phimin'], 
                      fields['dphi'], fields['nphi'])
    
    transp = {}
    transp['directory'] = path+'/'
    transp['runid'] = runid
    transp['time'] = equil['plasma']['time']
    from pyfidasim.TRANSP.profiles import read_transp_profiles
    profiles2 = read_transp_profiles(transp, s_new=np.linspace(0, s_max[0], 100), exponential=False)
    profiles2['ai'] = equil['plasma']['species_mass'][0]
    
    spec2 = {}
    spec2['dlam'] = (np.max(spectra['lambda']) - np.min(spectra['lambda']))/spectra['nlambda']
    spec2['nlam'] = int(spectra['nlambda'])
    spec2['lambda_max'] = np.max(spectra['lambda'])
    spec2['lambda_min'] = np.min(spectra['lambda'])
    spec2['wavel'] = np.linspace(spec2['lambda_min'],spec2['lambda_max'],int(spec2['nlam']))
    spec2['only_pi'] = False
    spec2['los_pos'] = geom['spec']['lens']
    spec2['los_vec'] = geom['spec']['axis']
    spec2['nlos'] = int(geom['spec']['nchan'])
    spec2['losname'] = ['Spectrometer' + str(i) for i in range(spec2['nlos'])]

    if fbm is not None:
        return nbi2, profiles2, spec2, fields, fbm
    else:
        return nbi2, profiles2, spec2, fields