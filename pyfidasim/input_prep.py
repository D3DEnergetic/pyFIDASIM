import numpy as np

def input_prep(FIDASIM):
    '''
    FIDASIM Dictionary
    created by Aidan Edmondson ajedmondson@wisc.edu
    For details on the code, please see the following paper:
        ************

    ---------------------------------------------------------------------------
    DESCRIPTION
    ---------------------------------------------------------------------------
    The following function will return the necessary inputs to run pyFIDASIM.
    In order to do so, a dictionary containing the key names listed below along
    with their associated values will need to be supplied to the function when
    it is called.

    ---------------------------------------------------------------------------
    KEY NAMES
    ---------------------------------------------------------------------------
    
    Simulation Settings
    -------------------
    nmarker [OPTIONAL-10000]: Int
        number of Monte-Carlo markers that will be used to simulate the netural
        beam injection. The higher the value, the higher the simulation
        precision at the cost of performance
    
    calc_halo [OPTIONAL-False]: Boolean value (True or False)
        determines if halo contribution from the netural beam should be
        calculated
        
    verbose [OPTIONAL-True]: Boolean value (True or False)
        determines if simulation information and progress should be printed
        to the console for the user to see. Even if False there will be a few
        print statements
    
    calc_photon_origin [OPTIONAL-False]: Boolean value (True or False)
        determines whether or not to store the location of the photon emission
        for spectra calculation. This is the raw value in grid3d and does not 
        take into account the finite-lifetime effect. 
        
    calc_photon_origin_type [OPTIONAL-True]: Boolean value (True or False)
        only relevant if calc_photon_origin is used. determines whether or not
        to store distinguished photon types (1st energy, 2nd energy, halo etc) 
        in the photon_origin output array
        
    calc_PSF [OPTIONAL-False]: Boolean value (True or False)
        determines if the point spread function is calculated taking into 
        account finite-lifetime effect, finite fiber size, and full light 
        cone per channel.
        
    calc_density [OPTIONAL-True]: Boolean value (True or False)
        determines if the beam density information should be kept and stored
        in a machine oriented voxel grid which is stored in grid3d
        
    calc_uvw [OPTIONAL-False]: Boolean value (True or False)
        determines if the beam density information should be kept and stored
        in a beam oriented voxel grid which is stored in grid3d
    
    calc_extended_emission [OPTIONAL-False]: Boolean value (True or False)
        determines if alternative line emissions other than the Balmer-Alpha
        line should be simulated. If True, other settings in the Extended
        Emission section must be filled out
        
    respawn_if_aperture_is_hit [OPTIONAL-True]: Boolean value (True or False)
        determines if neutral beam markers that miss the aperture opening
        should be regenerated. Should be True unless there is explicit
        desire otherwise
        
    seed [OPTIONAL- -1]: Int
        sets a fixed seed for random number generation with numpy. If
        specified, the results will be numerically consistent across separate 
        runs. If set negative, the results will be default numerically random
        
    batch_marker [OPTIONAL-10000]: Int
        this setting determines how many rays to batch compute at once.
        Generally the higher the value the faster it will run at the cost
        of more memory. ~100k is best depending on the computer, but
        10k is default to prevent memory issues. Due to memory and CPU cache
        size constraints, 1m+ batch sizes can actually slow down computational
        speed (computer dependant)
    
    
    Spectrometer Data
    -----------------
    los_pos: 2D N x 3 array
        row is x,y,z positions in cm relative to to the fusion device center
        each row represents a single spectrometer line-of-sight origin point
    
    los_vec: 2D N x 3 array
        row is x,y,z vectors in cm relative to to the fusion device center
        each row represents a single spectrometer line-of-sight vector
        
    los_name [OPTIONAL]: 1D N-size list
        each value represents the name of a spectrometer in corresponding order
        with los_vec and los_pos
        
    only_pi [OPTIONAL-False]: Boolean value (True or False)
        if set to True, will only include the pi polarized light for spectra
        
    dlam: Float or Int
        the resolution in nm of the spectra calculated by pyFIDASIM
        
    lambda_min: Float or Int
        the minimum value in nm for the wavelength range used to store spectra 
        values
        
    lambda_max: Float or Int
        the maximum value in nm for the wavelength range used to store spectra 
        values
    
    
    Extended Emission [OPTIONAL]
    ----------------------------
    spectrum_extended [OPTIONAL-False]: Boolean (True or False)
        enables the user to specify transition lines other than Balmer-Alpha
        to be simulated for the neutral beam emission spectra; if specified,
        the lambda_min and lambda_max become optional
    
    transition [MANDATORY if spectrum_extended is True]: List of two Ints
        the first int is the the max orbital number that will transition down 
        to the second int representing the smaller orbital number. can use max 
        orbital number of 6, 5, 4, and 3, where the second orbital number must 
        be smaller than the max orbital number
        
        
    Impurity Data
    -------------    
    impurities [OPTIONAL-['Carbon']]: List of N Strings
        list of impurities in the plasma from the following options
        ['Argon','Boron','Carbon','Neon','Nitrogen','Oxygen']
    
    path_to_tables [OPTIONAL]: String
        sets a custom path to load tables used for impurity calculations
        but will by default load from existing tables
    
    load_raw_data [OPTIONAL-True]: Boolean (True or False)
        if true, will load raw data from pre-supplied tables, if false, will
        load the data from fidasim_tables.hdf5
        
        
    Grid Information
    ----------------
    grid_drz [OPTIONAL-2.]: Float or Int
        precision of the grid in cm used by pyFIDASIM for storing density,
        calculating intersections, and other relevant information
        
    r_ran [OPTIONAL]: List of two Float values [cm]
        the R (major radius) range over which the grid is taken; if not 
        specified will default to the available range from the equilibria 
        
    z_ran [OPTIONAL]: List of two Float values [cm]
        the Z range over which the grid is taken; if not specified will default
        to the available range from the equilibria 
        
    u_range [OPTIONAL]: List of two Floats or Ints
        this option is for users wishing to store density information along a
        beam-aligned grid; must be defined in order to store beam-aligned
        density values
        
    du [OPTIONAL-1.]: Float
        the steplength in centimeters along the beam-direction to store density
        information (defines the precision in the beam-aligned direction)
        
    dvw [OPTIONAL-1.]: Float
        the steplength in centimeters along the axes prependicular to the beam
        path (defines the precision in the directions perpendicular to the beam
        path)
        
    v_width [OPTIONAL-5.] Float or Int
        the total length of the v-axis over which to store beam-aligned density
        values; should be set large enough to encompass the entire beams extent
        along that axis
        
    v_width [OPTIONAL-5.] Float or Int
        the total length of the w-axis over which to store beam-aligned density
        values; should be set large enough to encompass the entire beams extent
        along that axis
    
        
    Other
    -----
    ion_mass: Float or Int
        the mass number of the primary ion type in the plasma 
        (eg ~1 for protium)
        
    nbi_mass: Float or Int
        the mass number of the primary ion type injected by the neutral beam
        (eg ~1 for protium)

    Fast-Ion Distribution [OPTIONAL]
    --------------------------------
    fbm_file: String
        Filename of the Fast-Ion Distribution file (e.g. .cdf file). 
        If 'FIDASIM_check' is True, it looks in 'directory'. 
        Otherwise, provide full path or relative path.
    
    emin [OPTIONAL-0]: Float
        Minimum energy [keV] to load from distribution.
    
    emax [OPTIONAL-100]: Float
        Maximum energy [keV] to load from distribution.
        
    pmin [OPTIONAL- -1]: Float
        Minimum pitch angle (v_par/v) to load.
        
    pmax [OPTIONAL- 1]: Float
        Maximum pitch angle (v_par/v) to load.
    
    Fortran FIDASIM Files
    For users with Fortran FIDASIM files the following keys should be used
    ---------------------------------------------------------------------------
    FIDASIM_check: Bool
        Indicator to show you plan to use Fortran FIDASIM setup
    
    runid: String
        run id for the desired FIDASIM files (prefix of the input files,
        eg. runid_geometry.h5)
    
    directory: String
        location of Fortran FIDASIM files
    
    geqdsk: String
        file name for the desired geqdsk information which should be stored
        in the location specified by the "directory" key
        
    
    
    TRANSP Files
    For users with TRANSP files the following keys should be used
    ---------------------------------------------------------------------------
    transp [OPTIONAL-False]: Boolean value (True or False)
        if set to true, will use the TRANSP routines which is necessary for
        those with data from TRANSP files
        
    time: Float or Int
        the time point in the TRANSP run to take information from
        
    runid: String
        run id for the desired TRANSP run
        
    directory: String
        location of TRANSP files relative to your code location; the cdf and 
        dat file from TRANSP are necessary for setup
        
    ntheta [OPTIONAL-200]: Int
        the amount of theta divisions used to store the magnetic field, the 
        higher the value the more precise the spatial magnetic field
    
    nr [OPTIONAL-300]: Int
        the amount of r domain divisions used to store the magnetic field, the 
        higher the value the more precise the spatial magnetic field
    
    nz [OPTIONAL-300]: Int
        the amount of z domain divisions used to store the magnetic field, the 
        higher the value the more precise the spatial magnetic field
        
    phi_ran [OPTIONAL]: List of two radian Float values
        the phi range over which the equilibria is taken; if not specified, 
        will apply over the whole 0 to 2pi range
        
    Bt_sign [OPTIONAL-1]: 1 or -1
        the direction of the toroidal magnetic field
        
    Ip_sign [OPTIONAL-1]: 1 or -1
        the direction of the plasma current
    
    prof_plot [OPTIONAL-False]: Boolean (True or False)
        if set to True, will plot the plasma profiles
    
    s_new [OPTIONAL]: List of floats
        the new s-domain over which to define the plasma profiles
    
    n_decay [OPTIONAL-0.1]: Float
        the decay length from the s-domain which defines the rate of the ion 
        and electron density decay in the scrape-off layer (s>1.0)
    
    te_decay [OPTIONAL-0.1]: Float
        the decay length from the s-domain which defines the rate of the 
        electron temperature decay in the scrape-off layer (s>1.0)    
    
    ti_decay [OPTIONAL-0.1]: Float
        the decay length from the s-domain which defines the rate of the ion
        temperature decay in the scrape-off layer (s>1.0)    
    
    omega_decay [OPTIONAL-0.1]: Float
        the decay length from the s-domain which defines the rate of the omega
        decay (decay of rotational speed) in the scrape-off layer (s>1.0)    
    
    nbi_source_sel [OPTIONAL]: list of ints 
        select a sub-set of nbi sources found in TRANSP. Defaults to full list 
        nbi sources contained in TRANSP.
    
    nbi_plot [OPTIONAL-False]: Boolean (True or False)
        plots the neutral beam information as a function of time    
    
    Point Spread Function [OPTIONAL]:
    ---------------------------------------------------------------------------
    calc_PSF [OPTIONAL]: Boolean
        if running PSF calculations, this flag must be True. Following
        parameters are required only if this flag is True
        
    image_pos: List of float lists [x,y,z] [cm]
        central location of each image source location on the 
        collection lens
        
    image_vec: List of float lists [dx,dy,dz] (normalized vectors)
        vectors from the image pos to the focal point on the NBI
    
    los_image_arr: List of ints len(los_pos)
        an array of ints mapping los_index to which image they 
        are grouped onto
    
    d_fiber: Float [cm]
        diameter of the fibers used in cm. Will be multiplied to p_fiber
        
    p_fiber: List of len(2) lists of floats::
        central position of each fiber location in the multi-fiber LOS
        [[fiber1_x,fiber1_y],[fiber2_x,fiber2_y],...] in units of d_fiber
        
    npoint: Int
        number of points used to fill the circular area of each the fiber in 
        p_fiber
        
    n_rand: Int
        number of values used to randomly sample the finite lifetime decay
        
    f_lens: Float [cm]
        collection lens focal length, used to calculate the plasma spot
        size magnification using the thin lens equation.
        
    plot_blur_image [OPTIONAL-False]: Boolean
        if flag is true, will produce a plot of the unmagnified finite fiber
        blurring image kernel for input checking
        
    Device Specific Keys
    Ignore keys not related to the target device
        
    W7-X
    ---------------------------------------------------------------------------
    machine: String
        if running pyFIDASIM for W7-X, this key must be set to "W7X"    
    
    los_shot [OPTIONAL]: String
        the shot ID used to load line of sight data; if not set, will load from
        the default '20180823.035'
    
    los_head [OPTIONAL-'AEA']: String
        subset of line of sights to use in simulation
        (eg 'AEA', 'AE', 'AEM', 'AET')
        
    los_file [OPTIONAL]: String
        if specified, will load line of sight information from the given file
        rather than the pre-supplied file
        
    los_default [OPTIONAL-False]: Boolean
        if set to True, will use fixed default line of sight parameters

    los_new [OPTIONAL-True]: Boolean
        if set to True, will use the new line-of-sight setup
    
    progID [OPTIONAL]: String
        the desired investiagted discharge number, if not specified, will load
        the standard config from discharge '20180823.037'
        
    vmecID [OPTIONAL]: String
        the VMEC ID to load the "wout" file from; if progID is specified, it 
        will override this key
    
    eq_path [OPTIONAL]: String
        the path to the desired "wout" file; if progID or vemcID is specified, 
        they will override this key
        
    extended_vmec_factor [OPTIONAL-1.0]: Float or Int
        this key will extended the vmec file to a larger s-coordinate domain
        
    b0_factor [OPTIONAL-1.0]: Float or Int
        scaling factor for the magnetic field
    
    drz [OPTIONAL-2.0]: Float or Int
        precision of the r and z domains for the magnetic field
        
    phi_ran [OPTIONAL]: List of two radian Float values
        the phi range over which the equilibria is taken; if not specified, 
        will apply over the whole 0 to 2pi range
        
    prof_path [OPTIONAL]: String
        the path to an hdf5 to load profile information; if not specified, will
        load from 'Data/W7Xprofiles.h5'
    
    shot_num [OPTIONAL]: String
        the shot number for nbi information; by default will use '20180920.042'
    
    t_start [OPTIONAL-6.5]: Float
        start time of the NBI operation window
    
    t_stop [OPTIONAL-6.52]: Float
        end time of NBI operation window
        
    nbi_cur_frac [OPTIONAL]: List of Floats
        manually set neutral beam current fractions; if not set, will read the
        beam fractions using the shot number
    
    nbi_debug [OPTIONAL-False]: Boolean (True or False)
        if set to True, will plot NBI information for debugging
        
    nbi_default [OPTIONAL-False]: Boolean (True or False)
        if set to True, will use fixed default NBI parameters

    
    Skip Option
    -----------
    skip [OPTIONAL]: List of N-Strings
        for advanced users, can specify to skip the creation process for some
        of the return values (these values will return as None); need to 
        specify from the following list ['spec','tables','fields','profiles',
                                         'nbi','grid3d','PSF','fbm']
        if fields is skipped, grid3d will be skipped as well

    RETURNS
    ---------------------------------------------------------------------------
    returns: sim_settings, spec, tables, fields, profiles, nbi, grid3d, ncdf, PSF, fbm
    '''

    # Initialize variables to None
    spec = tables = fields = profiles = nbi = grid3d = PSF = fbm = ncdf = None
    
    if FIDASIM.get('FIDASIM_check', False):
        directory = variable_check(FIDASIM, 'directory', str)
        geqdsk_path = variable_check(FIDASIM, 'geqdsk', str)
        runid = variable_check(FIDASIM, 'runid', str)
        
        fbm_file = variable_check(FIDASIM, 'fbm_file', str, optional=True, default=None)
        emin = variable_check(FIDASIM, 'emin', (float, int), optional=True, default=0)
        emax = variable_check(FIDASIM, 'emax', (float, int), optional=True, default=100)
        pmin = variable_check(FIDASIM, 'pmin', (float, int), optional=True, default=-1)
        pmax = variable_check(FIDASIM, 'pmax', (float, int), optional=True, default=1)

        from .F90_to_py_loader import f90_to_py
        nbi, profiles, spec, fields, fbm = f90_to_py(
            path=directory, 
            runid=runid, 
            geqdsk=geqdsk_path, 
            fbm_file=fbm_file,
            emin=emin, emax=emax, pmin=pmin, pmax=pmax
        )
        
        phi_ran = variable_check(FIDASIM, "phi_ran", list, optional=True, default=[0.0, 2 * np.pi])
        if len(phi_ran) != 2 or not all(isinstance(x, (float, int)) for x in phi_ran):
            raise ValueError("The key 'phi_ran' must be a list of two floats/ints")
        r_ran = variable_check(FIDASIM, "r_ran", list, optional=True, default=None)
        if r_ran:
            if len(r_ran) != 2 or not all(isinstance(x, (float, int)) for x in r_ran):
                raise ValueError("The key 'r_ran' must be a list of two floats/ints")
        z_ran = variable_check(FIDASIM, "z_ran", list, optional=True, default=None)
        if z_ran:
            if len(z_ran) != 2 or not all(isinstance(x, (float, int)) for x in z_ran):
                raise ValueError("The key 'z_ran' must be a list of two floats/ints")
        fields['phimin'] = np.float64(phi_ran[0])
        fields['phimax'] = np.float64(phi_ran[1])
        fields['phi'] = np.array([np.float64((phi_ran[0] + phi_ran[1]) / 2.)])
        
        grid3d = start_grid3d(FIDASIM, fields)
        
        from .los import grid_intersections
        spec = grid_intersections(spec, fields, grid3d)
        
        tables = start_tables(FIDASIM)
        
    else:
        # Handle 'skip' option
        skip_list = FIDASIM.get('skip', [])
    
        # Process 'spec' if not skipped
        if 'spec' not in skip_list:
            if FIDASIM.get("machine") == "W7X":
                spec = start_spec_W7X(FIDASIM)
            else:
                spec = start_spec(FIDASIM)
            if FIDASIM.get("calc_extended_emission", False):
                spec, ncdf = extended_emission(FIDASIM, spec)
    
        # Process 'tables' if not skipped
        if 'tables' not in skip_list:
            tables = start_tables(FIDASIM)
    
        # Process 'fields' if not skipped
        if 'fields' not in skip_list:
            if FIDASIM.get('transp', False):
                transp = transp_check(FIDASIM)
                fields = start_fields_transp(FIDASIM, transp)
            else:
                if "machine" not in FIDASIM:
                    raise KeyError("The key 'machine' or 'transp' must be defined")
                fields = start_fields_machine(FIDASIM)
    
        # Process 'profiles' if not skipped
        if 'profiles' not in skip_list:
            if FIDASIM.get('transp', False):
                transp = transp_check(FIDASIM)
                profiles = start_profiles_transp(FIDASIM, transp)
            else:
                if "machine" not in FIDASIM:
                    raise KeyError("The key 'machine' or 'transp' must be defined")
                profiles = start_profiles_machine(FIDASIM)
    
        # Process 'nbi' if not skipped
        if 'nbi' not in skip_list:
            if FIDASIM.get('transp', False):
                transp = transp_check(FIDASIM)
                nbi = start_nbi_transp(FIDASIM, transp)
            else:
                if "machine" not in FIDASIM:
                    raise KeyError("The key 'machine' or 'transp' must be defined")
                nbi = start_nbi_machine(FIDASIM)
    
        # Process 'grid3d' if not skipped
        if 'grid3d' not in skip_list and fields is not None:
            grid3d = start_grid3d(FIDASIM, fields)
    
        # Process 'spec' further if able
        if spec is not None and grid3d is not None and fields is not None:
            from .los import grid_intersections
            spec = grid_intersections(spec, fields, grid3d)
        
        if 'PSF' not in skip_list:
            PSF = start_PSF(FIDASIM)

        if 'fbm' not in skip_list and 'fbm_file' in FIDASIM and fields is not None:
            fbm = start_fbm(FIDASIM, fields)

    # Initialize simulation settings
    sim_settings = start_sim_settings(FIDASIM)

    # Clean up dictionaries by removing unused keys
    if spec is not None:
        spec = cleanup_spec(spec)
    if profiles is not None:
        profiles = cleanup_profiles(profiles)
    if tables is not None:
        tables = cleanup_tables(tables)
    if fields is not None:
        fields = cleanup_fields(fields)
    if grid3d is not None:
        grid3d = cleanup_grid3d(grid3d)

    # Massage inputs if able
    if nbi is not None and spec is not None and fields is not None:
        sim_settings, spec, fields, nbi, fbm, ncdf, PSF = _input_massaging(
            sim_settings, spec, fields, nbi, ncdf, fbm, PSF
        )

    return sim_settings, spec, tables, fields, profiles, nbi, grid3d, ncdf, PSF, fbm

def variable_check(FIDASIM, key, expected_types, optional=False, default=None):
    """
    Checks if a key exists in the FIDASIM dictionary, and if its value is of the expected type(s).
    Returns the value if present and correct type, otherwise raises an exception.
    If optional is True and key is not present, returns default value.
    """
    if key in FIDASIM:
        value = FIDASIM[key]
        if not isinstance(value, expected_types):
            raise TypeError(f"The key '{key}' must be of type(s) {expected_types}, got {type(value)}")
        return value
    elif optional:
        return default
    else:
        raise KeyError(f"The key '{key}' is missing from the dictionary")

def start_PSF(FIDASIM):
    from .PSF import blur_image
    calc_PSF = variable_check(FIDASIM, "calc_PSF", bool, optional = True, default = False)
    if calc_PSF:
        image_pos = variable_check(FIDASIM, 'image_pos', (list,np.ndarray))
        if not all(len(pos)==3 for pos in image_pos) or not all(isinstance(val, (np.int32,int,np.float32,np.float64,float)) for pos in image_pos for val in pos):
            raise ValueError("Values in image_pos must be lists of three int/floats")
            
        image_vec = variable_check(FIDASIM, 'image_vec', (list,np.ndarray))
        if not all(len(pos)==3 for pos in image_vec) or not all(isinstance(val, (np.int32,int,np.float32,np.float64,float)) for pos in image_vec for val in pos):
            raise ValueError("Values in image_vec must be lists of three int/floats")
            
        los_image_arr = variable_check(FIDASIM, 'los_image_arr', (list,np.ndarray))
        los_pos = variable_check(FIDASIM, 'los_pos', (list,np.ndarray))
        if not len(los_image_arr)==len(los_pos[:,0]) or not all(isinstance(im_index, (np.int32,int)) for im_index in los_image_arr):
            raise ValueError("los_image_arr must be an array of ints with len(los_pos[:,0])")
        
        d_fiber = variable_check(FIDASIM, "d_fiber", (float,int))
        p_fiber = variable_check(FIDASIM, "p_fiber", (list,np.ndarray))
        if not all(len(pos)==2 for pos in p_fiber) or not all(isinstance(val, (np.int32,int,np.float32,np.float64,float)) for pos in p_fiber for val in pos):
            raise ValueError("Values in the p_fiber must be a lists of two int/floats")
            
        n_point = variable_check(FIDASIM, "n_point", int)
        n_rand = variable_check(FIDASIM, "n_rand", int)
        f_lens = variable_check(FIDASIM, "f_lens", (np.int32,int,np.float32,np.float64,float))
        PSF_dict = {}
        if type(p_fiber) == list:
            p_fiber = np.array(p_fiber)
        p_fiber = np.hstack([p_fiber, np.zeros((p_fiber.shape[0], 1))])*d_fiber
        PSF_dict['image_blur'] = blur_image(n_point,p_fiber,d_fiber,image_vec)
        PSF_dict['image_pos'] = image_pos
        PSF_dict['image_vec'] = image_vec
        PSF_dict['n_rand'] = n_rand
        PSF_dict['los_image_arr'] = los_image_arr
        PSF_dict['f_lens'] = f_lens
    else:
        PSF_dict = {}
        PSF_dict['image_blur'] = 0
        PSF_dict['image_pos'] = 1
        PSF_dict['image_vec'] = 1
        PSF_dict['n_rand'] = 0
        PSF_dict['los_image_arr'] = 0
        PSF_dict['f_lens'] = 1
        
    return PSF_dict
    
def extended_emission(FIDASIM, spec):
    nbi_mass = variable_check(FIDASIM, "nbi_mass", (float, int))
    if nbi_mass == 1.0:
        beam_type = "H"
    elif nbi_mass == 2.0:
        beam_type = "D"
    else:
        raise ValueError("Extended emission only works for H or D beam (nbi_mass = 1.0 or 2.0)")
    spectrum_extended = variable_check(FIDASIM,'spectrum_extended', bool, default=False)
    transition = variable_check(FIDASIM, "transition", list)
    if len(transition) != 2 or not all(isinstance(x, int) for x in transition):
        raise ValueError("The key 'transition' must be a list of two integers")
    if transition[0] < transition[1]:
        raise ValueError("The maximum orbital level should not be smaller than the minimum orbital level")

    filename = f"{beam_type}_n{transition[0]}_{transition[1]}.ncdf"
    from .toolbox import read_netCDF_stark
    import os
    from pathlib import Path
    import pyfidasim.tables
    tables_path = Path(pyfidasim.tables.__file__).parent
    file_path = os.path.join(tables_path, 'orbital_ncdf_data', filename)
    cdfvars, lambda0, trans, l_to_dwp = read_netCDF_stark(file_path)
    ncdf = {
        "cdfvars": cdfvars,
        "lambda0": lambda0,
        "trans": trans,
        "l_to_dwp": l_to_dwp,
        "spectrum_extended": spectrum_extended
    }
    spec["lambda_min"] = FIDASIM.get('lambda_min', lambda0 - 5)
    spec["lambda_max"] = FIDASIM.get('lambda_max', lambda0 + 5)
    spec['dlam'] = variable_check(FIDASIM, 'dlam', (float, int))
    spec['wavel'] = np.arange(spec['lambda_min'], spec['lambda_max'], spec['dlam'])
    spec['nlam'] = len(spec['wavel'])
    return spec, ncdf

def transp_check(FIDASIM):
    if 'machine' in FIDASIM:
        raise ValueError("The 'transp' key must be set to False when using the 'machine' key")
    time = variable_check(FIDASIM, 'time', (float, int))
    runid = variable_check(FIDASIM, 'runid', str)
    directory = variable_check(FIDASIM, 'directory', str)
    return {'time': time, 'runid': runid, 'directory': directory}

def start_spec(FIDASIM):
    spec = {}
    spec['los_pos'] = np.asarray(variable_check(FIDASIM, 'los_pos', (list, np.ndarray)))
    spec['los_vec'] = np.asarray(variable_check(FIDASIM, 'los_vec', (list, np.ndarray)))

    if spec['los_pos'].ndim != 2 or spec['los_pos'].shape[1] != 3:
        raise ValueError("The key 'los_pos' must be a 2D array with shape (N, 3)")
    if spec['los_vec'].ndim != 2 or spec['los_vec'].shape[1] != 3:
        raise ValueError("The key 'los_vec' must be a 2D array with shape (N, 3)")
    if spec['los_pos'].shape[0] != spec['los_vec'].shape[0]:
        raise ValueError("The number of lines of sight in 'los_pos' and 'los_vec' must match")

    spec['losname'] = variable_check(
        FIDASIM,
        'los_name',
        list,
        optional=True,
        default=[f"LOS {i}" for i in range(spec['los_pos'].shape[0])]
    )
    if len(spec['losname']) != spec['los_pos'].shape[0]:
        raise ValueError("The length of 'los_name' must match the number of lines of sight")

    spec['only_pi'] = variable_check(FIDASIM, 'only_pi', bool, optional=True, default=False)

    if not FIDASIM.get('calc_extended_emission', False):
        spec['dlam'] = variable_check(FIDASIM, 'dlam', (float, int), optional=True, default = 0.01)
        spec['lambda_min'] = variable_check(FIDASIM, 'lambda_min', (float, int), optional=True, default = 650.0)
        spec['lambda_max'] = variable_check(FIDASIM, 'lambda_max', (float, int), optional=True, default = 662.0)
        spec['wavel'] = np.arange(spec['lambda_min'], spec['lambda_max'], spec['dlam'])
        spec['nlam'] = len(spec['wavel'])

    spec['nlos'] = spec['los_pos'].shape[0]
    return spec

def start_spec_W7X(FIDASIM):
    shot = variable_check(FIDASIM, "los_shot", str, optional=True, default='20180823.035')
    head = variable_check(FIDASIM, "los_head", str, optional=True, default='AEA')
    file = variable_check(FIDASIM, "los_file", str, optional=True, default='')
    default = variable_check(FIDASIM, "los_default", bool, optional=True, default=False)
    new = variable_check(FIDASIM, "los_new", bool, optional=True, default=True)

    from .input_preparation.W7X.los_geometry import los_geometry
    spec = los_geometry(shot=shot, head=head, file=file, default=default, new=new)

    if not FIDASIM.get('calc_extended_emission', False):
        spec['dlam'] = variable_check(FIDASIM, 'dlam', (float, int))
        spec['lambda_min'] = variable_check(FIDASIM, 'lambda_min', (float, int))
        spec['lambda_max'] = variable_check(FIDASIM, 'lambda_max', (float, int))
        spec['wavel'] = np.arange(spec['lambda_min'], spec['lambda_max'], spec['dlam'])
        spec['nlam'] = len(spec['wavel'])

    return spec

def start_tables(FIDASIM):
    impurities = variable_check(FIDASIM, 'impurities', list, optional=True, default=['Carbon'])
    allowed_impurities = ['Argon', 'Boron', 'Carbon', 'Neon', 'Nitrogen', 'Oxygen']
    if not all(imp in allowed_impurities for imp in impurities):
        raise ValueError(f"Invalid impurity in 'impurities'; allowed values are {allowed_impurities}")

    path_to_tables = variable_check(FIDASIM, "path_to_tables", str, optional=True, default=None)

    from .cr_model import load_tables
    tables = load_tables(path_to_tables)
    return tables

def start_fields_transp(FIDASIM, transp):
    ntheta = variable_check(FIDASIM, "ntheta", int, optional=True, default=200)
    drz = variable_check(FIDASIM, "drz", (float, int), optional=True, default=1.0)
    phi_ran = variable_check(FIDASIM, "phi_ran", list, optional=True, default=[0.0, 2 * np.pi])
    if len(phi_ran) != 2 or not all(isinstance(x, (float, int)) for x in phi_ran):
        raise ValueError("The key 'phi_ran' must be a list of two floats/ints")

    Bt_sign = variable_check(FIDASIM, "Bt_sign", int, optional=True, default=1)
    if Bt_sign not in (-1, 1):
        raise ValueError("The key 'Bt_sign' must be either 1 or -1")

    Ip_sign = variable_check(FIDASIM, "Ip_sign", int, optional=True, default=1)
    if Ip_sign not in (-1, 1):
        raise ValueError("The key 'Ip_sign' must be either 1 or -1")

    from .TRANSP.fields import generate_s_grid_transp
    fields = generate_s_grid_transp(
        transp,
        ntheta=ntheta,
        drz=drz,
        phi_ran=phi_ran,
        Bt_sign=Bt_sign,
        Ip_sign=Ip_sign
    )
    return fields

def start_fields_machine(FIDASIM):
    machine = FIDASIM['machine']
    if machine == 'W7X':
        progID = variable_check(FIDASIM, "progID", str, optional=True, default='')
        vmecID = variable_check(FIDASIM, "vmecID", str, optional=True, default='')
        woutpath = variable_check(FIDASIM, "eq_path", str, optional=True, default='')
        extended_vmec_factor = variable_check(
            FIDASIM,
            "extended_vmec_factor",
            (float, int),
            optional=True,
            default=1
        )
        b0_factor = variable_check(FIDASIM, "b0_factor", (float, int), optional=True, default=1.0)
        drz = variable_check(FIDASIM, "drz", (float, int), optional=True, default=2)
        phi_ran = variable_check(FIDASIM, "phi_ran", list, optional=True, default=None)
        if len(phi_ran) != 2 or not all(isinstance(x, (float, int)) for x in phi_ran):
            raise ValueError("The key 'phi_ran' must be a list of two floats/ints")

        from .input_preparation.W7X.equilibrium import equilibrium
        fields = equilibrium(
            progID=progID,
            woutpath=woutpath,
            vmecID=vmecID,
            extended_vmec_factor=extended_vmec_factor,
            b0_factor=b0_factor,
            drz=drz,
            phi_ran=phi_ran
        )
    else:
        raise ValueError("Unsupported machine specified")
    return fields

def start_profiles_transp(FIDASIM, transp):
    doplt = variable_check(FIDASIM, "prof_plot", bool, optional=True, default=False)
    s_new = variable_check(FIDASIM, "s_new", np.ndarray, optional=True, default=[])
    ne_decay = variable_check(FIDASIM, "n_decay", (float, int), optional=True, default=0.1)
    te_decay = variable_check(FIDASIM, "te_decay", (float, int), optional=True, default=0.1)
    ti_decay = variable_check(FIDASIM, "ti_decay", (float, int), optional=True, default=0.1)
    omega_decay = variable_check(FIDASIM, "omega_decay", (float, int), optional=True, default=0.1)
    ion_mass = variable_check(FIDASIM, "ion_mass", (float, int))

    from .TRANSP.profiles import read_transp_profiles
    profiles = read_transp_profiles(
        transp,
        doplt=doplt,
        s_new=s_new,
        ne_decay=ne_decay,
        te_decay=te_decay,
        ti_decay=ti_decay,
        omega_decay=omega_decay
    )
    profiles['ai'] = ion_mass
    return profiles

def start_profiles_machine(FIDASIM):
    machine = FIDASIM['machine']
    if machine == 'W7X':
        file = variable_check(FIDASIM, "prof_path", str, optional=True, default='Data/W7Xprofiles.pkl')
        impurities = variable_check(FIDASIM, 'impurities', list, optional=True, default=['Carbon'])
        allowed_impurities = ['Argon', 'Boron', 'Carbon', 'Neon', 'Nitrogen', 'Oxygen']
        if not all(imp in allowed_impurities for imp in impurities):
            raise ValueError(f"Invalid impurity in 'impurities'; allowed values are {allowed_impurities}")

        zimp_dict = {'Argon': 18, 'Boron': 5, 'Carbon': 6, 'Neon': 10, 'Nitrogen': 7, 'Oxygen': 8}
        zimps = [zimp_dict[imp] for imp in impurities]

        from .input_preparation.W7X.plasma_profiles_from_pkl import plasma_profiles_from_pkl
        profiles = plasma_profiles_from_pkl(file=file, impurities=impurities, zimps=zimps)
    else:
        raise ValueError("Unsupported machine specified")
    return profiles

def start_nbi_transp(FIDASIM, transp):
    nbi_source_sel = variable_check(FIDASIM, "nbi_source_sel", list, default = False)
    if not all(isinstance(source, int) for source in nbi_source_sel):
        raise ValueError("Values in nbi_source_sel must be integers")

    doplt = variable_check(FIDASIM, "nbi_plot", bool, optional=True, default=False)
    nbi_mass = variable_check(FIDASIM, "nbi_mass", (float, int))

    from .TRANSP.nbi import transp_nbi
    nbi = transp_nbi(transp, doplt=doplt)
    nbi['ab'] = nbi_mass
    if nbi_source_sel:
        if not all(str(source) in nbi for source in nbi_source_sel):
            raise ValueError("Selected source not found in TRANSP nbi file")  
        nbi = {k: v for k, v in nbi.items() if not k.isdigit() or int(k) in nbi_source_sel}
        nbi['sources'] = [str(i) for i in nbi_source_sel]
    return nbi

def start_nbi_machine(FIDASIM):
    machine = FIDASIM['machine']
    if machine == 'W7X':
        shot_number = variable_check(FIDASIM, "shot_num", str, optional=True, default='20180920.042')
        t_start = variable_check(FIDASIM, "t_start", (float, int), optional=True, default=6.5)
        t_stop = variable_check(FIDASIM, "t_stop", (float, int), optional=True, default=6.52)
        fractions = variable_check(FIDASIM, "nbi_cur_frac", list, optional=True, default=None)
        if fractions is not None:
            if not all(isinstance(x, float) for x in fractions):
                raise ValueError("The key 'nbi_cur_frac' must be a list of floats")
        debug = variable_check(FIDASIM, "nbi_debug", bool, optional=True, default=False)
        default = variable_check(FIDASIM, "nbi_default", bool, optional=True, default=False)
        nbi_mass = variable_check(FIDASIM, "nbi_mass", (float, int))

        from .input_preparation.W7X.nbi import W7X_nbi
        nbi = W7X_nbi(
            shot_number=shot_number,
            t_start=t_start,
            t_stop=t_stop,
            fractions=fractions,
            debug=debug,
            default=default
        )
        nbi['ab'] = nbi_mass
    else:
        raise ValueError("Unsupported machine specified")
    return nbi

def start_grid3d(FIDASIM, fields):
    drz = variable_check(FIDASIM, "grid_drz", (float, int), optional=True, default=2.0)
    u_range = variable_check(FIDASIM, "u_range", list, optional=True, default=None)
    if u_range is not None:
        if len(u_range) != 2 or not all(isinstance(x, (float, int)) for x in u_range):
            raise ValueError("The key 'u_range' must be a list of two floats/ints")
    du = variable_check(FIDASIM, "du", (float, int), optional=True, default=1.0)
    dvw = variable_check(FIDASIM, "dvw", (float, int), optional=True, default=1.0)
    v_width = variable_check(FIDASIM, "v_width", (float, int), optional=True, default=5)
    w_width = variable_check(FIDASIM, "w_width", (float, int), optional=True, default=5)
    r_ran = variable_check(FIDASIM, "r_ran", list, optional=True, default=None)
    if r_ran is not None:
        if len(r_ran) != 2 or not all(isinstance(x, (float, int)) for x in r_ran):
            raise ValueError("The key 'r_ran' must be a list of two floats/ints")
    z_ran = variable_check(FIDASIM, "z_ran", list, optional=True, default=None)
    if z_ran is not None:
        if len(z_ran) != 2 or not all(isinstance(x, (float, int)) for x in z_ran):
            raise ValueError("The key 'z_ran' must be a list of two floats/ints")
    phi_ran = variable_check(FIDASIM, "phi_ran", list, optional=True, default=None)
    if phi_ran is not None:
        if len(phi_ran) != 2 or not all(isinstance(x, (float, int)) for x in phi_ran):
            raise ValueError("The key 'phi_ran' must be a list of two floats/ints")

    from .grid3d import define_grid3d
    grid3d = define_grid3d(
        fields,
        drz=drz,
        u_range=u_range,
        du=du,
        dvw=dvw,
        v_width=v_width,
        w_width=w_width,
        r_ran = r_ran,
        z_ran = z_ran,
        phi_ran = phi_ran
    )
    return grid3d

def start_fbm(FIDASIM, fields):
    fbm_file = variable_check(FIDASIM, 'fbm_file', str)
    emin = variable_check(FIDASIM, 'emin', (float, int), optional=True, default=0)
    emax = variable_check(FIDASIM, 'emax', (float, int), optional=True, default=100)
    pmin = variable_check(FIDASIM, 'pmin', (float, int), optional=True, default=-1)
    pmax = variable_check(FIDASIM, 'pmax', (float, int), optional=True, default=1)
    
    if FIDASIM.get('FIDASIM_check', False) and 'directory' in FIDASIM:
        import os
        if os.path.dirname(fbm_file) == '':
            path = FIDASIM['directory']
            full_path = os.path.join(path, fbm_file)
        else:
            full_path = fbm_file
    else:
        full_path = fbm_file

    from .TRANSP.transp_fbeam import transp_fbeam
    print(f"Loading Fast-Ion Distribution from: {full_path}")
    fbm = transp_fbeam(full_path, fields, emin=emin, emax=emax, pmin=pmin, pmax=pmax)
    return fbm

def start_sim_settings(FIDASIM):
    sim_settings = {}
    sim_settings['nmarker'] = variable_check(FIDASIM, "nmarker", int, optional=True, default=10000)
    sim_settings['calc_halo'] = variable_check(FIDASIM, "calc_halo", bool, optional=True, default=False)
    sim_settings['verbose'] = variable_check(FIDASIM, "verbose", bool, optional=True, default=True)
    sim_settings['calc_spectra'] = variable_check(FIDASIM, "calc_spectra", bool, optional=True, default=True)
    sim_settings['calc_photon_origin'] = variable_check(FIDASIM, "calc_photon_origin", bool, optional=True, default=False)
    sim_settings['calc_photon_origin_type'] = variable_check(FIDASIM, "calc_photon_origin_type", bool, optional=True, default=True)
    sim_settings['calc_PSF'] = variable_check(FIDASIM, "calc_PSF", bool, optional=True, default=False)
    sim_settings['calc_rzp_dens'] = variable_check(FIDASIM, "calc_density", bool, optional=True, default=True)
    sim_settings['separate_dcx'] = variable_check(FIDASIM, "separate_dcx", bool, optional=True, default=False)
    sim_settings['calc_uvw_dens'] = variable_check(FIDASIM, "calc_uvw", bool, optional=True, default=False)
    sim_settings['calc_extended_emission'] = variable_check(FIDASIM, "calc_extended_emission", bool, optional=True, default=False)
    sim_settings['respawn_if_aperture_is_hit'] = variable_check(FIDASIM, "respawn_if_aperture_is_hit",
        bool, optional=True, default=True)
    sim_settings['seed'] = variable_check(FIDASIM, "seed", int, optional=True, default=-1)
    sim_settings['batch_marker'] = variable_check(FIDASIM, "batch_marker", int, optional=True, default=10000)
    return sim_settings

def _input_massaging(sim_settings, spec, fields, nbi, ncdf=None, fbm=None, PSF=None):
    if fbm is None:
        fbm = {
            'fbm': np.zeros((1, 1, 1, 1)),
            'denf': np.zeros((1, 1, 1)),
            'afbm': 0,
            'btipsign': 0,
            'emin': 0.0,
            'eran': 0.0,
            'nenergy': 0,
            'energy': np.zeros(1),
            'dE': 0.0,
            'pmin': 0.0,
            'pran': 0.0,
            'npitch': 0,
            'pitch': np.zeros(1),
            'dP': 0.0
        }

    if ncdf is None:
        ncdf = {
            'cdfvars': np.zeros((1, 1, 1)),
            'trans': 1,
            'l_to_dwp': 1,
            'spectrum_extended': False,
        }
        if nbi['ab'] > 1.1:
            ncdf['lambda0'] = 656.10  # D-alpha emission
        else:
            ncdf['lambda0'] = 656.281  # H-alpha emission

    fields.setdefault('Er', np.zeros((1, 1, 1)))
    fields.setdefault('flux_dvol', 1)

    if spec:
        spec.setdefault('output_individual_stark_lines', False)
        spec.setdefault('sigma_to_pi_ratio', np.ones(spec.get('nlos', 1)))
        spec.setdefault(
            'los_grid_intersection_weight',
            np.ones_like(spec.get("grid_cell_crossed_by_los", np.zeros((1, 1, 1))))
        )
        spec.setdefault('los_f_len', np.array([0]))
        spec.setdefault('nlos', spec.get('nlos', 0))
        spec.setdefault('nlam', spec.get('nlam', 0))
    else:
        spec = {
            'grid_cell_crossed_by_los': np.zeros((1, 1, 1)),
            'dl_per_grid_intersection': np.zeros((1, 1)),
            'los_grid_intersection_indices': np.zeros((1, 1, 1)),
            'nlos': 0,
            'los_pos': np.zeros((1, 1)),
            'los_vec': np.zeros((1, 1)),
            'lambda_min': 0.0,
            'nlam': 0,
            'dlam': 1.0,
            'sigma_to_pi_ratio': np.zeros(1),
            'calc_stark_spitting': False,
            'output_individual_stark_lines': False,
            'los_grid_intersection_weight': np.ones((1, 1, 1)),
            'calc_stark_spitting': False
        }

    if PSF is None:
        PSF = {
            "image_pos": 1,
            "image_vec": 1,
            "image_blur": 0,
            "los_image_arr": 0,
            "n_rand": 0,
            "f_lens": 1
        }

    return sim_settings, spec, fields, nbi, fbm, ncdf, PSF

# Define cleanup functions for dictionaries
def cleanup_spec(spec):
    keys_to_keep = {
        'grid_cell_crossed_by_los', 'dl_per_grid_intersection',
        'los_grid_intersection_indices', 'los_grid_intersection_weight',
        'nlos', 'los_pos', 'los_vec', 'lambda_min', 'nlam', 'dlam',
        'sigma_to_pi_ratio', 'output_individual_stark_lines', 'lambda_max',
        'losname', 'wavel', 's_per_grid_intersection', 'full_los_length'
    }
    return {k: spec[k] for k in keys_to_keep if k in spec}

def cleanup_profiles(profiles):
    keys_to_keep = {'s', 'te', 'ti', 'dene', 'denimp', 'omega', 'ai', 'denp'}
    return {k: profiles[k] for k in keys_to_keep if k in profiles}

def cleanup_tables(tables):
    keys_to_keep = {
        'energy_ax', 'temp_ax', 'impurities', 'qptable_no_cx', 'qptable',
        'qetable', 'qitables', 'einstein', 'neutrates', 'levels', 'zimps'
    }
    cleaned_tables = {k: tables[k] for k in keys_to_keep if k in tables}
    cleaned_tables['nimps'] = len(cleaned_tables.get('impurities', []))
    return cleaned_tables

def cleanup_fields(fields):
    keys_to_keep = {
        'rotate_phi_grid', 'dphi_sym', 's', 'Rmin', 'Rmax', 'R', 'dR', 'nr',
        'Zmin', 'Zmax', 'Z', 'dZ', 'nz', 'phimin', 'phimax', 'phi', 'dphi',
        'nphi', 'flux_ns', 'flux_ds', 'flux_s_min', 'Br', 'Bz', 'Bphi',
        'Rsurf', 'Zsurf', 's_surf', 'flux_nr', 'flux_dvol', 'nsym', 'Rmean',
        'btipsign'
    }
    return {k: fields[k] for k in keys_to_keep if k in fields}

def cleanup_grid3d(grid3d):
    keys_to_keep = {
        'rotate_phi_grid', 'dphi_sym', 'dvol', 'Rmin', 'nR', 'dR', 'Zmin',
        'nZ', 'dZ', 'phimin', 'dphi', 'nphi', 'umin', 'du', 'vmin', 'dv',
        'wmin', 'dw', 'nu', 'nv', 'nw', 'wmax', 'wc', 'umax', 'vc', 'vmax',
        'uc', 'Z_c', 'Zmax', 'R_c', 'Rmax', 'phimax', 'phi_c'
    }
    return {k: grid3d[k] for k in keys_to_keep if k in grid3d}
