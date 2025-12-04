# pyFIDASIM

pyFIDASIM is a Monte Carlo code to simulate the neutral beam attenuation and emission in magnetically confined plasma.

## Installation
Since pyfidasim is currently not on "pip" or any other equivalent package distributor, there are several different installation methods.

1) If using Anaconda, add the repository directory (the directory containing this `README.md` file) to Anaconda using conda develop in an Anaconda console:
```bash
conda develop path_to_pyfidasim
```

2) Or, add the repository directory (the directory containing this `README.md` file) to the `PYTHONPATH` environment variable.  In Bash:
```bash
export PYTHONPATH=<repository directory>:$PYTHONPATH
```
Consider adding the export command to your `.bashrc` file, a shell script, or a personal `MODULEFILE`.

A start file can then be created by passing a dictionary with the necessary settings into the "input_prep" function from "input_prep.py". A simulated example can be found in the "Example" folder inside a file called "run_test.py" but requires pyQt6 to run. Real examples can be found in the "Example" folder inside a file called "FeautureShowoff.py". The necessary inputs are detailed below:

```
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

omega_decay [OPTIONAL-0.1]: Float  ###NEEDS BETTER DEFINITION (omega?)
    the decay length from the s-domain which defines the rate of the omega
    decay in the scrape-off layer (s>1.0)    

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
```

## Contributions

To contribute to pyFIDASIM, create a new branch and push the branch to the repository:
```bash
git checkout -b my_new_branch
git push -u origin my_new_branch
```

To keep your new branch up-to-date, you should routinely do `git pull origin master` and resolve any merge conflicts.  To merge your new branch into `master`, submit a merge request