#----------------------------------------------------------------------------------
#- routines to read NBI parameter information for a given shot and temporal range -
#----------------------------------------------------------------------------------

# General Note:
#   While the routine itself works from any directory, saving the data only works
#   when called from the 'examples/W7X' folder where the typical start file is located
#   For bug report and necessary fixes/alterations: thir@ipp.mpg.de

# Input:
#   shot_number: shot number in W7-X standard format YYYYMMDD.XXX
#   t_start:     starting time of NBI operation window
#   t_stop:      end time of NBI operation window
#   fractions:   variable to, if demanded, pass fractions which are then used instead of the ones written to the file
#
#   General structure is suited to assesses all NBI operation in a single shot
#   by passing t_start and t_stop as arrays
#
# example input nbi_parameters('20180920.042', t_start = 6.5, t_stop = 6.52, debug = True)

# OUTPUT:
#   beam_information:  dictionary suited as a pyFIDASIM input for the beam parameters - source resolved
#                      also save to the 'examples/W7X/Data/' folder

# Import general packages
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

# import the routines for loading & saving hdf5 files
import pyfidasim
from pyfidasim.toolbox import save_dict, load_dict


###############################################################################
# routine to read the NBI parameters from the W7-X data base & a lookup table #
###############################################################################
def nbi_parameters(shot_number = '20180920.042', t_start = 6.5, t_stop = 6.52, fractions = None, debug = False,default=False):
     
    if default:
        import copy
        nbi_params = {}
        nbi_params['sources'] = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
        
        params = {}
        params['voltage'] = 54.6  # keV
        params['power'] = 1.5  # MW
        params['fraction'] = [0.3, 0.5, 0.2]
        params['type'] = 'H'
    
        for source in nbi_params['sources']:
            nbi_params[source] = copy.deepcopy(params)
        return nbi_params

    
    # import 'custom' W7-X routines
    from w7xdia.nbi     import get_nspec_currentFractions
    # THE FOLLOWING IMPORTS NEED TO BE EXPANDED WITH ADDITIONAL SOURCES BECOMING AVAILABLE IN THE FUTURE
    from w7xdia.nbi import get_source_7_power,   get_source_8_power
    from w7xdia.nbi import get_source_7_voltage, get_source_8_voltage

    # put together the variable shot information - checks whether passed times are an array or single floats
    # float case    
    try:
        len(t_start)
        # turn single floats into arrays
        shot_information = [shot_number, t_start, t_stop]
    # array case
    except:
        # just put the variables together without altering them
        shot_information = [shot_number, [t_start], [t_stop]]
    
    # set a time out value for the database 
    timeout = 10

    # set up a dictionary with the routines - NEEDS TO BE EXPANDED WITH NEW SOURCES
    routines = {}
    # sub-levels for the voltage, energy and beam fractions
    routines['voltage']       = {}
    routines['voltage']['Q7'] = get_source_7_voltage
    routines['voltage']['Q8'] = get_source_8_voltage
    routines['power']         = {}
    routines['power'] ['Q7']  = get_source_7_power
    routines['power'] ['Q8']  = get_source_8_power
    routines['fractions']     = get_nspec_currentFractions

    # set up a dictionary to contain general source informations as well as the information which is later saved time resolved
    source_information                 = {}
    source_information['sources']      = ['Q7','Q8']#['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
    source_information['source_index'] = [6,7]      #[0,1,2,3,4,5,6,7]

    # array of all the active sources and the respective indices
    sources      = []
    source_index = []
    # threshold power value to determine whether a source is No documentation available on or not
    power_threshold = 0.1   # MW
    volt_threshold  = 40    # keV

    #########################
    # loop over all sources #
    #########################
    for idx, sour in enumerate(source_information['sources']):
        
        # set up a sub-dictionary for the source
        source_information[sour] = {}

        #############################################################
        # load source power from data base outside of temporal loop #
        #############################################################
        # as one import contains all the power and voltage data for a given shot,
        # load them before the different time stamps are assessed individually
        source_information[sour]['time_power'], source_information[sour]['power'] = routines['power'][sour](shot_information[0], timeout = timeout)

        # assign a source the 'turned on'-status if it's output power surpasses the given threshold
        if max(source_information[sour]['power']) > power_threshold:
            sources.append(sour)
            source_index.append(source_information['source_index'][idx])

        #####################################################################################
        # calculate the average output power per beam by averaging the passed time interval #
        #####################################################################################
        # only do that if the source was identified as relavant beforehand however
        if sour in sources:
            # now also loop over all passed time ranges
            for PIT_idx, PIT in enumerate(shot_information[1]):
                # add a sub level for the point in time to the dictionary
                handle    = 't = ' + str(PIT)
                source_information[sour][handle] = {}
                # rename variables to have shorter names
                time      = source_information[sour]['time_power']
                power     = source_information[sour]['power']
                # reduce the power array based on the time slot
                power_red = power[np.where(time <= shot_information[2][PIT_idx])]
                time_red  = time [np.where(time <= shot_information[2][PIT_idx])]
                power_red = power_red[np.where(time_red >= PIT)]
                time_red  = time_red [np.where(time_red >= PIT)]
                # do the averaging math
                power_avg = sum(power_red) / len(power_red)
                # save the result to the dictionary
                source_information[sour][handle]['power'] = power_avg

            # if wanted, plot the source time trend
            if debug:
                plt.plot(time, power, label = 'Source ' + str(source_information['sources'][idx]))
                plt.xlim(left=0)
                plt.xlabel('t in s')
                plt.ylabel('NBI power in MW')
                plt.legend()
                plt.title('%s - Power trace' % (shot_information[0]))
                plt.show()

        #####################################################################
        # read the acceleration voltage, again outside of the temporal loop #
        #####################################################################
        source_information[sour]['time_voltage'], source_information[sour]['voltage'] = routines['voltage'][sour](shot_information[0], timeout = timeout)

        # if the source was previously identified to be relevant, assess its voltage
        if sour in sources:
            # now also loop over all the passed time points
            for PIT_idx, PIT in enumerate(shot_information[1]):
                # put together the handle for the dictionary
                handle   = 't = ' + str(PIT)
                # rename variables to have shorter names
                time     = source_information[sour]['time_voltage']
                voltage  = source_information[sour]['voltage']
                # reduce the arrays for the relevant range
                volt_red = voltage[(time <= shot_information[2][PIT_idx]) & (time >= shot_information[1][PIT_idx])]
                time_red = time   [(time <= shot_information[2][PIT_idx]) & (time >= shot_information[1][PIT_idx])]
                # reduce the data points to only those above the threshold as we cannot afford to have 
                # an averaged voltage with zeros as we need the applied voltage
                volt_red = volt_red[(volt_red > volt_threshold)]
                # do the averaging math
                volt_avg = np.average(volt_red)
                # save the result to the dictionary
                source_information[sour][handle]['voltage'] = volt_avg

            # plot the data if demanded
            if debug:
                plt.plot(time, voltage, label = 'Source ' + str(source_information['sources'][idx]))
                plt.xlim(left=0)
                plt.xlabel('t in s')
                plt.ylabel('NBI voltage in kV')
                plt.legend()
                plt.title('%s - Voltage trace' % (shot_information[0]))
                plt.show()

        ##################################################
        # read the beam fractions from the look up table #
        ##################################################

        # assess all the different sources
        if sour in sources:
            # read in the beam fractions for the given source only when None are passed
            if fractions == None:
                beam_fractions = read_beam_fractions(shot_number, sour)
            else:
                beam_fractions = fractions
            # now also loop over all the NBI active points in time
            for PIT_idx, PIT in enumerate(shot_information[1]):
                # put together the handle for the dictionary
                handle   = 't = ' + str(PIT)
                # currently implemented via look up - needs to be changed in the long term
                source_information[sour][handle]['fraction'] = beam_fractions
                #get_nspec_currentFractions(shot_information[0], source_information['source_index'][idx])

        ###############################################################################
        # eventually write some general information which is the same for all sources #
        ###############################################################################
        source_information[sour]['type'] = 'H'    # obviously hardcoded atm, may change later
        source_information['source_arr'] = source_index

    # print the as used identified sources
    print('Sources %s found to be active in discharge' % sources)
        
    # get the path to the storage folder
    path = os.path.dirname(pyfidasim.__file__)
    file_path = os.path.abspath(os.path.join(path,os.pardir))
        
    # save the dictionary for all the points in time resolved to individual hdf5 files
    for PIT_idx, PIT in enumerate(shot_information[1]):
        # set up a dictionary for the data related to a single point in time
        beam_information = {}
        # put together the handle for the dictionary
        handle   = 't = ' + str(PIT)
        # fill the dictionary with some general information
        beam_information['sources']    = source_information['sources']
        beam_information['source_arr'] = source_information['source_arr']

        # loop over the sources
        for sour in source_information['sources']:
            # set up a sub-dictionary
            beam_information[sour] = {}
            # if the source was identified to be active earlier, save the data
            if sour in sources:
                # fill the dictionary with source specific information
                beam_information[sour]['voltage']    = source_information[sour][handle]['voltage']
                beam_information[sour]['power']      = source_information[sour][handle]['power']
                beam_information[sour]['fraction']   = source_information[sour][handle]['fraction']
                beam_information[sour]['type']       = source_information[sour]['type']
            # if the source was identified to be inactive, fill the dictionary with some generic values
            else:
                beam_information[sour]['voltage']    = 0
                beam_information[sour]['power']      = 0
                beam_information[sour]['fraction']   = [1.0, 0.0, 0.0] 
                beam_information[sour]['type']       = 'H'
                
            # get rid of nans in case of alternating sources in same shot
            voltage = np.array([beam_information[sour]['voltage'] ])
            power = np.array([beam_information[sour]['power'] ])
            
            if np.isnan(voltage) or voltage <=0.0: beam_information[sour]['voltage'] = 0.0
            if np.isnan(power) or power <= 0.0: beam_information[sour]['power'] = 0.0
                
        # put together the file name and the path to the data folder
        # I tried making the path nicer but the other package caused unnecessary complications
        file_path += '/examples/W7X/Data/'
        file_name = str(shot_information[0]) + '_' + str(int(round(PIT*1000))) + '_NBI_parameters.hdf5'
        # and save the dictionary as an hdf5 file - in try beacuse does work from any other directory
        try:
            save_dict(beam_information, file_path + file_name)
        except OSError:
            print('Failed at writing file due to assumed path structure')
            
    return beam_information

#############################################################
# routine to read the beam fractions from a hdf5 dictionary #
#############################################################
def read_beam_fractions(shot_number, source):
    # shot number specifies the day the data was taken at
    # as parameters seems to change on a day to day base no time information within the shot asked for
    
    # put together the path to the .hdf5 file
    path_name = os.path.dirname(pyfidasim.__file__)
    path_name = os.path.abspath(os.path.join(path_name, os.pardir))
    path_name += '/examples/W7X/Data/'
    
    # load the dictionary
    data = load_dict(path_name + 'W7X_beam_fractions_lookup.hdf5')
    
    # check whether data for the requested shot number is available
    try:
        beam_fractions  = data[shot_number][source]
        return beam_fractions
    
    except:
        print('\nNo beam fractions for requested shot %s found in look up dictionary' % (shot_number))
        print('To add data, call add_shot_beam_fractions - path to data is:')
        print('http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/raw/W7XAnalysis/CDX_NI_Spectroscopy/USB_HR4000_Q8_NeutralParticleFractions_DATASTREAM/V2/0/E1')
        print()
        raise ValueError('No beam fractions stored for shot %s' % (shot_number))
    
################################################################################
# routine to add an entry to the hdf5 dictionary containing the beam fractions #
################################################################################
# example input:
# add_shot_beam_fractions('20180920.042', [0.31, 0.54, 0.15], 'Q8')

def add_shot_beam_fractions(shot_number, beam_fractions, source, overwrite = False):
    
    # check for the inputs to have proper format - crude but should find obvious mistakes
    if len(beam_fractions) != 3:
        raise ValueError('Length of beam fraction array not applicable')
    day, program = shot_number.split('.')
    if len(day) != 8 or len(program) != 3:
        raise ValueError('Passed shot number not in correct format - needs to be yyyymmdd.xxx')
    try:
        int(day)
        int(program)
    except:
        raise ValueError('Passed shot number has non-number character')
    
    # put together the path to the .hdf5 file
    path_name = os.path.dirname(pyfidasim.__file__)
    path_name = os.path.abspath(os.path.join(path_name, os.pardir))
    path_name += '/examples/W7X/Data/'
    file_name = 'W7X_beam_fractions_lookup.hdf5'
    
    # open the dictionary
    data = load_dict(path_name + file_name)
    
    # check whether the requested shot already exists
    if shot_number in data:
        # check whether data for the given source is already present
        if source in data[shot_number].keys():
            if not overwrite:
                print('\nShot %s with fractions %s already present in dictionary' % (shot_number, beam_fractions))
                print('To overwrite data, call again with overwrite = True\n')
            elif overwrite:
                print('\nOverwriting fractions for shot %s to %s\n' % (shot_number, beam_fractions))
                data[shot_number][source] = beam_fractions
        else:
            data[shot_number][source] = beam_fractions
            print('\nAdded %s as fractions for shot %s\n' % (beam_fractions, shot_number))
    else:
        data[shot_number] = {}
        data[shot_number][source] = beam_fractions
        print('\nAdded %s as fractions for shot %s\n' % (beam_fractions, shot_number))
        
    # save the dict with the additional entry
    save_dict(data, path_name + file_name)
    
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


def nbi_geometry():
    """Fills a blank dictionary with nbi geometry related parameters.
    
    Returns
    ----------
    nbi_geometry : dictionary, nbi geometry parameters
    """
    
    default_geometry = {
        'ion_source_size': np.array([22.8, 50.6]), 
        'focal_length': np.array([650., 700.]), 
        'divergence': np.array([0.8, 0.8]) / 180 * np.pi, 
        'aperture_1_rectangular' : True, 
        'aperture_1_size': np.array([100., 100.]), 
        'aperture_1_distance': 650., 
        'aperture_1_offset': np.array([0., 0.]), 
        'aperture_2_rectangular' : True, 
        'aperture_2_size': np.array([100., 100.]), 
        'aperture_2_distance': 650.,   # cm
        'aperture_2_offset': np.array([0., 0.])
    }
    
    Q1_geometry = copy.deepcopy(default_geometry)
    Q1_geometry['ID'] = 'Q1'
    Q1_geometry['source_position'] = np.array(
        [607.5594, 1171.7889, 29.5000])  # cm
    Q1_geometry['direction'] = np.array([-0.365400, -0.926946, -0.085174])
    Arot, Brot = calc_arot_brot(Q1_geometry['direction'])
    Q1_geometry['uvw_xyz_rot'] = Brot @ Arot

    
    Q2_geometry = copy.deepcopy(default_geometry)
    Q2_geometry['ID'] = 'Q2'
    Q2_geometry['source_position'] = np.array(
        [692.2962, 1131.0989, 29.5000])  # cm
    Q2_geometry['direction'] = np.array([-0.494953, -0.864735, -0.085174])
    Arot, Brot = calc_arot_brot(Q2_geometry['direction'])
    Q2_geometry['uvw_xyz_rot'] = Brot @ Arot
    
    
    Q3_geometry = copy.deepcopy(default_geometry)
    Q3_geometry['ID'] = 'Q3'
    Q3_geometry['source_position'] = np.array(
        [692.2962, 1131.0989, -90.5000])  # cm
    Q3_geometry['direction'] = np.array([-0.494953, -0.864735, 0.085174])
    Arot, Brot = calc_arot_brot(Q3_geometry['direction'])
    Q3_geometry['uvw_xyz_rot'] = Brot @ Arot
    
    
    Q4_geometry = copy.deepcopy(default_geometry)
    Q4_geometry['ID'] = 'Q4'
    Q4_geometry['source_position'] = np.array(
        [607.5594, 1171.7889, -90.5000])  # cm
    Q4_geometry['direction'] = np.array([-0.365400, -0.926946, 0.085174])
    Arot, Brot = calc_arot_brot(Q4_geometry['direction'])
    Q4_geometry['uvw_xyz_rot'] = Brot @ Arot
    
    
    Q5_geometry = copy.deepcopy(default_geometry)
    Q5_geometry['ID'] = 'Q5'
    Q5_geometry['source_position'] = np.array(
        [197.2344, 1305.1116, -29.5000])  # cm
    Q5_geometry['direction'] = np.array([-0.249230, -0.964692, 0.085174])
    Arot, Brot = calc_arot_brot(Q5_geometry['direction'])
    Q5_geometry['uvw_xyz_rot'] = Brot @ Arot
    
    
    
    Q6_geometry = copy.deepcopy(default_geometry)
    Q6_geometry['ID'] = 'Q6'
    Q6_geometry['source_position'] = np.array(
        [104.7639, 1321.9997, -29.5000])  # cm
    Q6_geometry['direction'] = np.array([-0.107854, -0.990511, 0.085174])
    Arot, Brot = calc_arot_brot(Q6_geometry['direction'])
    Q6_geometry['uvw_xyz_rot'] = Brot @ Arot
    
    
    Q7_geometry = copy.deepcopy(default_geometry)
    Q7_geometry['ID'] = 'Q7'
    Q7_geometry['source_position'] = np.array([104.27, 1317.11, 90.67])  # cm
    Q7_geometry['direction'] = np.array([-0.1085, -0.99, -.085])
    Arot, Brot = calc_arot_brot(Q7_geometry['direction'])
    Q7_geometry['uvw_xyz_rot'] = Brot @ Arot
    Q7_geometry['aperture_1_distance']= 626.9   # cm
    Q7_geometry['aperture_1_offset'] = np.array([1.7,-4.8])
    Q7_geometry['aperture_1_size']   = np.array([34.1, 67.0])
    
    Q7_geometry['aperture_2_distance']= 708.4   # cm
    Q7_geometry['aperture_2_offset'] = np.array([1.9,3.2])
    Q7_geometry['aperture_2_size']   = np.array([34.1, 57.4])
    
    
    Q8_geometry = copy.deepcopy(default_geometry)
    Q8_geometry['ID'] = 'Q8'
    Q8_geometry['source_position'] = np.array([195.96, 1300.37, 90.67])  # cm
    Q8_geometry['direction'] = np.array([-0.2486, -0.9648, -.085])
    Q8_geometry['aperture_1_distance']= 626.9   # cm
    Q8_geometry['aperture_1_offset'] = np.array([5.6,-4.8])
    Q8_geometry['aperture_1_size']   = np.array([34.1, 67.0])
    
    Q8_geometry['aperture_2_distance']= 708.4   # cm
    Q8_geometry['aperture_2_offset'] = np.array([-5.9,3.2])
    Q8_geometry['aperture_2_size']   = np.array([34.1, 57.4])
    
    Arot, Brot = calc_arot_brot(Q8_geometry['direction'])
    Q8_geometry['uvw_xyz_rot'] = Brot @ Arot
    
    nbi_geometry = {}
    nbi_geometry['sources'] = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
    nbi_geometry['Q1'] = Q1_geometry
    nbi_geometry['Q2'] = Q2_geometry
    nbi_geometry['Q3'] = Q3_geometry
    nbi_geometry['Q4'] = Q4_geometry
    nbi_geometry['Q5'] = Q5_geometry
    nbi_geometry['Q6'] = Q6_geometry
    nbi_geometry['Q7'] = Q7_geometry
    nbi_geometry['Q8'] = Q8_geometry
    
    return nbi_geometry

def W7X_nbi(shot_number = '20180920.042', t_start = 6.5, t_stop = 6.52, fractions = None, debug = False,default=False):
    nbigeom = nbi_geometry()
    nbiparams = nbi_parameters(shot_number, t_start, t_stop, fractions, debug, default)
    
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
    