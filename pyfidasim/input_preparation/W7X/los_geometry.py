import numpy as np


def read_active_LOS_information_w7x(file=''):
    if file != '':
        f = open(file, 'r')
    else:
        try:
            f = open('Data/op12b_ils_geometry.txt', 'r')
        except BaseException:
            f = open('op12b_ils_geometry.txt', 'r')
    losraw = []
    for line in f:
        losraw.append(repr(line))
    f.close()

    los = {}
    los['unit'] = 'm'
#    'p0' = 'starting position'
#    'u'  = 'direction vector'
    for line in losraw:
        los[line.split('"')[3]] = {}
        tmp = line.split('"')[6].replace(':', '').replace(
            '[', '').replace(']', '').replace(',', '')
        b = []
        for i in tmp.split(' '):
            if i != '':
                b.append(float(i))
        los[line.split('"')[3]]['p0'] = b
        tmp = line.split('"')[8].replace(':', '').replace(
            '[', '').replace(']', '').replace(',', '')
        b = []
        for i in tmp.split(' '):
            if i != '':
                b.append(float(i))
        los[line.split('"')[3]]['u'] = b
        tmp = line.split('"')[10].replace(':', '').replace(
            '[', '').replace(']', '').replace(',', '')
        b = []
        for i in tmp.split(' '):
            if i != '':
                b.append(float(i))
        los[line.split('"')[3]]['pQ7'] = b
        tmp = line.split('"')[12].replace(':', '').replace(
            '[', '').replace(']', '').replace(',', '')
        b = []
        for i in tmp.split(' '):
            if i != '':
                b.append(float(i))
        los[line.split('"')[3]]['pQ8'] = b
    return los


def active_LOS_selection_w7x():
    losOn = ['NA', 'AEM21_S7:12', 'AEA21_A:06',
             'AEM21_S8:12', 'AEA21_A:08', 'AEM21_S7:16',
             'AEA21_A:10', 'AEM21_S8:16', 'AEA21_A:12',
             'AEM21_S7:20', 'AEA21_A:14', 'AEM21_S8:20',
             'AEA21_A:16', 'AEM21_S7:24', 'AEA21_A:18',
             'AEM21_S8:24', 'AEA21_A:20', 'AEM21_S7:28',
             'AEA21_A:22', 'AEM21_S8:28', 'AEA21_A:24',
             'AEM21_S7:32', 'AEA21_A:26', 'AEM21_S8:32',
             'AEA21_A:28', 'AEM21_S7:36', 'AEA21_A:30',
             'AEM21_S8:36', 'AEA21_A:32', 'AEM21_S7:40',
             'AEA21_A:34', 'AEM21_S8:40', 'AEA21_A:36',
             'AEM21_S7:44', 'AEA21_A:38', 'NA',
             'AEA21_X2:01', 'AEM21_S7:48', 'AEA21_X2:03',
             'AEA21_X2:04', 'AET21:10', 'AEM21_S7:51',
             'AET21:14', 'AEM21_S8:46', 'AET21:16',
             'AEA21_A:01', 'NA', 'AEA21_A:03',
             'AEA21_A:05', 'AEA21_X2:06', 'AEM21_X2:02',
             'AEM21_X2:04', 'AEM21_X2:08', 'NA']
    return losOn

def new_LOS_selection_w7x():
    losOn = ['AEM21_HPPS7:01', 'AEM21_HPPS7:02', 'AEM21_HPPS7:03',
             'AEM21_HPPS7:04', 'AEM21_HPPS7:05', 'AEM21_HPPS7:06',
             'AEM21_HPPS7:07', 'AEM21_HPPS7:08', 'AEM21_HPPS7:09',
             'AEM21_HPPS8:01', 'AEM21_HPPS8:02', 'AEM21_HPPS8:03',
             'AEM21_HPPS8:04', 'AEM21_HPPS8:05', 'AEM21_HPPS8:06',
             'AEM21_HPPS8:07', 'AEM21_HPPS8:08', 'AEM21_HPPS8:09']
    return losOn

def los_geometry(shot = '20180823.035',head='AEA',file='',default=False,spectrometers = ['ILS_Green'], new = False):

    if default:
        ## -----------------------------------------------------
        ## ---------- old implementation -----------------------
        ## -----------------------------------------------------
        if new:
            los_names = new_LOS_selection_w7x()
        else:   
            los_names = active_LOS_selection_w7x()
        los_info = read_active_LOS_information_w7x(file=file)

        
        if np.size(head)>1:
            nlos = np.int64(len(head))
            los_vec = np.zeros((nlos, 3))
            los_lens = np.zeros((nlos, 3))
            losnam = np.ones(nlos,dtype="<U10")
            for j in range(len(head)):
                headnam=head[j]
                if (len(headnam) < 2):
                    continue
                for i in range(nlos):
                    if (los_names[i][0:len(headnam)] == headnam):
                        dic = los_info[los_names[i]]
                        los_lens[j, :] = np.array(dic['p0']) * 100.
                        los_vec[j, :] = np.array(dic['u'])
                        losnam[j]=los_names[i]
                        break
        else:
            nlos = len(los_names)
            los_vec = np.zeros((nlos, 3))
            los_lens = np.zeros((nlos, 3))
            los_select = np.zeros(nlos, dtype=bool)
            losnam=[]
            for i in range(nlos):
                if los_names[i][0:len(head)] == head:
                    dic = los_info[los_names[i]]
                    los_lens[i, :] = np.array(dic['p0']) * 100.
                    los_vec[i, :] = np.array(dic['u'])
                    los_select[i] = True
                    losnam.append(los_names[i])

            los_vec = los_vec[los_select, :]
            los_lens = los_lens[los_select, :]
            nlos = np.int64(sum(los_select))
            
        los_names=losnam
        sigma_to_pi_ratio = np.ones(nlos)
    else:
        ## -----------------------------------------------------
        ## ---------- new implementation -----------------------
        ## -----------------------------------------------------
        import os
        import pickle
        import archivedb
        import pyfidasim
        from w7xspec.rawSpectra import getLOSInfo

        # set up arrays which a later returned - not empty as otherwise structure is not defined - is removed later
        los_lens = np.zeros((2,3))
        los_vec  = np.zeros((2,3))
        nlos     = 0
        los_names         = np.array([])
        sigma_to_pi_ratio = np.array([])
        
        # get the path to the pi/sigma information file
        path = os.path.dirname(pyfidasim.__file__)
        path = os.path.abspath(os.path.join(path, os.pardir))
        
        # load the pi/sigma values
        # those are defined as: I_pi+ = I_pi- = f*I_sigma with f being the stored value
        #filehandler = open(path + '\\pyfidasim\\input_preparation\\W7X\\pi_sigma_ratios.pkl', 'rb')
        filehandler = open(os.path.join(path, 'pyfidasim', 'input_preparation', 'W7X', 'pi_sigma_ratios.pkl'), 'rb')
        
        pi_sigma_values = pickle.load(filehandler)
        filehandler.close()

        # read in the LoS info from the data base for all passed spectrometers
        for spectrometer in spectrometers:

            # fetch the information
            losInfo   = getLOSInfo("QSK_CXRS", spectrometer, archivedb.get_program_t1("20181009.034"))

            # extract the LoS names, starting vectors, and los vectors for the given spectrometer
            LoS_names, Chan_names, start_points, LoS_vec = losInfo['losNames'], losInfo['chanNames'], losInfo['losStart'], losInfo['losUVec']

            # we need to remove some elements - try deals with None type objects
            # first condition removes Q-lines, second removes LoSs based on the passed head
            for index, los in enumerate(LoS_names):
                if not np.isnan(losInfo['losStart'][index][0]) and los[0:len(head)] == head:
                    # add the line of sight name
                    los_names = np.append(los_names, los)
                    
                    ## get the chan specific pi/sigma ratio
                    # we only have data for the ILS_Green spectrometer however
                    if spectrometer == 'ILS_Green':
                        handle = str(int(Chan_names[index][-2:]))
                    # in the other cases simply create a handle which returns and error in the try block
                    else:
                        handle = 'This will most certainly not work'
                    # try because we don't have the ratio for all lines of sight
                    try:
                        pi_sigma = pi_sigma_values[handle]
                    # in case that we don't have the data, set the ratio to unity
                    except:
                        pi_sigma = 1
                    # in Ben's implementation we need the sigma/pi ratio, i.e. invert
                    sigma_pi = 1/pi_sigma
                    # add the value to the list
                    sigma_to_pi_ratio = np.append(sigma_to_pi_ratio, sigma_pi)

                    add_lens  = start_points[index,:] * 100
                    add_vec   = LoS_vec[index,:]

                    los_lens  = np.concatenate((los_lens, add_lens[np.newaxis,:]), axis = 0)
                    los_vec   = np.concatenate((los_vec,  add_vec [np.newaxis,:]), axis = 0)
                    nlos += 1
                else:
                    continue

        # remove the initially initiated empty leading entries
        los_lens = los_lens[2:]
        los_vec  = los_vec [2:]

    # put the data into the correct data format
    spec = {}
    spec['los_pos'] = los_lens
    spec['los_vec']  = los_vec
    spec['nlos']     = nlos
    spec['losname']  = los_names
    spec['sigma_to_pi_ratio'] = sigma_to_pi_ratio
    return spec 

##############
# test input
##############
if __name__ == '__main__':
    spec1 = los_geometry(shot = '20180920.042', spectrometers = ['ILS_Green','AUG2','AUG1'], head = 'AE')
    #spec2 = los_geometry(head='AEA',default=True,file='../../../examples/W7X/Data/op12b_ils_geometry.txt', shot = '20180920.042')
    #bi_parameters('20180920.042', t_start = [6.5], t_stop = [6.52], debug = True)