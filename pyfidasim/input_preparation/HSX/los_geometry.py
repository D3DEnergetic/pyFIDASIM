import numpy as np
import h5py
from pathlib import Path

#===============================================================================
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
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans

#===============================================================================
def Thomson_setup(directory=None):
    
    if directory is None:
        directory = 'examples/HSX/Data'
    
    spec = {}

    los_uvec_file = directory + '/hsx_ts_los_uvec.txt'
    lens_pos_file = directory + '/hsx_ts_lens_pos.txt'
    
    los_uvec = np.loadtxt(los_uvec_file, delimiter=' ')
    lens_pos = np.loadtxt(lens_pos_file, delimiter=' ')
    
    los_dict = load_dict(directory + '/hsx_ts_los.hdf5')
    
    spec['los_dict'] = los_dict
    
    spec['dlam'] = 0.5
    spec['lambda_min'] = 650.
    spec['lambda_max'] = 1070. + spec['dlam']
    spec['wavel'] = np.arange(spec['lambda_min'], spec['lambda_max'],
                              spec['dlam'])
    spec['nlam'] = len(spec['wavel'])
    
    # Solid angle and etendue per polychromator/spatial channel
    etendue_file = directory + '/hsx_ts_etendue.txt'
    etendue_dtype = [('spatial', int), ('solid_angle', float),
                         ('area', float)]
    etendue_data = np.loadtxt(etendue_file, delimiter=' ',
                                     dtype=etendue_dtype)
    
    etendue = etendue_data['solid_angle'] * etendue_data['area']
    
    spec['etendue'] = etendue

    spec['los_lens'] = lens_pos
    spec['los_vec'] = los_uvec
    spec['nlos'] = np.shape(lens_pos)[0]
    spec['calc_stark_spitting'] = True
    spec['only_pi'] = False
    
    return(spec)
    
def DNB_poloidal_los_setup():
    # Brute Force, assign
    nlos = 20
    los_lens = np.array([[-0.08310226, 1.97433569, 0.04637054],
                         [-0.08310226, 1.97433569, 0.04637054],
                         [-0.08310226, 1.97433569, 0.04637054],
                         [-0.08310226, 1.97433569, 0.04637054],
                         [-0.08310226, 1.97433569, 0.04637054],
                         [-0.04793005, 1.93497613, 0.09550364],
                         [-0.04793005, 1.93497613, 0.09550364],
                         [-0.04793005, 1.93497613, 0.09550364],
                         [-0.04793005, 1.93497613, 0.09550364],
                         [-0.04793005, 1.93497613, 0.09550364],
                         [-0.02529182, 1.97433569, 0.15152735],
                         [-0.02529182, 1.97433569, 0.15152735],
                         [-0.02529182, 1.97433569, 0.15152735],
                         [-0.02529182, 1.97433569, 0.15152735],
                         [-0.02529182, 1.97433569, 0.15152735],
                         [0.00988039, 1.93497613, 0.20066044],
                         [0.00988039, 1.93497613, 0.20066044],
                         [0.00988039, 1.93497613, 0.20066044],
                         [0.00988039, 1.93497613, 0.20066044],
                         [0.00988039, 1.93497613, 0.20066044]
                         ]) * 100
    los_vec = np.zeros((nlos, 3))
    los_nbi_pos = np.array([[-0.96, 144.42, -1.75],
                            [-0.48, 144.42, -0.88],
                            [0, 144.42, 0],
                            [0.48, 144.42, 0.88],
                            [0.96, 144.42, 1.75],
                            [1.45, 144.42, 2.63],
                            [1.93, 144.42, 3.51],
                            [2.41, 144.42, 4.38],
                            [2.89, 144.42, 5.26],
                            [3.37, 144.42, 6.13],
                            [3.85, 144.42, 7.01],
                            [4.34, 144.42, 7.89],
                            [4.82, 144.42, 8.76],
                            [5.30, 144.42, 9.64],
                            [5.78, 144.42, 10.52],
                            [6.26, 144.42, 11.39],
                            [6.74, 144.42, 12.27],
                            [7.23, 144.42, 13.15],
                            [7.71, 144.42, 14.02],
                            [8.19, 144.42, 14.90]
                            ])  # Contains intersection positions

    for ilos in range(nlos):
        los_vec[ilos, :] = los_nbi_pos[ilos] - los_lens[ilos]
        los_vec[ilos, :] /= np.linalg.norm(los_vec[ilos, :])

    spec = {}
    spec['los_lens'] = los_lens
    spec['los_vec'] = los_vec
    spec['nlos'] = nlos 
    spec['calc_stark_spitting'] = True
    spec['only_pi'] = False  
    return(spec)
