import numpy as np
from .toolbox import rotate_uvw
from .grid3d import get_grid3d_indices
from ._fields import interp_fields
#from TRANSP.read_transp_profiles import read_transp_profiles
try:
    import numba
    numba_is_available = True
except:
    numba_is_available = False
    print('Numba is not available; consider installing Numba')

def conditional_numba(skip_numba=False):
    def decorator(func):
        if numba_is_available and not skip_numba:
            return numba.jit(func,cache=True, nopython=True,nogil=True, )
        else:
            return func
    return decorator

def nbi_intersection(spec, nbigeom, source_arr):
    # Select the first source in use
    ii = 0
    source_key = source_arr[ii]
    source = nbigeom['sources'][source_key]

    # Extract necessary parameters
    los_vec = spec['los_vec']          # Shape: (nlos, 3)
    los_pos = spec['los_pos']        # Shape: (nlos, 3)
    direction = nbigeom[source]['direction']    # Shape: (3,)
    source_position = nbigeom[source]['source_position']  # Shape: (3,)

    # Compute cross products
    # n = cross(los_vec, direction) for all LOS
    n = np.cross(los_vec, direction)    # Shape: (nlos, 3)

    # n2 = cross(direction, n) for all LOS
    n2 = np.cross(direction, n)        # Shape: (nlos, 3)

    # Compute the vector from LOS starting points to source position
    delta = source_position - los_pos  # Shape: (nlos, 3)

    # Compute dot products for numerator and denominator
    numerator = np.einsum('ij,ij->i', delta, n2)    # Shape: (nlos,)
    denominator = np.einsum('ij,ij->i', los_vec, n2)  # Shape: (nlos,)

    # Compute scalar factors for each LOS
    scalar_factors = (numerator / denominator)[:, np.newaxis]  # Shape: (nlos, 1)

    # Compute intersection points
    cpoint1 = los_pos + scalar_factors * los_vec  # Shape: (nlos, 3)

    # Compute distances from LOS starting points to intersection points
    distance = np.linalg.norm(cpoint1 - los_pos, axis=1)  # Shape: (nlos,)

    # Compute cylindrical coordinates
    R_los = np.sqrt(cpoint1[:, 0]**2 + cpoint1[:, 1]**2)   # Shape: (nlos,)
    Z_los = cpoint1[:, 2]                                  # Shape: (nlos,)
    phi_los = np.arctan2(cpoint1[:, 1], cpoint1[:, 0])    # Shape: (nlos,)

    return R_los, Z_los, phi_los, distance

def calc_s_along_los(los_pos,los_vec,fields,dl=0.1):  
    # Find intersection lengths of the LOS with the grid.
    dist_arr = np.arange(0, 2. * fields['Rmax'], dl)
    xyz_arr = np.zeros([3, len(dist_arr)])
    for kk in range(3):
        xyz_arr[kk, :] = los_pos[kk] + dist_arr[:] * los_vec[kk]
        

        # get the R,Z,phi positions from the xyz array of positions
    R = np.sqrt(xyz_arr[0, :]**2 + xyz_arr[1, :]**2)
    Z = xyz_arr[2, :]
    phi = np.arctan2(xyz_arr[1, :], xyz_arr[0, :]) + fields['rotate_phi_grid']
    # in case phi < 0, add 2pi
    index = phi < 0
    phi[index] += 2. * np.pi
    # if the simulation is only for one period and we make use of the symmetry,
    # then, dphi_sym is not 2.pi and we get another phi (phi2)
    phi = phi[:] % fields['dphi_sym']
    # sort out points outside the plasma
    index = (R > fields['Rmin']) & (R < fields['Rmax']) & \
            (Z > fields['Zmin']) & (Z < fields['Zmax']) & \
            (phi > fields['phimin']) & (phi < fields['phimax'])
    if sum(index) == 0:
        print('LOS outside grid')
        return()

    R = R[index]
    Z = Z[index]
    phi = phi[index]
    xyz_arr = xyz_arr[:, index]
    dist_arr = dist_arr[index]
    
    # get the s-coordinate along all points
    #s = interp_fields_array(fields, R, Z, phi, 's')
    s_arr = interp_fields(R, Z, phi, fields['s'], \
            fields['R'],fields['Rmin'],fields['dR'],fields['nr'], \
            fields['Z'],fields['Zmin'],fields['dZ'],fields['nz'], \
            fields['phi'],fields['phimin'],fields['dphi'],fields['nphi'])
    
    index = s_arr == s_arr
    if sum(index) == 0:
        print('all s-values are NAN')
        return()
   
    xyz_arr = xyz_arr[:, index]
    dist_arr = dist_arr[index]
    s_arr = s_arr[index]
    
    index = s_arr < 1.5
    xyz_arr = xyz_arr[:, index]
    dist_arr = dist_arr[index]
    s_arr = s_arr[index]    
    return(dist_arr,xyz_arr,s_arr)
    
@conditional_numba(skip_numba=False)  
def grid_cell_crossed_cone(grid_cell_crossed,full_los_length,r_3d, z_3d, phi_3d, spec_los_pos, spec_los_vec, spec_nlos, focal_length, lens_diameter):
    dr =(r_3d[1]-r_3d[0])
    dz =(z_3d[1]-z_3d[0])
    dphi = (phi_3d[1]-phi_3d[0])
    i=0
    for ilos in range(spec_nlos):
        l_pos = spec_los_pos[ilos,:]
        los = spec_los_vec[ilos,:]
        full_los_length[ilos] = 2*focal_length
        for ir, r in enumerate(r_3d):
            for iz, z in enumerate(z_3d):
                for iphi, phi in enumerate(phi_3d):
                    # xyz = np.array([(r+dr/2)*np.cos(phi+dphi/2),(r+dr/2)*np.sin(phi+dphi/2),z+dz/2])
                    # d_cross = norm(cross_(xyz-l_pos,2.*focal_length*los))/norm(2.*focal_length*los)
                    # d_los = abs(dot(los,xyz-(l_pos+focal_length*los)))
                    # d_cone = lens_diameter/(2*focal_length)*d_los
                    # r_min = np.sqrt(dr**2/4+dz**2/4+(r*dphi)**2/4)
                    # A_min = np.pi*r_min**2
                    # A = (np.pi/4)*(d_los*lens_diameter/focal_length)**2
                    # if d_cross < d_cone:
                    #     i=i+1
                    #     A = (np.pi/4)*(d_los*lens_diameter/focal_length)**2
                    #     grid_cell_crossed[ilos, ir, iz, iphi] = 1/A
                    # elif d_cross < r_min:
                    #     i=i+1
                    #     grid_cell_crossed[ilos, ir, iz, iphi] = 1/(A+np.pi*dr**2/4+d_los**2*np.pi)                    
                    
                    
                    xyz = np.array([(r+dr/2)*np.cos(phi+dphi/2),(r+dr/2)*np.sin(phi+dphi/2),z+dz/2])
                    d_cross = np.linalg.norm(np.cross(xyz-l_pos,2.*focal_length*los))/np.linalg.norm(2.*focal_length*los)
                    d_los = abs(np.dot(los,xyz-(l_pos+focal_length*los)))
                    d_cone = lens_diameter/(2*focal_length)*d_los
                    # d_min = np.sqrt(dr**2/4+dz**2/4)
                    d_min = np.sqrt(dr**2/4+dz**2/4+(r*dphi)**2/4)
                    A_min = np.pi*d_min**2/4
                    # print("d_min = ")
                    # print(d_min)
                    # print("d_cone =")
                    # print(d_cone)
                    # print("d_los = ")
                    # print(d_los)
                    if d_cross < d_cone or d_cross < d_min:
                          i=i+1
                          # print("d_los")
                          # print(d_los)
                          # print("d_cross =")
                          # print(d_cross)
                          A = (np.pi/4)*(d_los*lens_diameter/focal_length)**2
                          grid_cell_crossed[ilos, ir, iz, iphi] = 1/(A+A_min)
                         
                         
    print("total grid cels:")
    print(i)
    return(grid_cell_crossed)    

def grid_intersections_cone(spec,fields,grid3d,nbigeom, source_arr):
    r_3d = np.linspace(grid3d["Rmin"],grid3d["Rmax"],num = grid3d["nR"])
    z_3d = np.linspace(grid3d["Zmin"],grid3d["Zmax"],num = grid3d["nZ"])
    phi_3d = np.linspace(grid3d["phimin"],grid3d["phimax"],num = grid3d["nphi"])
    grid_cell_crossed = np.zeros([spec["nlos"],len(r_3d), len(z_3d), len(phi_3d)])
    full_los_length = np.zeros(spec["nlos"])
    grid_cell_d_los_full = grid_cell_crossed_cone(grid_cell_crossed,full_los_length,r_3d, z_3d, phi_3d, spec['los_pos'],spec['los_vec'], spec["nlos"],spec["los_f_len"][0], spec["d_lens"])
    grid_cell_cross_by_los = (grid_cell_d_los_full != 0)
    # spec["dl_per_grid_intersection"] = grid_cell_d_los_full[grid_cell_cross_by_los]
    los_grid_intersection_indices = []
    # dl_per_grid_intersection = []
    lenmax = 0 
    for i in range(spec["nlos"]): 
        ilen = len(grid_cell_d_los_full[i,:,:,:][grid_cell_cross_by_los[i,:,:,:]])
        if ilen > lenmax:
            lenmax = ilen

    for i in range(spec['nlos']):
        lendiff = lenmax -len(grid_cell_d_los_full[i,:,:,:][grid_cell_cross_by_los[i,:,:,:]])
        los_grid_intersection_indices.append(np.concatenate((np.array(np.where(grid_cell_cross_by_los[i,:,:,:])),np.zeros([3,lendiff])),axis=1))
        # dl_per_grid_intersection.append(np.concatenate((np.array(np.where(grid_cell_cross_by_los[i,:,:,:])).T,np.zeros(lendiff))))
        # dl_per_grid_intersection.append(np.concatenate((np.array(grid_cell_d_los_full[i,:,:,:][grid_cell_cross_by_los[i,:,:,:]]),np.zeros(lendiff))))
    spec['los_grid_intersection_indices'] = np.array(los_grid_intersection_indices,dtype=int)
    # spec['dl_per_grid_intersection'] = np.array(dl_per_grid_intersection,dtype=float)
    spec["grid_cell_crossed_by_los"] = np.logical_or.reduce(grid_cell_cross_by_los, axis=0,dtype=bool)
    spec["los_grid_intersection_weight"] = grid_cell_d_los_full
    # los_grid_intersection_indices = []
    # dl_per_grid_intersection = []
    # for ilos in range(spec["nlos"]):
    #     # los_grid_intersection_indices_i = []
    #     # dl_per_grid_intersection_i = []
    #     l_pos = spec['los_pos'][ilos,:]
    #     los = spec['los_vec'][ilos,:]
    #     spec['full_los_length'][ilos] = 2*fl
    #     for ir, r in enumerate(r_3d):
    #         for iz, z in enumerate(z_3d):
    #             for iphi, phi in enumerate(phi_3d):
    #                 xyz = [r*np.cos(phi),r*np.sin(phi),z]
    #                 d_cross = norm(cross_(xyz-l_pos,2.*fl*los))/norm(2.*fl*los)
    #                 d_los = abs(dot(los,xyz-(l_pos+fl*los)))
    #                 if d_cross < (ld/(2*fl))*d_los:
    #                      # los_grid_intersection_indices_i.append([ir,iz,iphi])
    #                      # dl_per_grid_intersection_i.append(d_los)
    #                      spec['grid_cell_crossed_by_los'][ir, iz, iphi] = 1
        # los_grid_intersection_indices.append(los_grid_intersection_indices_i)
        # dl_per_grid_intersection.append(dl_per_grid_intersection_i)
    # this is a list of indices where a given los crosses the simulation grid
    # spec['los_grid_intersection_indices'] = np.array(los_grid_intersection_indices)
    # this is a list of the intersection lenght of a LOS at a given grid cell
    # on the above defined list
    # spec['dl_per_grid_intersection'] = np.array(dl_per_grid_intersection)            
    return(spec)


def grid_intersections(spec, fields, grid3d):
    '''
    Routine to find intersection lengths of the LOS with the 3D grid.
    '''



    # Define step length for grid cell intersection determination
    dl = grid3d['dR'] / 20.0

    # Define storage arrays
    spec['grid_cell_crossed_by_los'] = np.zeros((grid3d['nR'], grid3d['nZ'], grid3d['nphi']), dtype=bool)
    spec['full_los_length'] = np.zeros(spec['nlos'])

    ncell_crossed_estimate = int(2.0 * grid3d['Rmax'] / (grid3d['dR'] / 2))
    los_grid_intersection_indices = np.zeros((spec['nlos'], ncell_crossed_estimate, 3), dtype=int)
    dl_per_grid_intersection = np.zeros((spec['nlos'], ncell_crossed_estimate))
    s_per_grid_intersection = np.full((spec['nlos'], ncell_crossed_estimate), 2.0)

    ncell_crossed_max = 0

    for ilos in range(spec['nlos']):
        # Calculate s-positions along LOS
        dist_arr, xyz_arr, s_arr = calc_s_along_los(spec['los_pos'][ilos, :], spec['los_vec'][ilos, :], fields, dl=dl)

        # Compute dl_arr
        dl_arr = np.diff(dist_arr)
        dl_arr = np.insert(dl_arr, 0, dl)  # Ensure dl_arr[0] corresponds to dl

        # Check whether the path leaves and re-enters the plasma
        index = dl_arr > 3 * dl
        if np.any(index):
            for ijump in np.where(index)[0]:
                ii = ijump + 1
                if np.any(s_arr[:ii] < 1.1):
                    dist_arr = dist_arr[:ii]
                    xyz_arr = xyz_arr[:, :ii]
                    s_arr = s_arr[:ii]
                    # Recompute dl_arr after adjusting dist_arr
                    dl_arr = np.diff(dist_arr)
                    dl_arr = np.insert(dl_arr, 0, dl)
                    # Set last value of dl_arr to dl
                    dl_arr[-1] = dl
                    break

        if len(dist_arr) == 0:
            print(f'LOS #{ilos} is outside the grid.')
            continue

        spec['full_los_length'][ilos] = dist_arr[-1]

        # Get grid3d indices
        ir_arr, iz_arr, iphi_arr = get_grid3d_indices(
            xyz_arr[np.newaxis, ...],
            grid3d['rotate_phi_grid'], grid3d['dphi_sym'],
            grid3d['Rmin'], grid3d['dR'], grid3d['nR'],
            grid3d['Zmin'], grid3d['dZ'], grid3d['nZ'],
            grid3d['phimin'], grid3d['dphi'], grid3d['nphi']
        )
        ir_arr = ir_arr[0]
        iz_arr = iz_arr[0]
        iphi_arr = iphi_arr[0]

        # Mask to ensure indices are within grid boundaries
        grid_mask = (
            (0 <= ir_arr) & (ir_arr < grid3d['nR']) &
            (0 <= iz_arr) & (iz_arr < grid3d['nZ']) &
            (0 <= iphi_arr) & (iphi_arr < grid3d['nphi'])
        )

        if not np.any(grid_mask):
            print(f'LOS #{ilos} is outside the grid.')
            continue

        # Filter arrays using the mask
        ir_arr = ir_arr[grid_mask]
        iz_arr = iz_arr[grid_mask]
        iphi_arr = iphi_arr[grid_mask]
        dl_arr = dl_arr[grid_mask]
        s_arr = s_arr[grid_mask]

        # Convert 3D indices to flat indices
        flat_indices = np.ravel_multi_index(
            (ir_arr, iz_arr, iphi_arr),
            (grid3d['nR'], grid3d['nZ'], grid3d['nphi'])
        )

        # Initialize per-LOS accumulation dictionaries
        cell_index_map = {}
        cell_indices_list = []
        cell_dl_list = []
        cell_sdl_list = []

        for idx, flat_idx in enumerate(flat_indices):
            if flat_idx not in cell_index_map:
                # First time this grid cell is encountered
                cell_index_map[flat_idx] = len(cell_indices_list)
                cell_indices_list.append(flat_idx)
                cell_dl_list.append(0.0)
                cell_sdl_list.append(0.0)
                # Update grid_cell_crossed_by_los
                ir = ir_arr[idx]
                iz = iz_arr[idx]
                iphi = iphi_arr[idx]
                spec['grid_cell_crossed_by_los'][ir, iz, iphi] = True

            cell_pos = cell_index_map[flat_idx]
            cell_dl_list[cell_pos] += dl_arr[idx]
            cell_sdl_list[cell_pos] += s_arr[idx] * dl_arr[idx]

        num_cells_crossed = len(cell_indices_list)
        if num_cells_crossed > ncell_crossed_estimate:
            print('Attention, arrays to store s and dl are too small!')
            num_cells_crossed = ncell_crossed_estimate

        # Convert cell_indices_list back to 3D indices
        ir_unique, iz_unique, iphi_unique = np.unravel_index(
            cell_indices_list[:num_cells_crossed],
            (grid3d['nR'], grid3d['nZ'], grid3d['nphi'])
        )

        # Store data in arrays
        los_grid_intersection_indices[ilos, :num_cells_crossed, 0] = ir_unique
        los_grid_intersection_indices[ilos, :num_cells_crossed, 1] = iz_unique
        los_grid_intersection_indices[ilos, :num_cells_crossed, 2] = iphi_unique
        dl_per_grid_intersection[ilos, :num_cells_crossed] = cell_dl_list[:num_cells_crossed]
        s_per_grid_intersection[ilos, :num_cells_crossed] = np.array(cell_sdl_list[:num_cells_crossed]) / np.array(cell_dl_list[:num_cells_crossed])

        if num_cells_crossed > ncell_crossed_max:
            ncell_crossed_max = num_cells_crossed

    # Trim arrays to the maximum number of cells crossed
    spec['los_grid_intersection_indices'] = los_grid_intersection_indices[:, :ncell_crossed_max, :]
    spec['dl_per_grid_intersection'] = dl_per_grid_intersection[:, :ncell_crossed_max]
    spec['s_per_grid_intersection'] = s_per_grid_intersection[:, :ncell_crossed_max]

    # Check if any LOS enters the plasma
    index = dl_per_grid_intersection > 0
    if not np.any(index):
        print('los.py: not a single line of sight enters the plasma!')
        raise Exception("Error")

    return spec

def uvw_grid_intersections(spec, fields, grid3d, source_array, nbigeom):
    spec['uvw_grid_cell_crossed_by_los'] = {}
    spec['uvw_dl_per_grid_intersection'] = {}
    dl=0.1 #1 mm
    
    for i in source_array:
        source = nbigeom['sources'][i]
    
        spec['uvw_grid_cell_crossed_by_los'][source] = np.zeros([grid3d['nu'], grid3d['nv'], grid3d['nw'], spec['nlos']], dtype=bool)
        spec['uvw_dl_per_grid_intersection'][source] = np.zeros([grid3d['nu'], grid3d['nv'], grid3d['nw'], spec['nlos']])
        
        for ilos in range(spec['nlos']): #[0]:
            los_pos_pre = spec['los_pos'][ilos, :]
            los_vec_pre = spec['los_vec'][ilos, :]
            los_pos = rotate_uvw(nbigeom[source]['Binv'],nbigeom[source]['Ainv'],los_pos_pre-nbigeom[source]['source_position'])
            los_vec = rotate_uvw(nbigeom[source]['Binv'],nbigeom[source]['Ainv'],los_vec_pre)
            los_vec /= np.linalg.norm(los_vec)
            passed_zone=False
            dl_total=0
            in_zone=False
            oldu,oldv,oldw=-1,-1,-1
            dl_inter=0
            while not passed_zone:
                current_pos=los_pos+dl_total*los_vec
                current_u=int((-current_pos[0]-grid3d['umin'])/grid3d['du'])
                current_v=int((current_pos[1]-grid3d['vmin'])/grid3d['dv'])
                current_w=int((current_pos[2]-grid3d['wmin'])/grid3d['dw'])
                dl_total+=dl
                if current_u != oldu or current_v != oldv or current_w != oldw:
                    if oldu > -1 and oldv > -1 and oldw > -1:
                        if (grid3d['nu']-1)>oldu and (grid3d['nv']-1)>oldv and (grid3d['nw']-1)>oldw:
                            spec['uvw_dl_per_grid_intersection'][source][oldu,oldv,oldw,ilos]=dl_inter
                    oldu,oldv,oldw=current_u,current_v,current_w
                    dl_inter=0
                dl_inter += dl
                if (grid3d['nu']-1)<current_u or current_u<0 or (grid3d['nv']-1)<current_v \
                    or current_v<0 or (grid3d['nw']-1)<current_w or current_w<0:
                    if in_zone:
                        passed_zone=True
                    if dl_total > 1000.:
                        print("Line of sight did not cross NBI")
                        break
                    continue
                spec['uvw_grid_cell_crossed_by_los'][source][current_u,current_v,current_w,ilos] = True
                in_zone=True
    return(spec)