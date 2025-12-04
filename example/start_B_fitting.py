# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:10:32 2022

@author: Samuel Stewart
"""
import numpy as np
import matplotlib.pyplot as plt
from pyfidasim._fields import interp_fields
from pyfidasim.toolbox import FWHM

try:
    import numba
    numba_is_available = True
except:
    numba_is_available = False
    print('Numba is not available; consider installing Numba')

# from scipy import interpolate
def conditional_numba(skip_numba=False):
    def decorator(func):
        if numba_is_available and not skip_numba:
            return numba.jit(func,cache=True, nopython=True,nogil=True, )
        else:
            return func
    return decorator


@conditional_numba(skip_numba=False)
def xyz_to_Bxyz(xyz,f_Br,f_Bz,f_Bphi,f_rotate_phi_grid,f_dphi_sym, \
                             f_R,f_Rmin,f_Rmax,f_dR,f_nr, \
                             f_Z,f_Zmin,f_Zmax,f_dZ,f_nz, \
                             f_phi,f_phimin,f_phimax,f_dphi,f_nphi):
    '''
    Get the magnetic field vecotr in x,y,z coordinates from the fields by
    using the 2d interpolation in "interp_fields"
    '''
    R = np.sqrt(xyz[0]**2 + xyz[1]**2)
    Z = xyz[2]
    phi = np.arctan2(xyz[1], xyz[0]) + f_rotate_phi_grid
    # if type(phi).__name__ == 'float64':
    #    print(ii,phi)
    if(phi < 0):
        phi += 2. * np.pi
    # re-define a new phi (phi2) in case we make use of the symmetry.
    phi2 = phi % f_dphi_sym

    Bxyz = np.zeros(3)
    '''
    Br2 = interp_fields_numba(R, Z, phi2,f_Br,f_dphi_sym,\
                             f_R,f_Rmin,f_Rmax,f_dR,f_nr, \
                             f_Z,f_Zmin,f_Zmax,f_dZ,f_nz, \
                             f_phi,f_phimin,f_phimax,f_dphi,f_nphi)
    if Br2 != Br2:  # check for NANs (this seems to occur for edge positions)
        return(Bxyz)
    Bz2 = interp_fields_numba(R, Z, phi2,f_Bz,f_dphi_sym, \
                             f_R,f_Rmin,f_Rmax,f_dR,f_nr, \
                             f_Z,f_Zmin,f_Zmax,f_dZ,f_nz, \
                             f_phi,f_phimin,f_phimax,f_dphi,f_nphi)
    Bphi2 = interp_fields_numba(R, Z, phi2,f_Bphi,f_dphi_sym, \
                             f_R,f_Rmin,f_Rmax,f_dR,f_nr, \
                             f_Z,f_Zmin,f_Zmax,f_dZ,f_nz, \
                             f_phi,f_phimin,f_phimax,f_dphi,f_nphi)
    '''


    Br2 = interp_fields(np.array([R]), np.array([Z]), np.array([phi2]), f_Br, \
                     f_R,f_Rmin,f_dR,f_nr, \
                     f_Z,f_Zmin,f_dZ,f_nz, \
                     f_phi,f_phimin,f_dphi,f_nphi)
    if Br2[0] != Br2[0]:  # check for NANs (this seems to occur for edge positions)
        return(Bxyz)
    Bz2 = interp_fields(np.array([R]), np.array([Z]), np.array([phi2]), f_Bz, \
                     f_R,f_Rmin,f_dR,f_nr, \
                     f_Z,f_Zmin,f_dZ,f_nz, \
                     f_phi,f_phimin,f_dphi,f_nphi)
    
    Bphi2 = interp_fields(np.array([R]), np.array([Z]), np.array([phi2]), f_Bphi, \
                     f_R,f_Rmin,f_dR,f_nr, \
                     f_Z,f_Zmin,f_dZ,f_nz, \
                     f_phi,f_phimin,f_dphi,f_nphi)    

    # to finally get Bxyz, remove the grid rotation (the LOS is not in the
    # rotated one!)
    phi -= f_rotate_phi_grid
    Bxyz = np.zeros(3)
    Bxyz[0] = -np.cos(np.pi * 0.5 - phi) * Bphi2[0] + np.cos(phi) * Br2[0]
    Bxyz[1] = np.sin(np.pi * 0.5 - phi) * Bphi2[0] + np.sin(phi) * Br2[0]
    Bxyz[2] = Bz2[0]
    return(Bxyz)

@conditional_numba(skip_numba=True) 
def field_line_following(xyz_start,dl,nstep, \
                               f_Br,f_Bz,f_Bphi,f_rotate_phi_grid,f_dphi_sym, \
                               f_R,f_Rmin,f_Rmax,f_dR,f_nr, \
                               f_Z,f_Zmin,f_Zmax,f_dZ,f_nz, \
                               f_phi,f_phimin,f_phimax,f_dphi,f_nphi):

    '''
    use the a leap-frog routine for field-line tracing (http://physics.ucsc.edu/~peter/242/leapfrog.pdf)
    inputs: xyz_start   - start position in XYZ
            dl          - step size
            nstep       - number of steps
    '''
    ## define arrays to store the outputs
    xyz_arr=np.zeros((nstep,3))      
    xyz=np.array(xyz_start)   ## start position
    ## --- get B and babs for the start-position ---
    B= xyz_to_Bxyz(xyz,f_Br,f_Bz,f_Bphi,f_rotate_phi_grid,f_dphi_sym, \
                             f_R,f_Rmin,f_Rmax,f_dR,f_nr, \
                             f_Z,f_Zmin,f_Zmax,f_dZ,f_nz, \
                             f_phi,f_phimin,f_phimax,f_dphi,f_nphi)

    bnorm=np.linalg.norm(B)
    
    for i in range(nstep): ## loop along the path of a given particles (nsteps)

        ## ----------------------------------------------------
        ## get the magnetic field vector at the "half way position"
        ## ---------------------------------------------------- 
        xyz_half_way=xyz+B/bnorm*0.5*dl                     ## go to the next position
        B = xyz_to_Bxyz(xyz_half_way,f_Br,f_Bz,f_Bphi,f_rotate_phi_grid,f_dphi_sym, \
                             f_R,f_Rmin,f_Rmax,f_dR,f_nr, \
                             f_Z,f_Zmin,f_Zmax,f_dZ,f_nz, \
                             f_phi,f_phimin,f_phimax,f_dphi,f_nphi)
        bnorm=np.linalg.norm(B) 
        if bnorm == 0:
            break
        xyz+= (B/bnorm*dl)   ## go to the next position
        ## store the velocity, Bfield and position in the array
        xyz_arr[i,:]=xyz
        
    ## reduce the size of the output array
    xyz_arr=xyz_arr[0:i,:]
    return(xyz_arr)

def B_coord_Tform(xyz_start, grid3d, fields):
    '''author - SStewart
    For a start position, calculates a field line and the corresponding coordinates
    for each point along the field line    
    '''
    dl = np.min([grid3d["dR"],grid3d["dZ"]])
    nstep = int(np.floor((grid3d["Rmin"]+0.5*(grid3d["Rmax"]-grid3d["Rmin"]))*(grid3d["phimax"]-grid3d["phimin"])/(dl)))
    # follows the Bfieldline in both directions for nstep (note factor of 2)
    Bfield_pos_arr_pos = field_line_following(xyz_start,dl,nstep,fields['Br'],fields['Bz'],fields['Bphi'],fields['rotate_phi_grid'],fields['dphi_sym'], \
                fields['R'],fields['Rmin'],fields['Rmax'],fields['dR'],fields['nr'], \
                fields['Z'],fields['Zmin'],fields['Zmax'],fields['dZ'],fields['nz'], \
                fields['phi'],fields['phimin'],fields['phimax'],fields['dphi'],fields['nphi'])
    Bfield_pos_arr_neg = field_line_following(xyz_start,dl,nstep,-fields['Br'],-fields['Bz'],-fields['Bphi'],fields['rotate_phi_grid'],fields['dphi_sym'], \
                fields['R'],fields['Rmin'],fields['Rmax'],fields['dR'],fields['nr'], \
                fields['Z'],fields['Zmin'],fields['Zmax'],fields['dZ'],fields['nz'], \
                fields['phi'],fields['phimin'],fields['phimax'],fields['dphi'],fields['nphi'])
    print(np.shape(Bfield_pos_arr_pos))
    #combines into a single field line
    Bfield_pos_arr = np.concatenate((np.flip(Bfield_pos_arr_neg,axis=0),Bfield_pos_arr_pos),axis=0)
    Bfield_coord_arr = np.zeros((len(Bfield_pos_arr),3,3))
    #find xyz find Flux path array in xyz find binorm in 
    for i, pos in enumerate(Bfield_pos_arr):
        if i == 0 or (pos == Bfield_pos_arr[-1]).all(): # Excludes first and last points on field line
            continue
        B = xyz_to_Bxyz(pos,fields['Br'],fields['Bz'],fields['Bphi'],fields['rotate_phi_grid'],fields['dphi_sym'], \
                fields['R'],fields['Rmin'],fields['Rmax'],fields['dR'],fields['nr'], \
                fields['Z'],fields['Zmin'],fields['Zmax'],fields['dZ'],fields['nz'], \
                fields['phi'],fields['phimin'],fields['phimax'],fields['dphi'],fields['nphi'])
        b_dir = B/np.linalg.norm(B) #Direction parallel to B
        
        pos_rz = [(np.sqrt(pos[0]**2+pos[1]**2),pos[2])]
        posr = pos_rz[0][0]
        posz = pos_rz[0][1]
        mindists = np.zeros((len(fields["Rsurf"][:,0,0]),5,2)) # [isurf, [mindist(1,2),]]
        for j in range(len(fields["Rsurf"][:,0,0])): # iterates over different flux surfaces 
            rz_surf = np.array([fields["Rsurf"][j,0,:].T , fields["Zsurf"][j,0,:].T]).T
            surf_dist,surf_num = spatial.KDTree(rz_surf).query(pos_rz,4) # Finds closest 4 points
            surf_num.sort()
            mindists[j,:,:] = [[surf_dist[0,0],surf_dist[0,1]], \
                               [rz_surf[surf_num[0,0],0],rz_surf[surf_num[0,0],1]],\
                               [rz_surf[surf_num[0,1],0],rz_surf[surf_num[0,1],1]],\
                               [rz_surf[surf_num[0,2],0],rz_surf[surf_num[0,2],1]],\
                               [rz_surf[surf_num[0,3],0],rz_surf[surf_num[0,3],1]],\
                            ] #Adds all sets of 4 points into a array
            
        
        fluxi = np.argmin((mindists[:,0,0]+mindists[:,0,1])/2) #Finds the flux surface number
        mindists_sub = np.concatenate((mindists[:fluxi,:,:], mindists[(fluxi+1):,:,:])) 
        flux1 = mindists[fluxi,:,:] #flux surface 1 with distances and points
        flux2 = mindists_sub[np.argmin((mindists_sub[:,0,0]+mindists_sub[:,0,1])/2),:,:]
        # Calculates the slopes of local norms and distances
        s1 = -(flux1[2,0]-flux1[3,0])/(flux1[2,1]-flux1[3,1])
        d11 = np.sqrt((flux1[2,0]-posr)**2 + (flux1[2,1]-posz)**2)
        d12 = np.sqrt((flux1[3,0]-posr)**2 + (flux1[3,1]-posz)**2)
        s2 = -(flux2[2,0]-flux2[3,0])/(flux2[2,1]-flux2[3,1])
        d21 = np.sqrt((flux2[2,0]-posr)**2 + (flux2[2,1]-posz)**2)
        d22 = np.sqrt((flux2[3,0]-posr)**2 + (flux2[3,1]-posz)**2)
        s3 = -(flux1[1,0]-flux1[2,0])/(flux1[1,1]-flux1[2,1])
        d31 = np.sqrt((flux1[1,0]-posr)**2 + (flux1[1,1]-posz)**2)
        d32 = np.sqrt((flux1[2,0]-posr)**2 + (flux1[2,1]-posz)**2)
        s4 = -(flux2[1,0]-flux2[2,0])/(flux2[1,1]-flux2[2,1])
        d41 = np.sqrt((flux2[1,0]-posr)**2 + (flux2[1,1]-posz)**2)
        d42 = np.sqrt((flux2[2,0]-posr)**2 + (flux2[2,1]-posz)**2)
        s5 = -(flux1[3,0]-flux1[4,0])/(flux1[3,1]-flux1[4,1])
        d51 = np.sqrt((flux1[3,0]-posr)**2 + (flux1[2,1]-posz)**2)
        d52 = np.sqrt((flux1[4,0]-posr)**2 + (flux1[3,1]-posz)**2)
        s6 = -(flux2[3,0]-flux2[4,0])/(flux2[3,1]-flux2[4,1])
        d61 = np.sqrt((flux2[3,0]-posr)**2 + (flux2[3,1]-posz)**2)
        d62 = np.sqrt((flux2[4,0]-posr)**2 + (flux2[4,1]-posz)**2)
        #Computes the weighted averages of normal slopes
        m = ((2*s1/(d11+d12))+(2*s2/(d21+d22))+ (2*s3/(d31+d32))+ \
             (2*s4/(d41+d42))+(2*s5/(d51+d52))+(2*s6/(d61+d62)))/ \
            ((2/(d11+d12))+(2/(d21+d22))+ (2/(d31+d32))+ \
             (2/(d41+d42))+(2/(d51+d52))+(2/(d61+d62)))
        
        norm_dir_rz = [1,m]
        norm_dir_rz = norm_dir_rz/np.linalg.norm(norm_dir_rz)
        
        norm_dir_rz = np.append(norm_dir_rz, np.arctan2(pos[1],pos[0]))
        norm_dir_rz = norm_dir_rz/np.linalg.norm(norm_dir_rz)
        b_norm_f_xyz = [norm_dir_rz[0]*np.cos(norm_dir_rz[2]),\
                        norm_dir_rz[0]*np.sin(norm_dir_rz[2]),\
                        norm_dir_rz[1]]
        b_norm_f_xyz = b_norm_f_xyz/np.linalg.norm(b_norm_f_xyz)
        # r = np.linspace(168,170,num = 100)
        # z = (r-posr)*m + posz
        # plt.close("all")
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_aspect('equal', adjustable='box')
        # ax.scatter(pos_rz[0][0],pos_rz[0][1])
        # ax.scatter(flux2[1:,0],flux2[1:,1])
        # ax.scatter(flux1[1:,0],flux1[1:,1])
        # ax.plot(r,z)
        # plt.ylim((9.5,13.5))
        # plt.xlim((175,179))       
        # plt.show()
        # Calculates the normal in plane of curvature of field line
        b_rc = (pos - Bfield_pos_arr[i-1])+ (pos-Bfield_pos_arr[i+1]) 
        
        b_rc = b_rc/np.linalg.norm(b_rc)
        b_norm = b_rc/(abs(np.linalg.norm(Bfield_pos_arr[i-1]-Bfield_pos_arr[i+1]))) + b_norm_f_xyz
        b_norm =  b_norm_f_xyz#b_norm/np.linalg.np.linalg.norm(b_norm)
        b_binorm = np.cross(b_norm,b_dir)
        b_binorm = b_binorm/np.linalg.norm(b_binorm)
        Bfield_coord_arr[i]+=[b_dir,b_norm,b_binorm]
    return Bfield_pos_arr, Bfield_coord_arr

from scipy import spatial
@conditional_numba(skip_numba=False)    
def Bfit_data_old(ilos, photons_los, spec_los_indices,ri,zi,phi,Bfield_pos,B_coord, PSFres,PSFsize):
    PSFpnts = np.zeros((PSFres,PSFres))
    pnts_raw = np.zeros((len(spec_los_indices[ilos,:].T),2))
    val_raw = np.zeros(len(spec_los_indices[ilos,:].T))
    for i, pnt in enumerate(spec_los_indices[ilos,:].T):
        if pnt.all() == 0.: 
            continue
        # print("I'm here 2")
        if photons_los[pnt[0],pnt[1],pnt[2]] == 0.0:
            continue
        # print("I'm here 3")
        # print(pnt)
        # print(ri[pnt[0]])
        # print(np.cos(phi[pnt[2]]))
        # print(np.sin(phi[pnt[2]]))
        # print(zi[pnt[1]])
        xyz_pnt = np.array([ri[pnt[0]]*np.cos(phi[pnt[2]]),ri[pnt[0]]*np.sin(phi[pnt[2]]),zi[pnt[1]]])
        mindist = 1e9
        j_min = 0 
        for j,f_line_pos in enumerate(Bfield_pos): #Finds closest point on field line
            if j == 0 or (f_line_pos == Bfield_pos[-1]).all():
                continue
            # normdist = np.dot(B_coord[j,1],xyz_pnt-f_line_pos)
            # binormdist =np.dot(B_coord[j,2],xyz_pnt-f_line_pos)
            # dist = np.sqrt(normdist**2+binormdist**2)
            dist = np.linalg.norm(f_line_pos - xyz_pnt)
            if dist < mindist:
                mindist = dist
                j_min = j
        normdistmin = np.dot(B_coord[j_min,1],xyz_pnt-Bfield_pos[j_min])
        norm_j = int(np.floor(normdistmin*(PSFres/PSFsize))) + int(PSFres/2)
        binormdistmin = np.dot(B_coord[j_min,2],xyz_pnt-Bfield_pos[j_min])
        binorm_k = int(np.floor(binormdistmin*(PSFres/PSFsize))) + int(PSFres/2)
        # plt.scatter(norm_j,binorm_k)
        pnts_raw[i,:] = [normdistmin, binormdistmin]
        val_raw[i] = photons_los[pnt[0],pnt[1],pnt[2]]
        PSFpnts[binorm_k,norm_j] += photons_los[pnt[0],pnt[1],pnt[2]]
    return PSFpnts, pnts_raw, val_raw


@conditional_numba(skip_numba=True)    
def Bfit_data(photons_los, los_indices,ri,zi,phi,Bfield_pos,B_coord, PSFres,PSFsize):
    PSFpnts = np.zeros((PSFres,PSFres))
    pnts_raw = np.zeros((len(los_indices.T),2))
    val_raw = np.zeros(len(los_indices.T))
    for i, pnt in enumerate(los_indices.T):
        # print(pnt)
        if pnt.all() == 0.: 
            continue
        # print("I'm here 2")
        if photons_los[pnt[0],pnt[1],pnt[2]] == 0.0:
            continue
        # print("I'm here 3")
        # print(pnt)
        # print(ri[pnt[0]])
        # print(np.cos(phi[pnt[2]]))
        # print(np.sin(phi[pnt[2]]))
        # print(zi[pnt[1]])
        xyz_pnt = np.array([ri[pnt[0]]*np.cos(phi[pnt[2]]),ri[pnt[0]]*np.sin(phi[pnt[2]]),zi[pnt[1]]])
        mindist = 1e9
        j_min = 0 
        for j,f_line_pos in enumerate(Bfield_pos): #Finds closest point on field line
            if j == 0 or (f_line_pos == Bfield_pos[-1]).all():
                continue
            # normdist = np.dot(B_coord[j,1],xyz_pnt-f_line_pos)
            # binormdist =np.dot(B_coord[j,2],xyz_pnt-f_line_pos)
            # dist = np.sqrt(normdist**2+binormdist**2)
            dist = np.linalg.norm(f_line_pos - xyz_pnt)
            if dist < mindist:
                mindist = dist
                j_min = j
        normdistmin = np.dot(B_coord[j_min,1],xyz_pnt-Bfield_pos[j_min])
        norm_j = int(np.floor(normdistmin*(PSFres/PSFsize))) + int(PSFres/2)
        binormdistmin = np.dot(B_coord[j_min,2],xyz_pnt-Bfield_pos[j_min])
        binorm_k = int(np.floor(binormdistmin*(PSFres/PSFsize))) + int(PSFres/2)
        # plt.scatter(norm_j,binorm_k)
        pnts_raw[i,:] = [normdistmin, binormdistmin]
        val_raw[i] = photons_los[pnt[0],pnt[1],pnt[2]]

        PSFpnts[binorm_k,norm_j] += photons_los[pnt[0],pnt[1],pnt[2]]
        # return
    return PSFpnts, pnts_raw, val_raw

# photons = np.sum(spec["photon_origin"],axis=(0)) # in grid3d r,z,phi
from pyfidasim.toolbox import load_dict
spec = load_dict("C:/UW_Projects/pyfidasim_savefiles/spec.pkl")
fields = load_dict("C:/UW_Projects/pyfidasim_savefiles/fields.pkl")
grid3d = load_dict("C:/UW_Projects/pyfidasim_savefiles/grid3d.pkl") 
nbi = load_dict("C:/UW_Projects/pyfidasim_savefiles/nbi.pkl")
PSF = load_dict("C:/UW_Projects/pyfidasim_savefiles/PSF.pkl")

# photons = np.sum(spec["photon_origin"], axis =0)
# photons_los= photons[ilos,:,:,:]
# los_grid_interp = np.array(np.where(photons[ilos,:,:,:]!=0))

photons = np.sum(spec["photon_origin"], axis =0)
photons_los=np.sum(photons[np.where(PSF['los_image_arr']==0),:,:], axis=(0,1))
los_grid_interp = np.array(np.where(photons_los!=0))

isurf = np.argmin(np.abs(fields['s_surf'] - 1.)) #Look for last close flux surf
ri = np.linspace(grid3d["Rmin"],grid3d["Rmax"],num = grid3d["nR"])
zi = np.linspace(grid3d["Zmin"],grid3d["Zmax"],num = grid3d["nZ"])
phi = np.linspace(grid3d["phimin"],grid3d["phimax"],num = grid3d["nphi"])
flux_pos = np.unravel_index(np.argmax(photons_los),photons_los.shape) #Finds greatest density position
xyz_start = PSF['image_pos'][0]
Bfield_pos, B_coord = B_coord_Tform(xyz_start, grid3d, fields)
dnorm = np.sqrt(grid3d["dR"]**2+grid3d["dZ"]**2)
dbinorm = dnorm
PSFsize = 50 #cm
PSFres = int(PSFsize/grid3d['dR'])
# print("I'm here 1")

fig_3d = plt.figure()
ax1 = fig_3d.add_subplot(111,projection='3d')
ax1.plot3D(Bfield_pos[:,0],Bfield_pos[:,1],Bfield_pos[:,2], color='r', lw=1.)
dl = 1.

for pnt_i, pnt in enumerate(Bfield_pos):
    if pnt_i % 100 ==0:
        B_dir = np.array([pnt, pnt+10*B_coord[pnt_i,0,:]])
        B_norm = np.array([pnt, pnt+10*B_coord[pnt_i,1,:]])
        B_binorm = np.array([pnt, pnt+10*B_coord[pnt_i,2,:]])
        ax1.plot3D(B_dir[:,0],B_dir[:,1],B_dir[:,2], color= "grey")
        ax1.plot3D(B_norm[:,0],B_norm[:,1],B_norm[:,2], color= "black")
        ax1.plot3D(B_binorm[:,0],B_binorm[:,1],B_binorm[:,2], color= "orange")
idx = np.where(spec["grid_cell_crossed_by_los"]==True)
idx = np.array(idx).T
rr = (idx[:,0]*grid3d["dR"]+grid3d["Rmin"])
zz = (idx[:,1]*grid3d["dZ"]+grid3d["Zmin"])
pp = (idx[:,2]*grid3d["dphi"]+grid3d["phimin"])

xx = rr*np.cos(pp)
yy = rr*np.sin(pp)
ax1.plot3D(xx,yy,zz, color = "blue")
# plot shape for the selected phi_range
isurf = np.argmin(np.abs(fields['s_surf'] - 1.))
for i, phi_val in enumerate(np.arange(grid3d['phimin'] - grid3d['rotate_phi_grid'],
                      grid3d['phimax'] - grid3d['rotate_phi_grid'], 0.01)):
    if i % 5 !=0:
        continue
    X = fields['Rsurf'][isurf,0, :] * np.cos(phi_val)
    Y = fields['Rsurf'][isurf,0, :] * np.sin(phi_val)
    ax1.plot3D(X, Y, fields['Zsurf'][isurf,0, :], color='k', lw=1., alpha=0.5)


ax1.axes.set_xlim3d(left=50, right=200) 
ax1.axes.set_ylim3d(bottom=-50, top=-200) 
ax1.axes.set_zlim3d(bottom=-75, top=75) 

PSFpnts, pnts_raw, val_raw = Bfit_data(photons_los, los_grid_interp, ri, zi, phi, Bfield_pos, B_coord, PSFres, PSFsize)

# from scipy import interpolate
# PSFres = 100
# PSFsize = 5 #cm
# grid_n = np.linspace(-PSFsize/2,PSFsize/2, num = PSFres)
# grid_b = np.linspace(-PSFsize/2,PSFsize/2, num = PSFres)
# grid_n,grid_b = np.meshgrid(grid_n, grid_b)
# pnts_raw = pnts_raw
# PSFinterp = interpolate.griddata(pnts_raw, val_raw, (grid_n,grid_b), method="cubic")



# for i, pnt in enumerate(spec["los_grid_intersection_indices"][ilos,:].T):
#     if pnt.all() == 0.: 
#         continue
#     # print("I'm here 2")
#     if photons_los[pnt[0],pnt[1],pnt[2]] == 0.0:
#         continue
#     # print("I'm here 3")
#     # print(pnt)
#     # print(ri[pnt[0]])
#     # print(np.cos(phi[pnt[2]]))
#     # print(np.sin(phi[pnt[2]]))
#     # print(zi[pnt[1]])
#     xyz_pnt = [ri[pnt[0]]*np.cos(phi[pnt[2]]),ri[pnt[0]]*np.sin(phi[pnt[2]]),zi[pnt[1]]]
#     mindist = 1e9
#     j_min = 0 
#     for j,f_line_pos in enumerate(Bfield_pos): #Finds closest point on field line
#         if j == 0 or (f_line_pos == Bfield_pos[-1]).all():
#             continue
#         # normdist = np.dot(B_coord[j,1],xyz_pnt-f_line_pos)
#         # binormdist =np.dot(B_coord[j,2],xyz_pnt-f_line_pos)
#         # dist = np.sqrt(normdist**2+binormdist**2)
#         dist = np.linalg.np.linalg.norm(f_line_pos - xyz_pnt)
#         if dist < mindist:
#             mindist = dist
#             j_min = j
#     normdistmin = np.dot(B_coord[j_min,1],xyz_pnt-Bfield_pos[j_min])
#     norm_j = int(np.floor(normdistmin*(PSFres/PSFsize))) + int(PSFres/2)
#     binormdistmin = np.dot(B_coord[j_min,2],xyz_pnt-Bfield_pos[j_min])
#     binorm_k = int(np.floor(binormdistmin*(PSFres/PSFsize))) + int(PSFres/2)
#     PSFpnts[binorm_k,norm_j] += photons_los[pnt[0],pnt[1],pnt[2]]

plt.figure()
plt.imshow(PSFpnts)
plt.xlabel("Normal distance [cm] " + str(np.round((PSFsize/PSFres),3)) +" cm/pixel")
plt.ylabel("Binormal distance [cm] " + str(np.round((PSFsize/PSFres),3)) +" cm/pixel")
# plt.title("LOS = "+ str(ilos))
norm = np.linspace(-PSFsize/2,PSFsize/2,num =PSFres)
binorm = np.linspace(-PSFsize/2,PSFsize/2,num=PSFres)
fig, ax = plt.subplots()
c = ax.pcolor(norm, binorm, PSFpnts, cmap='plasma')
ax.set_xlabel("Normal distance [cm]")
ax.set_ylabel("Binormal distance [cm]")
ax.set_xlim([norm.min(),norm.max()])
ax.set_ylim([binorm.min(),binorm.max()])
ax.set_title("2D PSF (Beam L D-alpha) Full")
ax.set_aspect('equal', adjustable='box')
fig.colorbar(c, ax=ax, label = "Photon Count "+ r'$10^{18}$ph/(s sr nm m$^2$)')



N = PSFres
d = grid3d['dR']
nnpoints = PSFres
window_n = np.hanning(N)
window_b = np.hanning(N)
window_n,window_b = np.meshgrid(window_n,window_b) 
norm_k = np.fft.fftshift(2*np.pi*np.fft.fftfreq(N,d=d))
binorm_k = np.fft.fftshift(2*np.pi*np.fft.fftfreq(N,d=d))
fig, ax = plt.subplots()
FFT = np.abs(np.fft.fftshift(np.fft.fft2(PSFpnts*window_n*window_b, s =[nnpoints,nnpoints], norm = "ortho")))
norm_FFT =FFT/np.max(FFT)
c = ax.pcolor(norm_k, binorm_k,norm_FFT, cmap = 'plasma')

ax.set_xlabel("k_norm [cm-1]")
ax.set_ylabel("k_binormal [cm-1]")
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
ax.set_title("2D STF (Beam L D-alpha) Full")
ax.set_aspect('equal', adjustable='box')
fig.colorbar(c, ax=ax, label = "Normalised Power")



fig, ax = plt.subplots(figsize=(5., 5.))
from mpl_toolkits.axes_grid1 import make_axes_locatable
norm_PSFpnts = PSFpnts/np.max(PSFpnts)
norm_PSFpnts[norm_PSFpnts<1e-3] = np.nan
ax.pcolor(norm,binorm,norm_PSFpnts,cmap="plasma")
ax.set_ylabel("Binormal distance [cm]")
ax.set_xlim([norm.min(),norm.max()])
ax.set_ylim([binorm.min(),binorm.max()])
ax.axvline(x=0, color ="red")
ax.axvline(x=-3, color ="red")
ax.axvline(x=3, color ="red")
ax.set_title("2D PSF (Beam L D-alpha) Full")
ax.set_aspect('equal', adjustable='box')
divider = make_axes_locatable(ax)
ax_x = divider.append_axes("bottom", 0.6, pad=0.1, sharex=ax)
ax_y = divider.append_axes("right", 0.6, pad=0.1, sharey=ax)
ax_x.set_xlabel("Normal distance [cm]")
ax_x.xaxis.set_tick_params(labelbottom=True)
ax_y.yaxis.set_tick_params(labelleft=False)
norm = np.linspace(-PSFsize/2,PSFsize/2,num =PSFres)
binorm = np.linspace(-PSFsize/2,PSFsize/2,num=PSFres)
vb_data_x =np.sum(PSFpnts,axis=0)
vb_data_x= vb_data_x/np.max(vb_data_x)
vb_data_y =np.sum(PSFpnts,axis=1)
vb_data_y= vb_data_y/np.max(vb_data_y)
# ax_x.plot(ri, vb_data, label = "FWHM="+str(np.round(FWHM(ri,vb_data),3)))
ax_x.plot(norm, vb_data_x,label = "FWHM: "+str(np.round(FWHM(norm,vb_data_x),3)))
ax_y.plot(vb_data_y,binorm, label = "FWHM:\n"+str(np.round(FWHM(binorm,vb_data_y),3)))
ax_x.legend(loc="upper left")
ax_y.legend()
leg = ax_x.legend(handlelength=0, handletextpad=0, fancybox=False, frameon =True)
# for item in leg.legendHandles:
#     item.set_visible(False)
# leg = ax_y.legend(handlelength=0, handletextpad=0, fancybox=False, frameon =False)
# for item in leg.legendHandles:
#     item.set_visible(False)   
ax_x.set_yticks([0, 0.5, 1])
ax_y.set_xticks([0, 0.5, 1])
plt.show()


# for ilos in range(np.shape(photons)[0]):
#     photons_los = photons[ilos,:,:,:]
#     isurf = np.argmin(np.abs(fields['s_surf'] - 1.)) #Look for last close flux surf
#     ri = np.linspace(grid3d["Rmin"],grid3d["Rmax"],num = grid3d["nR"])
#     zi = np.linspace(grid3d["Zmin"],grid3d["Zmax"],num = grid3d["nZ"])
#     phi = np.linspace(grid3d["phimin"],grid3d["phimax"],num = grid3d["nphi"])
#     flux_pos = np.unravel_index(np.argmax(photons_los),photons_los.shape)
#     xyz_start = [ri[flux_pos[0]]*np.cos(phi[flux_pos[2]]),ri[flux_pos[0]]*np.sin(phi[flux_pos[2]]),zi[flux_pos[1]]]
#     B_coords = B_coord_Tform(xyz_start, grid3d, fields)
#     for i, pnt in enumerate(spec["los_grid_intersection_indices"][ilos,:]):
#         if pnt.all() == 0. : 
#             continue
#         xyz_pnt = [ri[pnt[0]]*np.cos(phi[pnt[2]]),ri[pnt[0]]*np.sin(phi[pnt[2]]),zi[pnt[1]]]
#         mindist = 1e9
#         for j,f_line_pos in enumerate(Bfield_pos_arr):
#             if j == 0 or (f_line_pos == Bfield_pos_arr[-1]).all():
#                 continue
#             normdist = np.dot(Bfield_coord_arr[j,1],xyz_pnt-f_line_pos)
#             binormdist =np.dot(Bfield_coord_arr[j,2],xyz_pnt-f_line_pos)
#             dist = np.sqrt(normdist**2+binormdist**2)
#             if dist < mindist:
#                 mindist = dist
#         param += photons_los[pnt[0],pnt[1],pnt[2]]*mindist
#         print(param)