# The following are examples of different machines and setups. They will not run correctly
# unless you have the required files. Therefore it is advised to use these as templates
# for your own simulations for real input files.

import numpy as np
import matplotlib as matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
plt.close('all')

# TRANSP setup for DIII-D 
fidasim = {}
fidasim['transp'] = True
fidasim['time'] = 1.9
fidasim['runid'] = '179854Z03'
fidasim['directory'] =  'Data/'
fidasim['phi_ran'] = [1.65*np.pi, 1.85*np.pi]
fidasim['r_ran'] =  [215,240]
fidasim['z_ran'] =  [-5,5]
fidasim['grid_drz'] = 2.0
fidasim['Bt_sign'] = -1
fidasim['Ip_sign'] = +1
fidasim['prof_plot'] = True

# Atomic physics setup
fidasim['nbi_mass'] = 2.0 # Deuterium
fidasim['ion_mass'] = 2.0 # Deuterium
fidasim['nbi_source_sel'] = [3,4,5,6]
fidasim['dlam'] = 0.01

# Simulation setup
fidasim['nmarker'] = 1000

# ---- Extended emission ----
# fidasim['calc_extended_emission'] = True
# fidasim['spectrum_extended'] = True
# fidasim['transition'] = [4,2]

# Option 1: ---- Single DIII-D LOS ----
fidasim['los_pos'] = [[268.29427061, -78.27024486,  14.98478096]]
fidasim['los_vec'] = [[-0.80990582,-0.57922626,-0.09246356]]

# Option 2: ---- Full Lens of DIII-D LOS (sans lensopt) ----
# import pyfidasim.PSF as PSF
# fidasim['image_pos'] =   np.array([[1.374844472100626831e+02,-1.717498959980588893e+02,0.000000000000000000e+00]]) # position of beam_crossing
# collection_lens_center = np.array([268.0147, -76.0752, 15.0865]) # Center of BES collection optics window
# collection_lens_radius = 11.4194  # cm
# n_los_orig = 120
# n_image = fidasim['image_pos'].shape[0]
# fidasim['nlos'] = n_los_orig*n_image

# fidasim['image_vec'] = np.zeros([n_image,3])
# fidasim['los_pos'] = np.zeros([fidasim['nlos'],3])
# fidasim['los_vec'] = np.zeros([fidasim['nlos'],3])
# fidasim['los_image_arr'] = np.zeros(fidasim['nlos']).astype(int)
# for ii, im in enumerate(fidasim['image_pos']):
#     image_vec = (im-collection_lens_center)/np.linalg.norm(im-collection_lens_center)
#     los_pos = PSF.blur_image(n_los_orig,[[0.0,0.0,0.0]],collection_lens_radius,np.array([image_vec]))[0,:,:]+collection_lens_center
#     los_vec = np.array([(im-pos)/np.linalg.norm(im-pos) for pos in los_pos])
#     nlos = los_pos.shape[0]
#     fidasim['image_vec'][ii,:] = image_vec
#     fidasim['los_pos'][ii:ii+n_los_orig,:] = los_pos
#     fidasim['los_vec'][ii:ii+n_los_orig,:] = los_vec
#     fidasim['los_image_arr'][ii:ii+n_los_orig] = ii

# Option 3: ---- Full lens of DIII-D LOS (with lensopt) ---- 
# R = [220.]  # cm
# Z = [0.]  # cm
# viewed_beam = ['15L']
# num_horizontal_grid_points = [12,12]
# num_vertical_grid_points = [12,12]
# n_image = len(R)
# fidasim["image_pos"] = np.zeros((n_image,3))
# fidasim['image_vec'] = np.zeros((n_image,3))
# for im in range(n_image):
#     from multiprocessing import freeze_support
#     from lensopt.toolbox_new import set_los_from_data_adv as Lensopt_LOS_setup
#     freeze_support()
#     xyz_loc, los_pos, spec_temp = Lensopt_LOS_setup(R[im], pts_loc='./', num_horizontal_los_points=num_horizontal_grid_points[im],
#                                                       num_vert_los_points=num_vertical_grid_points[im], z=Z[im], beam_choice=viewed_beam[im],
#                                                       plt_vacwin=False, plt_3dlos=False, plt_realtime=False)
#     if im == 0:
#         fidasim['los_pos'] = spec_temp['los_pos']
#         fidasim['los_vec'] = spec_temp['los_vec']
#         fidasim['los_image_arr'] = np.full(spec_temp['nlos'],im)
#         fidasim['nlos'] = spec_temp['nlos']
#     else:
#         fidasim['los_pos'] = np.vstack((fidasim['los_pos'],spec_temp['los_pos']))
#         fidasim['los_vec'] = np.vstack((fidasim['los_vec'],spec_temp['los_vec']))
#         fidasim['los_image_arr'] = np.concatenate((fidasim['los_image_arr'],np.full(spec_temp['nlos'],im)))
#         fidasim['nlos'] = fidasim['nlos']+spec_temp['nlos']
    
#     fidasim["image_pos"][im,:]= xyz_loc
#     fidasim['image_vec'][im,:] = (xyz_loc-los_pos)/np.linalg.norm(xyz_loc-los_pos)

# ---- PSF settings ----
# fidasim['calc_photon_origin'] = True
# fidasim['calc_photon_origin_type'] = False
# fidasim['calc_PSF'] = True
# fidasim['d_fiber'] = 0.1 #1mm diameter fibres
# fidasim['p_fiber'] =[[-(np.sqrt(3)/2), 1.5],[0.,1.0], [(np.sqrt(3)/2), 1.5],\
#                     [-(np.sqrt(3)/2), 0.5],[0.,0.],      [(np.sqrt(3)/2), 0.5],\
#                     [-(np.sqrt(3)/2),-0.5],[0.,-1.0],[(np.sqrt(3)/2),-0.5],\
#                     [-(np.sqrt(3)/2),-1.5], [(np.sqrt(3)/2),-1.5]]
# fidasim['n_point'] = 120
# fidasim['plot_blur_image'] = True
# fidasim['f_lens'] = 40 #cm
# fidasim['n_rand'] = 1000


from pyfidasim.input_prep import input_prep
sim_settings,spec,tables,fields,profiles,nbi,grid3d,ncdf,PSF,fbm = input_prep(fidasim)
# -------------------------------
# -- Plot LOS and NBI -----------
# -------------------------------
fig_geo = plot_geometry_3d(fields, spec, nbi=nbi, grid3d=grid3d, plot_crossed_cells=False)
fig_geo.show() 
# ----------------------------
# -- run pyFIDASIM -----------
# ----------------------------
import time
from pyfidasim.main import calc_attenuation
t1 = time.time()
grid3d,spec=calc_attenuation(sim_settings,profiles,nbi,spec,fields,grid3d,tables,ncdf,PSF,fbm)
t2 = time.time()

print("Time taken: ",(t2-t1))

# -----------------------------
# Plot Beam Emission Spectra
# -----------------------------
if 'intens' in spec:
    fig_spec = plot_spectra_interactive(spec, labels=['full', 'half', 'third'])
    fig_spec.show()

# -----------------------------
# Plot Radial Density Profiles
# -----------------------------
fig_dens = plot_midplane_heatmap(grid3d, fields)
fig_dens.show()

# -----------------------------
# Plot 2D PSF
# -----------------------------
# ri = np.linspace(grid3d["Rmin"],grid3d["Rmax"],num = grid3d["nR"])
# zi = np.linspace(grid3d["Zmin"],grid3d["Zmax"],num = grid3d["nZ"])
# photons = spec["photon_origin"].sum(axis=(0,1,4)) 
# fig, ax = plt.subplots()
# c = ax.pcolor(ri, zi, photons.T,shading='auto')
# ax.set_xlabel("Radial location [cm]")
# ax.set_ylabel("Vertical Location [cm]")
# ax.set_title("2D PSF Control")
# fig.colorbar(c, ax=ax, label = "Normalised Intensity")


fidasim = {}
fidasim['machine'] = 'W7X'
fidasim['drz'] = 1.0
fidasim['phi_ran'] = [0.45*np.pi,0.53*np.pi]
fidasim['grid_drz'] = 1.0
fidasim['ion_mass'] = 1.0 # Hydrogen
fidasim['lambda_min'] = 652.0
fidasim['lambda_max'] = 662.0
fidasim['dlam'] = 0.01
fidasim['los_file'] = 'Data/losInfo_New_op2.txt'
fidasim['los_head'] = 'AEM21_HPPS7:'
fidasim['los_default'] = True
fidasim['nbi_default'] = True
fidasim['nbi_mass'] = 1.0 # Hydrogen
fidasim['u_range'] = [700,900]
fidasim['v_width'] = 120.0
fidasim['w_width'] = 120.0
fidasim['calc_halo'] = True
fidasim['calc_uvw'] = True
fidasim['seed'] = 12345
fidasim["nmarker"] = 10000
fidasim["batch_marker"] = fidasim["nmarker"]

from pyfidasim.input_prep import input_prep
sim_settings,spec,tables,fields,profiles,nbi,grid3d,ncdf,PSF,fbm = input_prep(fidasim)

# Manually only keep NBI source 7
nbi['sources'] = ['Q7']
pop_keys = ("Q1","Q2","Q3","Q4","Q5","Q6","Q8")
for popper in pop_keys:
    nbi.pop(popper)

# -------------------------------
# -- Plot LOS and NBI -----------
# -------------------------------

from pyfidasim.plotting_routines import plot_geometry_3d, plot_spectra_interactive, plot_midplane_heatmap
fig_w7x_geo = plot_geometry_3d(fields, spec, nbi=nbi, grid3d=grid3d)
fig_w7x_geo.show()

# ----------------------------
# -- run pyFIDASIM -----------
# ----------------------------
import time
from pyfidasim.main import calc_attenuation

t1 = time.time()
grid3d,spec=calc_attenuation(sim_settings,profiles,nbi,spec,fields,grid3d,tables,ncdf,PSF,fbm)
t2 = time.time()

print("Time taken: ",(t2-t1))

# -----------------------------
# Plot Beam Emission Spectra
# -----------------------------
if 'intens' in spec:
    fig_spec = plot_spectra_interactive(spec, labels=['full', 'half', 'third'])
    fig_spec.show()

# -----------------------------
# Plot Radial Density Profiles
# -----------------------------
fig_dens = plot_midplane_heatmap(grid3d, fields)
fig_dens.show()
# -----------------------------
# Plot Beam-Path Density Profiles
# -----------------------------
from pyfidasim.plotting_routines import plot_u_density
plot_u_density(grid3d)

# -------------------------------
# -- pyFIDASIM inputs -----------
# -------------------------------

fidasim = {}
fidasim['FIDASIM_check'] = True
fidasim['directory'] = 'comp_data'
fidasim['runid'] = '141648E01'
fidasim['geqdsk'] = 'g141648.00185'
fidasim['grid_drz'] = 1.0
fidasim['u_range'] = [980,1200]
fidasim['v_width'] = 120.0
fidasim['w_width'] = 120.0
fidasim['phi_ran'] = [0.20*np.pi,1.20*np.pi]
fidasim['nmarker'] = 10000
fidasim['seed'] = 12345
fidasim['calc_halo'] = True
fidasim['calc_uvw'] = True
fidasim["batch_marker"] = fidasim["nmarker"]

from pyfidasim.input_prep import input_prep
sim_settings,spec,tables,fields,profiles,nbi,grid3d,ncdf,PSF,fbm = input_prep(fidasim)

# -------------------------------
# -- Plot LOS and NBI -----------
# -------------------------------
fig_geo = plot_geometry_3d(fields, spec, nbi=nbi, grid3d=grid3d, plot_crossed_cells=False)
fig_geo.show()

# ----------------------------
# -- run pyFIDASIM -----------
# ----------------------------
import time

from pyfidasim.main import calc_attenuation

t1 = time.time()
grid3d,spec=calc_attenuation(sim_settings,profiles,nbi,spec,fields,grid3d,tables,ncdf,PSF,fbm)
t2 = time.time()

print("Time taken: ",(t2-t1))

# -----------------------------
# Plot Beam Emission Spectra
# -----------------------------
if 'intens' in spec:
    fig_spec = plot_spectra_interactive(spec, labels=['full', 'half', 'third'])
    fig_spec.show()

# -----------------------------
# Plot Radial Density Profiles
# -----------------------------
fig_dens = plot_midplane_heatmap(grid3d, fields)
fig_dens.show()

# -----------------------------
# Plot Beam-Path Density Profiles
# -----------------------------
from pyfidasim.plotting_routines import plot_u_density
plot_u_density(grid3d)
