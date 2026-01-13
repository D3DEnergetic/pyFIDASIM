import numpy as np
import scipy.constants as consts
import numba
from .spectrum import spectrum, spectrum_other
from .grid3d import (
    get_grid3d_indices,
    get_uvw_grid_indices
)
from copy import deepcopy
from .PSF import PSF
from ._fields import (
    xyz_to_Bxyz,
    get_s_along_path,
    xyz_to_denf
)
from .toolbox import rotate_uvw
from .cr_model import colrad, table_interp
from .mc_fastion import mc_fastion

def conditional_numba(skip_numba=False):
    def decorator(func):
        if not skip_numba:
            return numba.jit(func,cache=True, nopython=True,nogil=True)
        else:
            return func
    return decorator

#@conditional_numba(skip_numba=True)
def pyfidasim_core_routine(
    f_rotate_phi_grid, f_dphi_sym, f_s,
    f_Rmin, f_Rmax, f_R, f_dR, f_nr,
    f_Zmin, f_Zmax, f_Z, f_dZ, f_nz,
    f_phimin, f_phimax, f_phi, f_dphi, f_nphi,
    f_flux_nr, f_flux_ds, f_flux_s_min,
    f_Br, f_Bz, f_Bphi, f_Er,
    g3_rotate_phi_grid, g3_dphi_sym, g3_dvol,
    g3_Rmin, g3_nR, g3_dR,
    g3_Zmin, g3_nZ, g3_dZ,
    g3_phimin, g3_dphi, g3_nphi,
    g3_umin, g3_du, g3_vmin, g3_dv,
    g3_wmin, g3_dw, g3_nu, g3_nv, g3_nw,
    spec_grid_cell_crossed_by_los, spec_dl_per_grid_intersection,
    spec_los_grid_intersection_indices, spec_los_grid_intersection_weight,
    spec_nlos, spec_los_lens, spec_los_vec,
    spec_lambda_min, spec_nlam, spec_dlam,
    spec_sigma_to_pi_ratio,
    spec_output_individual_stark_lines,
    fbm_fbm, fbm_denf, fbm_afbm, fbm_btipsign,
    fbm_emin, fbm_eran, fbm_nenergy, fbm_energy, fbm_dE,
    fbm_pmin, fbm_pran, fbm_npitch, fbm_pitch, fbm_dP,
    prof_s, prof_te, prof_ti, prof_dene, prof_denp,
    prof_denimp, prof_omega, prof_ai,
    tb_levels, tb_energy_ax, tb_temp_ax,
    tb_nimps, tb_zimps, tb_impurities,
    tb_qptable_no_cx, tb_qptable, tb_qetable, tb_qitable,
    tb_einstein, tb_neutrates,
    nbi_vec, nbi_source_position, nbi_rot_inv,
    ikind, factor, states, start_pos, jump,
    step_length, vnorm, vabs, E0, mass_in,
    ncx_iter, seed, calc_spectra,calc_photon_origin,
    calc_photon_origin_type, calc_PSF,
    calc_density, calc_uvw, density_uvw, separate_dcx,
    density, intensity, photon_origin, density1d,
    cdfvars, lambda0, trans, l_to_dwp, spectrum_extended,
    los_image_arr,n_rand,image_pos,image_vec,image_blur,f_lens
):
    mass = mass_in
    step_length_adj = deepcopy(step_length)
    sumstates = np.sum(states, axis=1)
    diagonal_einstein = np.sum(tb_einstein, axis=0)

    halod = np.zeros(start_pos.shape[0], dtype=bool)
    ikind = np.repeat(ikind,start_pos.shape[0])
    vabs = np.full(start_pos.shape[0], vabs)
    v_xyz = vnorm * vabs[:, np.newaxis]
    
    if fbm_afbm > 0:
        fbm_params = {
            'fbm': fbm_fbm, 'afbm': fbm_afbm, 'btipsign': fbm_btipsign,
            'emin': fbm_emin, 'eran': fbm_eran, 'nenergy': fbm_nenergy, 'energy': fbm_energy, 'dE': fbm_dE,
            'pmin': fbm_pmin, 'pran': fbm_pran, 'npitch': fbm_npitch, 'pitch': fbm_pitch, 'dP': fbm_dP
        }
        fields_params = {
            'Br': f_Br, 'Bz': f_Bz, 'Bphi': f_Bphi, 'rotate_phi_grid': f_rotate_phi_grid, 'dphi_sym': f_dphi_sym,
            'R': f_R, 'Rmin': f_Rmin, 'Rmax': f_Rmax, 'dR': f_dR, 'nr': f_nr,
            'Z': f_Z, 'Zmin': f_Zmin, 'Zmax': f_Zmax, 'dZ': f_dZ, 'nz': f_nz,
            'phi': f_phi, 'phimin': f_phimin, 'phimax': f_phimax, 'dphi': f_dphi, 'nphi': f_nphi
        }
    
    for cx_iter in range(ncx_iter):
        if cx_iter > 0:
            if not np.any(halod):
                break
            mass = prof_ai * consts.atomic_mass
            jump = 0.0
            step_length_adj = deepcopy(step_length)/15 #30
            start_pos = start_pos[halod,:]
            vnorm = vnorm[halod,:]
        # Determine Path along the marker's velocity vector
        s_arr_o, xyzc_arr_o, dl_o, first_step_o = get_s_along_path(
            start_pos, vnorm, step_length_adj.astype(np.float32),
            f_rotate_phi_grid, f_dphi_sym, f_s, prof_s,
            f_Rmin, f_Rmax, f_R, f_dR, f_nr,
            f_Zmin, f_Zmax, f_Z, f_dZ, f_nz,
            f_phimin, f_phimax, f_phi, f_dphi, f_nphi,
            jump=jump, 
            seed=np.random.randint(1,high=1E8) if seed >= 0 else seed
        )
        # Count non-zero values in each row of s_arr_o and create mask
        non_zero_counts = np.count_nonzero(xyzc_arr_o[:,0,:], axis=1)
        mask = non_zero_counts >= 2
        if not mask.any():
            break
        if cx_iter > 0:
            halod[halod] = mask
            ikind = ikind[halod]
            start_pos = start_pos[mask,:]
            vnorm = vnorm[mask,:]
            states = states[halod,:]
            vabs = vabs[halod]
            v_xyz = vnorm * vabs[:, np.newaxis]
        else:
            ikind = ikind[mask]
            start_pos = start_pos[mask,:]
            vnorm = vnorm[mask,:]
            states = states[mask,:]
            vabs = vabs[mask]
            v_xyz = vnorm * vabs[:, np.newaxis]
        
        # Extract the number of steps for rows that are kept
        nstep_max = non_zero_counts[mask].max()
        
        # Filter arrays using the mask
        s_arr = s_arr_o[mask, :nstep_max]
        xyzc_arr = xyzc_arr_o[mask, :, :nstep_max]
        dl = dl_o[mask, :nstep_max]
        first_step = first_step_o[mask]
        
        del s_arr_o, xyzc_arr_o, dl_o, first_step_o, mask
        
        # Kinetic Profiles along Path
        te_log = np.log10(np.interp(s_arr, prof_s, prof_te))
        ti = np.interp(s_arr, prof_s, prof_ti)
        ti_log = np.log10(ti)
        tp_log = np.log10(ti / prof_ai)
        s_outside_index = s_arr > prof_s.max()
        dene = np.interp(s_arr, prof_s, prof_dene) # electron density
        denp = np.interp(s_arr, prof_s, prof_denp) # electron density
        
        denimp = np.array([np.interp(s_arr, prof_s, prof_denimp[jj, :]) for jj in range(tb_nimps)])
        denimp = denimp.transpose(1, 0, 2)
        #subtractor = np.tensordot(denimp, tb_zimps, axes=([1], [0]))
        
        # enforce quasi neutrality
        #denp = dene - subtractor
        dene[s_outside_index] = 0.
        denp[s_outside_index] = 0.
        denimp[s_outside_index[:, np.newaxis, :]] = 0.
        
        # consider plasma rotation and calcuate eb_log        
        ## vphi =m/s= rad/s m/rad=  omega * 2piR/2pi = omega*R
        omega = np.interp(s_arr, prof_s, prof_omega) # rad/s
        vrot = np.zeros((np.shape(s_arr)[0],3,nstep_max))
        vnet = np.zeros((np.shape(s_arr)[0],3,nstep_max))
        vphi = omega * np.sqrt(xyzc_arr[:, 0, :]**2 + xyzc_arr[:, 1, :]**2)  # cm/s
        phi = np.arctan2(xyzc_arr[:, 1, :], xyzc_arr[:, 0, :])
        arg = np.pi * 0.5 - phi
        vrot[:, 0, :] = -np.cos(arg) * vphi
        vrot[:, 1, :] = np.sin(arg) * vphi
        vnet = v_xyz[:, :, np.newaxis] - vrot
        vnet_square = np.sum(vnet**2, axis=1) / 1.e4  # [m/s]**2
        eb_log = np.log10(0.5 * consts.atomic_mass * vnet_square / consts.e / 1.e3)  # [keV/amu]
        
        if calc_uvw:
            # Store the beam density on a beam-aligned grid
            uvw_start = rotate_uvw(nbi_rot_inv, start_pos - nbi_source_position)
            uvw_ray = rotate_uvw(nbi_rot_inv, vnorm)
            iu_arr, iv_arr, iw_arr = get_uvw_grid_indices(
                uvw_start, uvw_ray, dl, first_step,
                g3_umin, g3_du, g3_vmin, g3_dv, g3_wmin, g3_dw,
                g3_nu, g3_nv, g3_nw
            )
            
        # Get grid indices of the cylindrical and UVW storage grids
        ir_arr, iz_arr, iphi_arr = get_grid3d_indices(
            xyzc_arr,
            g3_rotate_phi_grid, g3_dphi_sym,
            g3_Rmin, g3_dR, g3_nR,
            g3_Zmin, g3_dZ, g3_nZ,
            g3_phimin, g3_dphi, g3_nphi
        )

        # Density function array calculation
        if fbm_afbm > 0:
            denf_arr = xyz_to_denf(xyzc_arr, fbm_denf, f_rotate_phi_grid, f_dphi_sym,
                                   f_R, f_Rmin, f_dR, f_nr,
                                   f_Z, f_Zmin, f_dZ, f_nz,
                                   f_phi, f_phimin, f_dphi, f_nphi)

        # Time array calculation
        dt_arr = dl / vabs[:, np.newaxis]

        # Initialize non-disabled and halod masks
        dis = np.ones(s_arr.shape[0], dtype=bool)
        halod = np.zeros_like(dis, dtype=bool)
        
        for ii in np.arange(nstep_max): ## loop along track of a given neutral marker
            dis[dis] &= ~(dt_arr[dis,ii]==0)
            dis[dis] = xyzc_arr[dis, 0, ii] != 0
            if not dis.any():
                break
            ir = ir_arr[dis,ii] ## grid3D indices
            iz = iz_arr[dis,ii] ## grid3D indices
            iphi = iphi_arr[dis,ii]  ## grid3D indices
            grid_mask = (
                        (0 <= ir) & (ir <= g3_nR - 1) &
                        (0 <= iz) & (iz <= g3_nZ - 1) &
                        (0 <= iphi) & (iphi <= g3_nphi - 1)
                        )
            sindex = ((s_arr[dis, ii] - f_flux_s_min) / f_flux_ds).astype(np.int64)
            sindex = np.where((sindex < 0) | (sindex >= f_flux_nr), -1, sindex)

            # Read Tables
            ## Proton tables (charge exchange, impact ionization and excitation)
            qp = table_interp(tb_qptable,eb_log[dis,ii],tp_log[dis,ii], \
                                   tb_energy_ax,tb_temp_ax) * denp[dis,ii][:, np.newaxis, np.newaxis] # [1/s]
            ## Electron tables (impact ionization and excitation)
            qe = table_interp(tb_qetable,eb_log[dis,ii],te_log[dis,ii], \
                               tb_energy_ax,tb_temp_ax)* dene[dis,ii][:, np.newaxis, np.newaxis] # [1/s]
            ## Impurity tables (charge exchange, impact ionization and excitation)
            # set up a 2D array for the impurity related rates
            qc = np.zeros((np.shape(ir)[0],tb_nimps,7,6)) ## hardcoded 6 states, the 7th state considers ionization.
            # fill this array with the weighted rates
            for jj in range(tb_nimps):
                qc[:,jj,:] = (table_interp(tb_qitable[jj,:],eb_log[dis,ii],ti_log[dis,ii], \
                               tb_energy_ax,tb_temp_ax) * denimp[dis,jj,:][:,ii][:, np.newaxis, np.newaxis]) # [1/s]

            ## -------------------------------------------------------------
            ## -Determine CHARGE EXCHANGE (CX) and IONIZATION probabiliites-
            ## -------------------------------------------------------------
            ## For this we firstly define the velocity vector. We will either use a thermal velocity:
            '''
            v_xyz_new = np.sqrt(ti[dis,ii][:,np.newaxis] * 1.e3 * consts.e / (prof_ai*consts.atomic_mass)) * 100. * np.random.randn(np.shape(ir)[0],3)  + vrot[dis,:, ii]
            
            ## Check if we should consider a fast-ion velocity vector:
            if fbm_afbm > 0:
                denp_slice = np.copy(denp[dis, ii])
                denf_slice = denf_arr[dis, ii]
                #Correct for quasineutrality
                denp_slice -= np.nan_to_num(denf_arr[dis, ii])
                
                ratio = np.zeros_like(denp_slice)
                
                valid_mask = (denp_slice > 0) & ~np.isnan(denf_slice)
                ratio[valid_mask] = denf_slice[valid_mask] / denp_slice[valid_mask]
                
                fast_ion_mask = ratio > np.random.rand(ratio.shape[0])
                
                if np.any(fast_ion_mask):
                    xyz_for_fast_ion = xyzc_arr[dis, :, ii][fast_ion_mask]
                    
                    fast_ion_velocities = mc_fastion(xyz_for_fast_ion, fields_params, fbm_params)
                    
                    v_xyz_new[fast_ion_mask, :] = fast_ion_velocities
            else:
                fast_ion_mask = np.zeros_like(ir,dtype = bool)
            
            vnet_new= v_xyz[dis,:]-v_xyz_new
            vnet_new_square = np.sum(vnet_new**2,axis=1) / 1.e4  # [m/s]**2
            ## and calculate the relative collision energy with the neutral (NBI or halo neutral)
            eb_nrate_log = np.log10(0.5 * consts.atomic_mass * vnet_new_square / consts.e / 1.e3)  # [keV/amu]
            ## this allows us to infer the neutraliztion rate. Note that we assume Ti=0 which means that only the relative velocity
            ## between the ion and neutral counts -- i.e. this is the bare cross-section (note also the log(-99)~0)
            sigma_v_cx = table_interp(tb_neutrates,eb_nrate_log,np.ones(np.shape(eb_nrate_log)[0])*-99.,tb_energy_ax,tb_temp_ax) # [cm^3/s]
            nrate = sigma_v_cx*denp[dis,ii, np.newaxis, np.newaxis] # [1/s]
            
            '''
            ## -------------------------------------------------------------
            ## -Determine CHARGE EXCHANGE (CX) and IONIZATION probabiliites-
            ## -------------------------------------------------------------
            # 1. Setup initial thermal velocities
            v_xyz_new = np.sqrt(ti[dis,ii][:,np.newaxis] * 1.e3 * consts.e / (prof_ai*consts.atomic_mass)) * 100. * np.random.randn(np.shape(ir)[0],3)  + vrot[dis,:, ii]
            
            # 2. Initialize variables for the whole batch
            nrate = np.zeros((ir.shape[0], 7, 6))
            fast_ion_mask = np.zeros(ir.shape[0], dtype=np.bool_)
            
            # 3. Only process particles inside the grid to prevent boundary crashes
            if np.any(grid_mask):
                # Subset data for grid locations
                v_xyz_sub = v_xyz[dis,:][grid_mask, :]
                v_th_sub = v_xyz_new[grid_mask, :]
                denp_sub = denp[dis, ii][grid_mask]
                
                # Thermal Rates for subset
                vnet_th = v_xyz_sub - v_th_sub
                vnet_th_sq = np.sum(vnet_th**2, axis=1) / 1.e4
                eb_th_log = np.log10(0.5 * consts.atomic_mass * vnet_th_sq / consts.e / 1.e3)
                sig_th = table_interp(tb_neutrates, eb_th_log, np.ones_like(eb_th_log)*-99., tb_energy_ax, tb_temp_ax)
                
                if fbm_afbm > 0:
                    denf_sub = denf_arr[dis, ii][grid_mask]
                    # Fast-Ion Rates for subset
                    v_fa_sub = mc_fastion(xyzc_arr[dis, :, ii][grid_mask], fields_params, fbm_params)
                    vnet_fa = v_xyz_sub - v_fa_sub
                    vnet_fa_sq = np.sum(vnet_fa**2, axis=1) / 1.e4
                    eb_fa_log = np.log10(0.5 * consts.atomic_mass * vnet_fa_sq / consts.e / 1.e3)
                    sig_fa = table_interp(tb_neutrates, eb_fa_log, np.ones_like(eb_fa_log)*-99., tb_energy_ax, tb_temp_ax)
                    
                    # Competitive Ratio calculation
                    ni_th_sub = np.maximum(denp_sub - denf_sub, 0.0)
                    r_th_sub = np.sum(sig_th, axis=(1, 2)) * ni_th_sub
                    r_fa_sub = np.sum(sig_fa, axis=(1, 2)) * denf_sub
                    r_tot_sub = r_th_sub + r_fa_sub
                    
                    # Determine winner for the subset
                    prob_fa_sub = np.divide(r_fa_sub, r_tot_sub, out=np.zeros_like(r_fa_sub), where=r_tot_sub > 0)
                    mask_sub = prob_fa_sub > np.random.rand(len(prob_fa_sub))
                    
                    # Update the global fast_ion_mask and velocities
                    fast_ion_mask[grid_mask] = mask_sub
                    
                    # Velocity update for winning fast neutrals
                    v_res_sub = v_th_sub
                    for k in range(len(mask_sub)):
                        if mask_sub[k]:
                            v_res_sub[k] = v_fa_sub[k]
                    v_xyz_new[grid_mask] = v_res_sub
                    
                    # Combined nrate for attenuation solver
                    nrate[grid_mask] = (sig_th * ni_th_sub[:, np.newaxis, np.newaxis] + 
                                       sig_fa * denf_sub[:, np.newaxis, np.newaxis])
                else:
                    # No Fast Ions case
                    nrate[grid_mask] = sig_th * denp_sub[:, np.newaxis, np.newaxis]
            

            ## ----------------------------------------
            ## ---- determine CX probability ----------
            ## ----------------------------------------
            ## the CX probabiity is obtained by summing all final states and multiplying by dt
            cx_prob=np.sum(nrate,axis=1)*dt_arr[dis,ii, np.newaxis]   # [1/s]
            cx_prob_weighted=cx_prob*states[dis,:]/np.max(states,axis=1)[dis,np.newaxis] ## Now we have to weight the different states
            cx_prob_tot=1.-np.prod(1.-cx_prob_weighted,axis=1)
            
            ## ---------------------------------------
            ## -- determine Ionization probability ---
            ## ---------------------------------------
            ## This part considers all other ionization channels (also CX with impurities). To get the 
            ## ionization rates for main-ions, we use the qptable_no_cx.
            qp_no_cx = table_interp(tb_qptable_no_cx,eb_log[dis,ii],tp_log[dis,ii],tb_energy_ax,tb_temp_ax)*denp[dis,ii][:, np.newaxis, np.newaxis] # [1/s]
            ## The probability for ionization is the
            # first, add the impurity species unrelated part
            i_prob = (qp_no_cx[:,6,:]+qe[:,6,:])
            # and then add the data for the passed impurities
            for jj in range(tb_nimps):
                i_prob += qc[:,jj,6,:]
            i_prob *= dt_arr[dis,ii,np.newaxis]
            i_prob_weighted=i_prob*states[dis,:]/np.max(states[dis,:],axis=1)[:,np.newaxis]
            i_prob_tot=1.-np.prod(1.-i_prob_weighted,axis=1)

            # Convert Bernoulli-in-dt to equivalent exponential rates
            eps = 1e-16
            dtv = dt_arr[dis, ii]  # (K,)
            lam_cx  = -np.log1p(-np.clip(cx_prob_tot, 0.0, 1.0 - eps)) / dtv
            lam_ion = -np.log1p(-np.clip(i_prob_tot, 0.0, 1.0 - eps)) / dtv
            Lam = lam_cx + lam_ion
            
            # Probability at least one event occurs in dt
            p_any = 1.0 - np.exp(-Lam * dtv)
            
            u  = np.random.rand(Lam.shape[0])
            u2 = np.random.rand(Lam.shape[0])
            
            event = u < p_any
            # Conditional probability the event is CX given that "an" event occurred
            # (classical competing exponentials)
            with np.errstate(invalid='ignore', divide='ignore'):
                p_cx_given_any = np.where(Lam > 0.0, lam_cx / Lam, 0.0)
            
            cx_happens_in_this_cell         = event & (u2 < p_cx_given_any)
            ionization_happens_in_this_cell = event & ~ (u2 < p_cx_given_any)
            
            ## ----------------------------------------------------------------
            ## ---------- run COLRAD ------------------------------------------
            ## ----------------------------------------------------------------
            ## fill matrix with exitation and de-excitation rates
            # set up an empty matrix first
            rate_matrix  = np.zeros((np.shape(ir)[0],6,6))
            # fill the matrix with the non-impurity related elements
            rate_matrix += (tb_einstein[np.newaxis,:,:] + qp[:,0:6,:] + qe[:,0:6,:])
            # add the contributions from the different impurities
            for jj in range(tb_nimps):
                rate_matrix += qc[:,jj,0:6,:]
            # now also add the diagonal elements
            diagonal_non_impurity = np.sum(qp, axis=1) + np.sum(qe, axis=1)
            diagonal_impurity = np.sum(qc, axis=(1, 2))
            cx_loss_diagonal = np.sum(nrate, axis=1) 
            
            diagonal = (
                diagonal_einstein[np.newaxis, :] +
                diagonal_non_impurity +
                diagonal_impurity +
                cx_loss_diagonal[:, 0:6]
            )
            rate_matrix[:, np.arange(6), np.arange(6)] = -diagonal
            iflux = np.sum(states[dis,:],axis=1)
            ## run COLRAD
            states[dis,:], dens = colrad(dt_arr[dis,ii], rate_matrix, states[dis,:])
            sumstates = np.sum(states[dis,:],axis=1)
            
            ## now normalize the states
            states[dis,:]*=iflux[:,np.newaxis]/sumstates[:,np.newaxis]
            ## calculate the density assuming that there is no attenuation.
            dens*=iflux[:,np.newaxis]/sumstates[:,np.newaxis]
            
            # Save 1D data (cx losses)
            density1d[:, sindex] += np.sum(dens, axis=1)[np.newaxis, :] * factor

            # Save neutral density
            if calc_density:
                    
                dens_mask = grid_mask & ~cx_happens_in_this_cell & ~ionization_happens_in_this_cell

                np.add.at(
                    density,
                    (ikind[dis][dens_mask], slice(None), ir[dens_mask], iz[dens_mask], iphi[dens_mask]),
                    (dens[dens_mask] / g3_dvol[ir[dens_mask], np.newaxis]) * factor
                )
                
            # Conditional vectorized update for density_uvw
            if calc_uvw:
                iu = iu_arr[dis,ii]
                iv = iv_arr[dis,ii]
                iw = iw_arr[dis,ii]
                uvw_grid_mask = (
                            (0 <= iu) & (iu <= g3_nu - 1) &
                            (0 <= iv) & (iv <= g3_nv - 1) &
                            (0 <= iw) & (iw <= g3_nw - 1) &
                            ~cx_happens_in_this_cell      &
                            ~ionization_happens_in_this_cell
                            )
                
                np.add.at(
                    density_uvw,
                    (ikind[dis][uvw_grid_mask], slice(None), iu[uvw_grid_mask], iv[uvw_grid_mask], iw[uvw_grid_mask]),
                    (dens[uvw_grid_mask] / (g3_du * g3_dv * g3_dw)) * factor
                )

            # Calculate Spectra
            if calc_spectra:
                cross_pos1 = spec_grid_cell_crossed_by_los[ir[grid_mask], iz[grid_mask], iphi[grid_mask]]
                cross_pos = np.copy(grid_mask)
                cross_pos[cross_pos] = cross_pos1
                if cross_pos.any():
                    Bxyz = xyz_to_Bxyz(
                        xyzc_arr[dis, :, ii][cross_pos],
                        f_Br, f_Bz, f_Bphi, f_rotate_phi_grid, f_dphi_sym,
                        f_R, f_Rmin, f_Rmax, f_dR, f_nr,
                        f_Z, f_Zmin, f_Zmax, f_dZ, f_nz,
                        f_phi, f_phimin, f_phimax, f_dphi, f_nphi
                    )
                    
                    combined_condition = (
                        (spec_los_grid_intersection_indices[:, :, 0] == ir[cross_pos, None, None]) &
                        (spec_los_grid_intersection_indices[:, :, 1] == iz[cross_pos, None, None]) &
                        (spec_los_grid_intersection_indices[:, :, 2] == iphi[cross_pos, None, None])
                    )
                    
                    # Contains which lines of sight crossed which ray at what position
                    # Row 0: Index into the subset of particles (compressed by cross_pos)
                    # Row 1: LOS index
                    # Row 2: Grid cell index
                    cross_indices = np.array(np.where(combined_condition))
                    
                    # Get the particle type (ikind) for every particle in the compressed subset
                    indices_dis = np.where(dis)[0]
                    indices_crossing = indices_dis[cross_pos] 
                    ikinds_crossing = ikind[indices_crossing]

                    if not spectrum_extended: # Traditional way
                        ## get the electric field
                        Exyz = np.zeros(3)
                        photons_all = dens[cross_pos, 2][cross_indices[0,:]] * \
                                      tb_einstein[1, 2] / \
                                      g3_dvol[ir[cross_pos][cross_indices[0,:]]] * factor 

                        unique_kinds = np.unique(ikinds_crossing)
                        
                        for k in unique_kinds:
                            # k_mask shape: (N_particles_crossing,)
                            k_mask = (ikinds_crossing == k)
                            interaction_mask = k_mask[cross_indices[0, :]]
                            
                            if not np.any(interaction_mask):
                                continue

                            indices_k = cross_indices[:, interaction_mask]
                            photons_k = photons_all[interaction_mask]
                            
                            intensity[k, :, :, :] += spectrum(
                                mass, lambda0, 
                                v_xyz[dis,:][cross_pos,:]/100.,
                                indices_k,
                                Exyz, 
                                Bxyz,
                                spec_nlos, spec_los_vec, spec_lambda_min, spec_nlam, spec_dlam,
                                spec_dl_per_grid_intersection, spec_sigma_to_pi_ratio,
                                spec_output_individual_stark_lines, 
                                photons_k
                            )
                    
                    else: # Other emission method (Extended)
                        num_orbital = int(((trans / 3) - 1) / 2)
                        rel_indices = np.where(cross_pos)[0]
                        for idx_in_subset, (rel_idx, batch_idx) in enumerate(zip(rel_indices, indices_crossing)):

                            this_k = ikind[batch_idx]
                            photons = dens[rel_idx, num_orbital] * tb_einstein[1, num_orbital] / \
                                      g3_dvol[ir[rel_idx]] * factor

                            # Find which LOS/Cells this specific particle intersects
                            subset_interactions = (cross_indices[0, :] == idx_in_subset)
                            
                            if not np.any(subset_interactions):
                                continue
                                
                            indices_for_func = cross_indices[1:, subset_interactions]

                            intensity[this_k, :, :, :] += spectrum_other(
                                cdfvars, lambda0, trans, l_to_dwp,
                                v_xyz[batch_idx,:] / 100.0,
                                indices_for_func.T, 
                                Bxyz[idx_in_subset,:],
                                spec_sigma_to_pi_ratio,
                                spec_dl_per_grid_intersection, spec_lambda_min,
                                spec_nlos, spec_los_vec, spec_nlam, spec_dlam,
                                spec_output_individual_stark_lines
                            ) * photons
            
                    # Calculate photon origin positions if required
                    ## Photon origin is used to show the spatial resolution of a given BES diagnostic. The positions of the 
                    ## photon emission are saved into the "photon_origin" array.
                    if calc_photon_origin:
                        if calc_PSF:
                            num_orbital = int(((trans / 3) - 1) / 2)  # Determine top orbital energy level
                            photon_origin = PSF(photon_origin,calc_photon_origin_type,dis,cross_pos,cross_indices,dens,tb_einstein,g3_dvol,ir,factor, ii,
                                          rate_matrix,v_xyz,g3_rotate_phi_grid, g3_dphi_sym,g3_Rmin, g3_dR, g3_nR, g3_Zmin, g3_dZ, g3_nZ,
                                          g3_phimin, g3_dphi, g3_nphi,ikind[dis][cross_pos],xyzc_arr,los_image_arr,n_rand,image_pos,image_vec,image_blur,f_lens)
                        else:
                            ilos_flat = cross_indices[1,:]
                            if calc_photon_origin_type:
                                
                                np.add.at(
                                    photon_origin,
                                    (ikind[dis][cross_pos], ilos_flat, ir[cross_pos], iz[cross_pos], iphi[cross_pos]),
                                    photons
                                )
                            else:
                                np.add.at(
                                    photon_origin,
                                    (0, ilos_flat, ir[cross_pos], iz[cross_pos], iphi[cross_pos]),
                                    photons
                                )
            
            
            if np.any(cx_happens_in_this_cell) and ncx_iter > 1:
                indices_dis = np.where(dis)[0]
                indices_subset = indices_dis[cx_happens_in_this_cell]
                cx_flux = np.einsum('ijk,ik->ij', nrate[cx_happens_in_this_cell], dens[cx_happens_in_this_cell])
                states[indices_subset] = (cx_flux[:, 0:6] / np.sum(cx_flux[:, 0:6], axis=1, keepdims=True)) \
                                 * sumstates[cx_happens_in_this_cell, np.newaxis]
                start_pos[indices_subset, :] = xyzc_arr[indices_subset, :, ii] \
                                 - 0.5 * dl[indices_subset, ii, np.newaxis] * vnorm[indices_subset, :] \
                                 + dt_arr[indices_subset, ii, np.newaxis] * v_xyz[indices_subset, :]
                v_xyz[indices_subset] = v_xyz_new[cx_happens_in_this_cell]
                vabs[indices_subset] = np.linalg.norm(v_xyz[indices_subset],axis=1)
                vnorm[indices_subset] = v_xyz[indices_subset] / vabs[indices_subset, np.newaxis]
                halod[indices_subset] = True
                if separate_dcx and cx_iter==0:
                    ikind[indices_dis[cx_happens_in_this_cell & ~fast_ion_mask]] = 4
                else:
                    ikind[indices_dis[cx_happens_in_this_cell & ~fast_ion_mask]] = 3
                if fbm_afbm > 0:
                    if separate_dcx:
                        ikind[indices_dis[cx_happens_in_this_cell & fast_ion_mask]] = 5
                    else:
                        ikind[indices_dis[cx_happens_in_this_cell & fast_ion_mask]] = 4

            
            
            index = (np.sum(states[dis,:],axis=1) > 6E-14)
            index2 = ~ionization_happens_in_this_cell & ~cx_happens_in_this_cell
            halod[dis] &= index 
            dis[dis] &= index & index2
            
    return(density,density_uvw,intensity,photon_origin,density1d) # this marker leaves the plasma. stop the cx-loop


