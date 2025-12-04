import sys
import os
import numpy as np
import time

# GUI Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTreeWidget, QTreeWidgetItem, QHeaderView, QLineEdit, QDoubleSpinBox, 
    QSpinBox, QCheckBox, QPushButton, QLabel, QTabWidget, QSplitter, 
    QProgressBar, QMessageBox, QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Matplotlib Integration
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# PyFIDASIM Imports
try:
    from generate_synthetic_inputs import get_default_config, build_simulation_data
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from generate_synthetic_inputs import get_default_config, build_simulation_data

from pyfidasim.cr_model import load_tables
from pyfidasim.grid3d import define_grid3d
from pyfidasim.los import grid_intersections
from pyfidasim.main import calc_attenuation
from pyfidasim.plotting_routines import (
    plot_geometry_3d, plot_midplane_heatmap, plot_spectra_interactive, 
    plot_profiles, plot_magnetic_equilibrium
)

# =============================================================================
# MASTER DESCRIPTION DICTIONARY
# =============================================================================
TOOLTIPS = {
    # -------------------------------------------------------------------------
    # USER INPUTS (Normally set in input_prep.py)
    # -------------------------------------------------------------------------
    
    # --- SIMULATION SETTINGS ---
    'nmarker': "Int: Number of Monte-Carlo markers. Higher = better precision/spectra, but slower execution (Default: 10k-100k).",
    'calc_halo': "Bool: If True, calculates Halo neutrals (thermal neutrals created by charge exchange with the beam).",
    'verbose': "Bool: If True, prints simulation progress steps and details to the console.",
    'calc_photon_origin': "Bool: Stores the 3D location (R,Z,Phi) where every photon was emitted (useful for checking spatial resolution).",
    'calc_photon_origin_type': "Bool: If True, separates photon origin data by type (Full/Half/Third/Halo) in the output array.",
    'calc_PSF': "Bool: If True, calculates Point Spread Function (finite lifetime, fiber size, light cone geometry).",
    'calc_density': "Bool: Stores beam/halo neutral density on the standard machine grid (R, Z, Phi).",
    'calc_rzp_dens': "Bool: Alias for calc_density (Standard machine grid storage).",
    'calc_uvw': "Bool: Stores beam/halo neutral density on the beam-aligned grid (U, V, W).",
    'calc_uvw_dens': "Bool: Alias for calc_uvw (Beam aligned grid storage).",
    'calc_extended_emission': "Bool: If True, simulates transitions other than Balmer-Alpha (requires 'transition' parameter).",
    'respawn_if_aperture_is_hit': "Bool: If True, markers hitting the aperture are regenerated until a valid ray enters the plasma.",
    'seed': "Int: Random number generator seed. Set to -1 for random runs, or a fixed integer for reproducibility.",
    'batch_marker': "Int: Number of markers processed per batch to manage memory usage. (Default: 10k-50k).",
    'separate_dcx': "Bool: If True, separates Direct Charge Exchange (DCX) from other halo contributions in output.",

    # --- SPECTROSCOPY (SPEC) ---
    'nlos': "Int: Number of Lines of Sight (Spectrometer Channels).",
    'los_pos': "Array [N, 3]: Cartesian starting position (x, y, z) [cm] of the collection lens for each channel.",
    'los_vec': "Array [N, 3]: Normalized unit vector direction (x, y, z) for each channel.",
    'losname': "List [String]: Descriptive labels/names for each channel.",
    'only_pi': "Bool: If True, only calculates Pi-polarized light emission.",
    'dlam': "Float: Spectral resolution / wavelength bin width [nm].",
    'lambda_min': "Float: Minimum wavelength for spectra storage [nm].",
    'lambda_max': "Float: Maximum wavelength for spectra storage [nm].",
    'nlam': "Int: Number of wavelength bins.",
    'sigma_to_pi_ratio': "Array [N]: Efficiency ratio of Sigma vs Pi light collection for each channel.",
    'output_individual_stark_lines': "Bool: If True, outputs raw Stark multiplet components separately instead of summed.",

    # --- GRID DEFINITIONS (Inputs) ---
    'grid_drz': "Float: Spatial resolution of the machine grid (R and Z directions) [cm].",
    'r_ran': "List [min, max]: Major radius extent of the simulation domain [cm].",
    'z_ran': "List [min, max]: Vertical extent of the simulation domain [cm].",
    'phi_ran': "List [min, max]: Toroidal angle extent [radians].",
    'u_range': "List [min, max]: Extent of the beam-aligned grid along the beam propagation axis [cm].",
    'v_width': "Float: Total width of the beam-aligned grid perpendicular to beam (horizontal) [cm].",
    'w_width': "Float: Total width of the beam-aligned grid perpendicular to beam (vertical) [cm].",
    'du': "Float: Step length along beam direction (u) for beam grid [cm].",
    'dvw': "Float: Step length perpendicular to beam (v, w) for beam grid [cm].",

    # -------------------------------------------------------------------------
    # DERIVED FIELDS & ARRAYS (Generated by input_prep routines)
    # These explain the "Fields" tab in the GUI.
    # -------------------------------------------------------------------------
    
    # --- 3D Field Arrays ---
    's': "Derived Array [R, Z, Phi]: Normalized flux coordinate (rho^2 or poloidal flux) mapped to the 3D grid.",
    'Br': "Derived Array [R, Z, Phi]: Radial Magnetic Field component [Tesla].",
    'Bz': "Derived Array [R, Z, Phi]: Vertical Magnetic Field component [Tesla].",
    'Bphi': "Derived Array [R, Z, Phi]: Toroidal Magnetic Field component [Tesla].",
    'Er': "Derived Array [R, Z, Phi]: Radial Electric Field component [V/cm].",
    
    # --- Grid Coordinates (1D) ---
    'R': "Derived Array: 1D array of Major Radius coordinates for the grid [cm].",
    'Z': "Derived Array: 1D array of Vertical coordinates for the grid [cm].",
    'phi': "Derived Array: 1D array of Toroidal angles for the grid [rad].",
    
    # --- Grid Metadata ---
    'Rmin': "Float: Minimum Major Radius of the grid [cm].",
    'Rmax': "Float: Maximum Major Radius of the grid [cm].",
    'Zmin': "Float: Minimum Vertical position [cm].",
    'Zmax': "Float: Maximum Vertical position [cm].",
    'phimin': "Float: Minimum Toroidal angle [rad].",
    'phimax': "Float: Maximum Toroidal angle [rad].",
    'dR': "Float: Grid spacing in Major Radius [cm].",
    'dZ': "Float: Grid spacing in Vertical direction [cm].",
    'dphi': "Float: Grid spacing in Toroidal angle [rad].",
    'nr': "Int: Number of radial grid points.",
    'nz': "Int: Number of vertical grid points.",
    'nphi': "Int: Number of toroidal grid points.",
    
    # --- Symmetry & Rotation ---
    'nsym': "Int: Toroidal symmetry number (e.g., 5 for W7-X, 1 for Tokamak).",
    'dphi_sym': "Float: Angular extent of one symmetry period [rad] (2pi/nsym).",
    'rotate_phi_grid': "Float: Offset angle applied to grid rotation [rad].",
    
    # --- Flux Surface Properties (1D) ---
    'flux_s': "Derived Array: 1D array of 's' coordinates corresponding to flux surfaces.",
    'flux_ds': "Float: Step size in 's' for flux surface interpolation.",
    'flux_s_min': "Float: Minimum 's' value for flux interpolation.",
    'flux_nr': "Int: Number of radial points in flux grid.",
    'flux_ns': "Int: Number of flux surfaces stored.",
    'flux_dvol': "Float/Array: Differential volume element between flux surfaces [cm^3].",
    
    # --- Visualization Surfaces ---
    's_surf': "Derived Array: Specific 's' values chosen for visualization contours.",
    'Rsurf': "Derived Array [s, phi, theta]: Radial coordinates of flux surface contours for plotting.",
    'Zsurf': "Derived Array [s, phi, theta]: Vertical coordinates of flux surface contours for plotting.",
    'Rmean': "Float: Mean Major Radius of the device [cm].",
    'btipsign': "Int: Direction of B-field vector (-1 or 1). Affects fast-ion orbit direction.",

    # -------------------------------------------------------------------------
    # PHYSICS & AUXILIARY INPUTS
    # -------------------------------------------------------------------------
    'ion_mass': "Float: Atomic mass of the bulk thermal plasma ions (e.g. 1.0 for H, 2.0 for D).",
    'nbi_mass': "Float: Atomic mass of the injected beam ions (e.g. 2.0 for D).",
    'ab': "Float: Alias for nbi_mass used in some NBI dictionaries.",
    'impurities': "List [Strings]: List of active impurities, e.g. ['Carbon', 'Neon', 'Boron'].",
    'path_to_tables': "String: Path to custom atomic data tables (optional).",
    'load_raw_data': "Bool: If True, loads raw data from pre-supplied binary tables.",

    # --- FAST ION DISTRIBUTION (FBM) ---
    'fbm_file': "String: Path to the Fast-Ion Distribution file (.cdf).",
    'emin': "Float: Minimum energy to load from FBM [keV].",
    'emax': "Float: Maximum energy to load from FBM [keV].",
    'pmin': "Float: Minimum pitch angle (v_par/v) to load.",
    'pmax': "Float: Maximum pitch angle (v_par/v) to load.",
    'fbm': "Array: The loaded 4D Fast-Ion Distribution function.",
    'denf': "Array: The loaded Fast-ion density distribution.",

    # --- NBI GEOMETRY & PROPERTIES ---
    'sources': "List [String]: Keys identifying active beam sources.",
    'power': "Float: Injected Neutral Beam Power [MW].",
    'voltage': "Float: Beam Acceleration Voltage [kV].",
    'current_fractions': "List [f1, f2, f3]: Current fractions for Full, Half, and Third energy components.",
    'source_position': "Array [x, y, z]: Cartesian coordinates of the ion source grid center [cm].",
    'direction': "Array [dx, dy, dz]: Normalized vector defining beam injection direction.",
    'divergence': "Array [h_div, v_div]: Beam divergence angles in radians [horizontal, vertical].",
    'focal_length': "Array [h_foc, v_foc]: Focal lengths of beam optics [cm].",
    'ion_source_size': "Array [w, h]: Physical size of the ion source grid [cm].",
    'aperture_1_distance': "Float: Distance from source to 1st aperture [cm].",
    'aperture_1_size': "Array [w, h]: Size of 1st aperture [cm].",
    'aperture_1_offset': "Array [dx, dy]: Misalignment offset of 1st aperture [cm].",
    'aperture_1_rectangular': "Bool: True if rectangular, False if circular.",
    'aperture_2_distance': "Float: Distance from source to 2nd aperture [cm].",
    'aperture_2_size': "Array [w, h]: Size of 2nd aperture [cm].",
    'aperture_2_offset': "Array [dx, dy]: Misalignment offset of 2nd aperture [cm].",
    'aperture_2_rectangular': "Bool: True if rectangular, False if circular.",
    'uvw_xyz_rot': "Matrix [3x3]: Rotation matrix transforming beam coords (UVW) to machine coords (XYZ).",

    # --- POINT SPREAD FUNCTION (PSF) ---
    'image_pos': "List [x, y, z]: Central location of image source on collection lens.",
    'image_vec': "List [dx, dy, dz]: Vector from image pos to focal point.",
    'los_image_arr': "List [int]: Mapping of LOS index to image group.",
    'd_fiber': "Float: Diameter of fibers [cm].",
    'p_fiber': "List [[x,y]...]: Central position of fibers in units of d_fiber.",
    'npoint': "Int: Number of points to fill circular fiber area.",
    'n_rand': "Int: Number of random samples for finite lifetime decay.",
    'f_lens': "Float: Collection lens focal length [cm].",
    'plot_blur_image': "Bool: If True, plot unmagnified fiber blurring kernel.",

    # --- EXTENDED EMISSION ---
    'spectrum_extended': "Bool: Enable non-Balmer-Alpha transitions.",
    'transition': "List [n_upper, n_lower]: Quantum numbers for the transition (e.g., [4, 2] for H-beta).",

    # --- TRANSP INPUTS ---
    'transp': "Bool: Use TRANSP routines for profile/grid generation.",
    'time': "Float: Time point in TRANSP run to load.",
    'runid': "String: Run ID for TRANSP file lookup.",
    'directory': "String: Path to TRANSP files.",
    'ntheta': "Int: Number of theta divisions for magnetic field storage.",
    'Bt_sign': "Int: Direction of toroidal B-field (1 or -1).",
    'Ip_sign': "Int: Direction of plasma current (1 or -1).",
    'prof_plot': "Bool: Plot loaded plasma profiles.",
    's_new': "List: New s-domain for profile interpolation.",
    'n_decay': "Float: Decay length for density in SOL.",
    'te_decay': "Float: Decay length for Te in SOL.",
    'nbi_source_sel': "List [int]: Subset of NBI sources to use from TRANSP.",

    # --- W7-X SPECIFIC ---
    'machine': "String: Set to 'W7X' for W7-X runs.",
    'los_shot': "String: Shot ID for LOS data.",
    'los_head': "String: Subset of LOS to use (e.g. 'AEA').",
    'progID': "String: Discharge number.",
    'vmecID': "String: VMEC ID for wout file.",
    'eq_path': "String: Path to wout file.",
    'extended_vmec_factor': "Float: Scaling factor for flux labeling extension.",
    'b0_factor': "Float: Scaling factor for B-field magnitude.",

    # --- GUI SYNTHETIC PARAMETERS ---
    'ne_axis': "Float: Core Electron Density [cm^-3].",
    'ne_ped': "Float: Pedestal Electron Density [cm^-3].",
    'ne_sep': "Float: Separatrix Electron Density [cm^-3].",
    'te_axis': "Float: Core Electron Temp [keV].",
    'te_ped': "Float: Pedestal Electron Temp [keV].",
    'te_sep': "Float: Separatrix Electron Temp [keV].",
    'ti_axis': "Float: Core Ion Temp [keV].",
    'ti_ped': "Float: Pedestal Ion Temp [keV].",
    'ti_sep': "Float: Separatrix Ion Temp [keV].",
    'omega_axis': "Float: Core Rotation [rad/s].",
    'lens_center': "Array: Center of collection lens [cm].",
    'target_x': "Float: X-coord of view intersection [cm].",
    'target_y_start': "Float: Start Y of view scan [cm].",
    'target_y_end': "Float: End Y of view scan [cm].",
}

# =============================================================================
# REFERENCE DATA GENERATION (For Educational Tab)
# =============================================================================
def get_reference_templates():
    """
    Returns dictionaries containing keys for FBM, PSF, Extended Emission, and TRANSP 
    that might not be used in the synthetic run, but are vital for 
    educational reference regarding input_prep.py structure.
    """
    fbm = {
        'fbm_file': "path/to/dist.cdf", 
        'emin': 0.0, 'emax': 100.0, 
        'pmin': -1.0, 'pmax': 1.0, 
        'fbm': "Array [E, P, R, Z] (Loaded at runtime)", 
        'denf': "Array [R, Z] (Loaded at runtime)"
    }
    
    psf = {
        'calc_PSF': False, 
        'image_pos': [[100.0, 50.0, 0.0]], 
        'image_vec': [[-1.0, 0.0, 0.0]],
        'los_image_arr': [0, 0, 0],
        'd_fiber': 0.1, 
        'npoint': 100, 
        'n_rand': 100, 
        'f_lens': 50.0,
        'p_fiber': [[0.0, 0.0], [0.1, 0.0]], 
        'plot_blur_image': False
    }
    
    extended = {
        'calc_extended_emission': False,
        'spectrum_extended': False,
        'transition': [4, 2] # n=4 to n=2 (H-beta)
    }
    
    transp = {
        'transp': False, 
        'runid': "12345X01", 
        'time': 1.5,
        'directory': "./Data/",
        'ntheta': 200, 'nr': 100, 'nz': 100, 
        'Bt_sign': -1, 'Ip_sign': 1,
        'n_decay': 0.1, 'te_decay': 0.1
    }
    
    w7x = {
        'machine': 'W7X',
        'los_shot': '20180823.035',
        'los_head': 'AEA',
        'progID': '20180823.037',
        'vmecID': 'w7x_ref_66',
        'eq_path': './Data/wout.nc',
        'extended_vmec_factor': 1.0
    }
    
    return fbm, psf, extended, transp, w7x

# =============================================================================
# WORKER THREAD
# =============================================================================
class SimulationWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, config, tables):
        super().__init__()
        self.config = config
        self.tables = tables

    def run(self):
        try:
            self.log.emit("--- Building Physics Data from Config ---")
            
            # 1. Rebuild Heavy Data from Config
            sim_settings, fields, profiles, nbi, spec = build_simulation_data(self.config)
            
            # 2. Define Grid using updated Grid Config
            gc = self.config['grid']
            grid3d = define_grid3d(
                fields, 
                drz=gc['grid_drz'], 
                nphi=fields['nphi'], 
                u_range=gc['u_range'], 
                v_width=gc['v_width'], 
                w_width=gc['w_width'],
                du=gc['du'],
                dvw=gc['dvw']
            )
            
            self.log.emit("Calculating Intersections...")
            spec = grid_intersections(spec, fields, grid3d)
            spec['los_grid_intersection_weight'] = np.ones_like(spec['grid_cell_crossed_by_los'], dtype=float)

            # Placeholders
            ncdf = {'active': False, 'lambda0': 656.1, 'trans': 1, 'l_to_dwp': 1, 'spectrum_extended': False, 'cdfvars': np.zeros((1,1,1))}
            PSF = {'image_pos': 1, 'image_vec': 1, 'image_blur': 0, 'los_image_arr': 0, 'n_rand': 0, 'f_lens': 1}
            fbm = {'afbm': 0, 'fbm': np.zeros((1,1,1,1)), 'denf': np.zeros((1,1,1)), 'btipsign': -1, 'emin':0,'eran':0,'nenergy':0,'energy':np.zeros(1),'dE':0,'pmin':0,'pran':0,'npitch':0,'pitch':np.zeros(1),'dP':0}

            self.log.emit("--- Starting pyFIDASIM Core ---")
            t1 = time.time()
            grid3d_out, spec_out = calc_attenuation(
                sim_settings, profiles, nbi, spec, fields, grid3d, self.tables, ncdf, PSF, fbm
            )
            t2 = time.time()
            self.log.emit(f"Done in {t2-t1:.2f}s.")

            results = {
                'grid3d': grid3d_out, 'spec': spec_out, 'fields': fields, 'profiles': profiles, 'nbi': nbi
            }
            self.finished.emit(results)

        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())

# =============================================================================
# WIDGETS
# =============================================================================
class MatplotlibCanvas(QWidget):
    """A Widget to hold a Matplotlib Figure."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.canvas = None
        self.toolbar = None
        self.layout.setContentsMargins(0,0,0,0)

    def update_figure(self, fig):
        if self.canvas is not None:
            self.layout.removeWidget(self.canvas)
            self.layout.removeWidget(self.toolbar)
            self.canvas.close()
            self.toolbar.close()
        
        if fig is None: return
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

class DictionaryEditor(QTreeWidget):
    """Recursively edits a dictionary."""
    def __init__(self, data_dict, parent=None):
        super().__init__(parent)
        self.data = data_dict
        self.setHeaderLabels(["Parameter", "Value", "Description"])
        self.setColumnWidth(0, 180)
        self.setColumnWidth(1, 150)
        self.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.build_tree()

    def build_tree(self):
        self.clear()
        self._populate_tree(self.data, self.invisibleRootItem())
        self.expandAll()

    def update_data(self, new_data):
        self.data = new_data
        self.build_tree()

    def _populate_tree(self, data, parent_item):
        for key, value in data.items():
            item = QTreeWidgetItem(parent_item)
            item.setText(0, str(key))
            
            desc = TOOLTIPS.get(key, "No description available.")
            item.setText(2, desc)
            item.setToolTip(0, desc)
            item.setToolTip(2, desc)

            if isinstance(value, dict):
                item.setExpanded(True)
                self._populate_tree(value, item)
            
            elif isinstance(value, (list, np.ndarray)):
                # Handle large arrays safely
                is_array = isinstance(value, np.ndarray)
                size = value.size if is_array else len(value)
                
                if size < 20:
                    val_str = ", ".join(map(str, value.flatten())) if is_array else ", ".join(map(str, value))
                    widget = QLineEdit(val_str)
                    if is_array:
                        widget.editingFinished.connect(lambda k=key, d=data, w=widget: self._update_array(d, k, w.text()))
                    else:
                        widget.editingFinished.connect(lambda k=key, d=data, w=widget: self._update_list(d, k, w.text()))
                    self.setItemWidget(item, 1, widget)
                else:
                    shape_str = str(value.shape) if is_array else f"len={size}"
                    item.setText(1, f"Array {shape_str}")
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            elif isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                widget.toggled.connect(lambda val, k=key, d=data: d.update({k: val}))
                self.setItemWidget(item, 1, widget)
            
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(-1, 100000000)
                widget.setValue(value)
                widget.valueChanged.connect(lambda val, k=key, d=data: d.update({k: val}))
                self.setItemWidget(item, 1, widget)
            
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-1e9, 1e9)
                widget.setDecimals(4)
                widget.setValue(value)
                widget.valueChanged.connect(lambda val, k=key, d=data: d.update({k: val}))
                self.setItemWidget(item, 1, widget)
            
            elif isinstance(value, str):
                widget = QLineEdit(value)
                widget.textChanged.connect(lambda val, k=key, d=data: d.update({k: val}))
                self.setItemWidget(item, 1, widget)
            
            else:
                item.setText(1, str(value))

    def _update_list(self, data_dict, key, text):
        try:
            parts = text.split(',')
            new_list = []
            for p in parts:
                p = p.strip()
                try: new_list.append(float(p))
                except: new_list.append(p)
            data_dict[key] = new_list
        except: pass

    def _update_array(self, data_dict, key, text):
        try:
            new_list = [float(x.strip()) for x in text.split(',')]
            data_dict[key] = np.array(new_list)
        except: pass

# =============================================================================
# MAIN WINDOW
# =============================================================================
class PyFidaSimGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyFIDASIM: Synthetic Simulation Suite")
        self.resize(1500, 950)
        
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Loading Tables...")
        
        try:
            self.tables = load_tables()
            self.config = get_default_config()
            # Initial heavy build
            self.sim_settings, self.fields, self.profiles, self.nbi, self.spec = build_simulation_data(self.config)
            self.initial_data = (self.sim_settings, self.fields, self.profiles, self.nbi, self.spec)
            
            # Reference Dictionaries
            self.fbm_tpl, self.psf_tpl, self.ext_tpl, self.transp_tpl, self.w7x_tpl = get_reference_templates()
            
        except Exception as e:
            QMessageBox.critical(self, "Init Error", f"{e}")
            sys.exit(1)

        self.init_ui()
        self.status_bar.showMessage("Ready. Modify settings in '1. Synthetic Settings' or view derived data in '2. PyFIDASIM Reference'.")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # --- LEFT PANEL: DUAL TAB LAYOUT ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_tabs = QTabWidget()
        
        # TAB 1: GENERATOR SETTINGS (The "Knobs")
        self.tab_gen = QWidget()
        gen_layout = QVBoxLayout(self.tab_gen)
        
        gen_subtabs = QTabWidget()
        self.tree_sim = DictionaryEditor(self.config['simulation'])
        gen_subtabs.addTab(self.tree_sim, "Settings")
        
        self.tree_grid = DictionaryEditor(self.config['grid'])
        gen_subtabs.addTab(self.tree_grid, "Grid")
        
        self.tree_mach = DictionaryEditor(self.config['machine'])
        gen_subtabs.addTab(self.tree_mach, "Machine")
        
        self.tree_prof = DictionaryEditor(self.config['profiles'])
        gen_subtabs.addTab(self.tree_prof, "Profiles")
        
        self.tree_nbi_gen = DictionaryEditor(self.config['nbi'])
        gen_subtabs.addTab(self.tree_nbi_gen, "NBI")
        
        self.tree_spec_gen = DictionaryEditor(self.config['spec'])
        gen_subtabs.addTab(self.tree_spec_gen, "Spec")
        
        gen_layout.addWidget(gen_subtabs)
        left_tabs.addTab(self.tab_gen, "1. Synthetic Settings")

        # TAB 2: REFERENCE INPUTS (The "Results")
        # Shows what input_prep.py actually produces, including FBM/PSF/Fields
        self.tab_ref = QWidget()
        ref_layout = QVBoxLayout(self.tab_ref)
        ref_subtabs = QTabWidget()
        
        # Actual Generated Data (Derived from Gen Settings)
        self.tree_ref_sim = DictionaryEditor(self.sim_settings)
        ref_subtabs.addTab(self.tree_ref_sim, "Sim Settings")
        
        self.tree_ref_fields = DictionaryEditor(self.fields)
        ref_subtabs.addTab(self.tree_ref_fields, "Fields")
        
        self.tree_ref_nbi = DictionaryEditor(self.nbi)
        ref_subtabs.addTab(self.tree_ref_nbi, "NBI")
        
        self.tree_ref_spec = DictionaryEditor(self.spec)
        ref_subtabs.addTab(self.tree_ref_spec, "Spectroscopy")
        
        # Templates (Educational)
        self.tree_ref_fbm = DictionaryEditor(self.fbm_tpl)
        ref_subtabs.addTab(self.tree_ref_fbm, "FBM (Template)")
        
        self.tree_ref_psf = DictionaryEditor(self.psf_tpl)
        ref_subtabs.addTab(self.tree_ref_psf, "PSF (Template)")
        
        self.tree_ref_ext = DictionaryEditor(self.ext_tpl)
        ref_subtabs.addTab(self.tree_ref_ext, "Ext. Emission (Template)")
        
        self.tree_ref_transp = DictionaryEditor(self.transp_tpl)
        ref_subtabs.addTab(self.tree_ref_transp, "TRANSP (Template)")
        
        self.tree_ref_w7x = DictionaryEditor(self.w7x_tpl)
        ref_subtabs.addTab(self.tree_ref_w7x, "W7-X (Template)")
        
        ref_layout.addWidget(ref_subtabs)
        left_tabs.addTab(self.tab_ref, "2. PyFIDASIM Reference")

        left_layout.addWidget(left_tabs)
        splitter.addWidget(left_widget)

        # --- RIGHT: PLOTS ---
        self.plot_tabs = QTabWidget()
        splitter.addWidget(self.plot_tabs)
        splitter.setSizes([500, 900])

        self.canvases = {}
        for name in ["Profiles", "Equilibrium", "Geometry", "Beam Density", "Spectra"]:
            c = MatplotlibCanvas()
            self.plot_tabs.addTab(c, name)
            self.canvases[name] = c

        # --- BOTTOM: BUTTONS ---
        bot = QHBoxLayout()
        layout.addLayout(bot)
        
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setStyleSheet("background: #4CAF50; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.start_sim)
        
        self.btn_preview = QPushButton("Update Previews / Reference")
        self.btn_preview.setFixedHeight(40)
        self.btn_preview.clicked.connect(self.update_previews)

        bot.addWidget(QLabel("Status:"))
        self.lbl_status = QLabel("Idle")
        bot.addWidget(self.lbl_status)
        bot.addWidget(self.btn_preview)
        bot.addWidget(self.btn_run)

        # Initial Render
        self.update_previews()

    def update_previews(self):
        try:
            # Rebuild heavy data from current config state
            self.sim_settings, self.fields, self.profiles, self.nbi, self.spec = build_simulation_data(self.config)
            
            # Update Reference Trees
            self.tree_ref_sim.update_data(self.sim_settings)
            self.tree_ref_fields.update_data(self.fields)
            self.tree_ref_nbi.update_data(self.nbi)
            self.tree_ref_spec.update_data(self.spec)
            
            # Update Static Plots
            self.canvases["Profiles"].update_figure(plot_profiles(self.profiles))
            self.canvases["Equilibrium"].update_figure(plot_magnetic_equilibrium(self.fields))
            self.canvases["Geometry"].update_figure(plot_geometry_3d(self.fields, self.spec, self.nbi))
            
            self.status_bar.showMessage("Previews and Reference Data Updated.")
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", str(e))

    def start_sim(self):
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Simulating...")
        self.plot_tabs.setCurrentIndex(4) # Spectra tab

        # Pass the current config state to worker
        self.worker = SimulationWorker(self.config, self.tables)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.log.connect(lambda s: self.status_bar.showMessage(s))
        self.worker.start()

    def on_finished(self, res):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Complete")
        
        # Ensure reference tabs show what was actually run
        self.tree_ref_fields.update_data(res['fields'])
        self.tree_ref_nbi.update_data(res['nbi'])
        self.tree_ref_spec.update_data(res['spec'])
        
        self.plot_results(res['grid3d'], res['fields'], res['profiles'], res['nbi'], res['spec'])

    def plot_results(self, grid3d, fields, profiles, nbi, spec):
        self.canvases["Profiles"].update_figure(plot_profiles(profiles))
        self.canvases["Equilibrium"].update_figure(plot_magnetic_equilibrium(fields))
        fig_geo = plot_geometry_3d(fields, spec, nbi, grid3d, plot_crossed_cells=True)
        self.canvases["Geometry"].update_figure(fig_geo)
        self.canvases["Beam Density"].update_figure(plot_midplane_heatmap(grid3d, fields))
        if 'intens' in spec:
            self.canvases["Spectra"].update_figure(plot_spectra_interactive(spec, scale_factor=1e18))

    def on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Error")
        QMessageBox.critical(self, "Sim Error", msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = PyFidaSimGUI()
    w.show()
    sys.exit(app.exec())