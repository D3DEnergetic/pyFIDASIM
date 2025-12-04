# gui/input_panel.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QHBoxLayout, QPushButton, QFileDialog
)
import numpy as np

class InputPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self._create_widgets()
        self._connect_signals()
        self.toggle_mode()

    def _create_file_browser(self, line_edit, caption, file_filter="", directory=False):
        """Helper function to create a line edit with a browse button."""
        button = QPushButton("Browse...")
        layout = QHBoxLayout()
        layout.addWidget(line_edit)
        layout.addWidget(button)
        layout.setContentsMargins(0, 0, 0, 0)

        def open_dialog():
            if directory:
                path = QFileDialog.getExistingDirectory(self, caption, line_edit.text())
            else:
                path, _ = QFileDialog.getOpenFileName(self, caption, line_edit.text(), filter=file_filter)
            if path:
                line_edit.setText(path)

        button.clicked.connect(open_dialog)
        return layout

    def _create_widgets(self):
        # --- Mode Selection ---
        mode_group = QGroupBox("Simulation Mode"); mode_group.setCheckable(False)
        mode_layout = QFormLayout()
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["W7-X", "Fortran FIDASIM", "TRANSP"])
        mode_layout.addRow("Select Mode:", self.mode_selector)
        mode_group.setLayout(mode_layout)

        # --- Groups for different modes ---
        self._create_w7x_group()
        self._create_f90_group()
        self._create_transp_group()

        # --- Common/General Settings ---
        self._create_general_group()
        self._create_uvw_group()
        self._create_advanced_sim_group()
        self._create_impurity_group()
        self._create_extended_emission_group()

        self.main_layout.addWidget(mode_group)
        self.main_layout.addWidget(self.w7x_group)
        self.main_layout.addWidget(self.f90_group)
        self.main_layout.addWidget(self.transp_group)
        self.main_layout.addWidget(self.general_group)
        self.main_layout.addWidget(self.uvw_group)
        self.main_layout.addWidget(self.advanced_sim_group)
        self.main_layout.addWidget(self.impurity_group)
        self.main_layout.addWidget(self.extended_emission_group)
        self.main_layout.addStretch()

    def _create_w7x_group(self):
        self.w7x_group = QGroupBox("W7-X Settings")
        w7x_layout = QFormLayout()

        # --- LOS Settings ---
        self.w7x_los_shot = QLineEdit("20180823.035")
        self.w7x_los_head = QLineEdit("AEM21_HPPS7:")
        self.w7x_los_file = QLineEdit("Data/losInfo_New_op2.txt")
        self.w7x_los_default = QCheckBox(); self.w7x_los_default.setChecked(False)
        self.w7x_los_new = QCheckBox(); self.w7x_los_new.setChecked(True)

        # --- Equilibrium Settings ---
        self.w7x_progID = QLineEdit("")
        self.w7x_vmecID = QLineEdit("")
        self.w7x_eq_path = QLineEdit("")
        self.w7x_extended_vmec_factor = QDoubleSpinBox(); self.w7x_extended_vmec_factor.setValue(1.0)
        self.w7x_b0_factor = QDoubleSpinBox(); self.w7x_b0_factor.setValue(1.0)
        self.w7x_drz = QDoubleSpinBox(); self.w7x_drz.setValue(2.0)
        self.w7x_phi_ran_min = QDoubleSpinBox(); self.w7x_phi_ran_min.setValue(0.45)
        self.w7x_phi_ran_max = QDoubleSpinBox(); self.w7x_phi_ran_max.setValue(0.53)

        # --- Profile Settings ---
        self.w7x_prof_path = QLineEdit("Data/W7Xprofiles.h5")

        # --- NBI Settings ---
        self.w7x_shot_num = QLineEdit("20180920.042")
        self.w7x_t_start = QDoubleSpinBox(); self.w7x_t_start.setValue(6.5)
        self.w7x_t_stop = QDoubleSpinBox(); self.w7x_t_stop.setValue(6.52)
        self.w7x_nbi_cur_frac = QLineEdit("")
        self.w7x_nbi_cur_frac.setToolTip("Comma-separated floats for current fractions")
        self.w7x_nbi_debug = QCheckBox()
        self.w7x_nbi_default = QCheckBox()
        self.w7x_ion_mass = QDoubleSpinBox(); self.w7x_ion_mass.setValue(1.0)
        self.w7x_nbi_mass = QDoubleSpinBox(); self.w7x_nbi_mass.setValue(1.0)

        # Adding widgets to layout
        w7x_layout.addRow("LOS Shot:", self.w7x_los_shot)
        w7x_layout.addRow("LOS Header:", self.w7x_los_head)
        w7x_layout.addRow("LOS File:", self._create_file_browser(self.w7x_los_file, "Select LOS File"))
        w7x_layout.addRow("Use Default LOS:", self.w7x_los_default)
        w7x_layout.addRow("Use New LOS Setup:", self.w7x_los_new)
        w7x_layout.addRow("Equilibrium ProgID:", self.w7x_progID)
        w7x_layout.addRow("Equilibrium VMEC ID:", self.w7x_vmecID)
        w7x_layout.addRow("Equilibrium Path (wout):", self._create_file_browser(self.w7x_eq_path, "Select WOUT File"))
        w7x_layout.addRow("Extend VMEC Factor:", self.w7x_extended_vmec_factor)
        w7x_layout.addRow("B0 Factor:", self.w7x_b0_factor)
        w7x_layout.addRow("Equilibrium DRZ [cm]:", self.w7x_drz)
        w7x_layout.addRow("Phi Range (x π):", self.w7x_phi_ran_min)
        w7x_layout.addRow("", self.w7x_phi_ran_max)
        w7x_layout.addRow("Profile Path (HDF5):", self._create_file_browser(self.w7x_prof_path, "Select Profile File", "*.h5"))
        w7x_layout.addRow("NBI Shot Number:", self.w7x_shot_num)
        w7x_layout.addRow("NBI Time Start [s]:", self.w7x_t_start)
        w7x_layout.addRow("NBI Time Stop [s]:", self.w7x_t_stop)
        w7x_layout.addRow("NBI Current Fractions:", self.w7x_nbi_cur_frac)
        w7x_layout.addRow("NBI Debug Plot:", self.w7x_nbi_debug)
        w7x_layout.addRow("Use Default NBI:", self.w7x_nbi_default)
        w7x_layout.addRow("Ion Mass:", self.w7x_ion_mass)
        w7x_layout.addRow("NBI Mass:", self.w7x_nbi_mass)
        self.w7x_group.setLayout(w7x_layout)

    def _create_f90_group(self):
        self.f90_group = QGroupBox("Fortran FIDASIM Settings")
        f90_layout = QFormLayout()
        self.f90_directory = QLineEdit("comp_data")
        self.f90_runid = QLineEdit("141648E01")
        self.f90_geqdsk = QLineEdit("g141648.00185")
        f90_layout.addRow("Directory:", self._create_file_browser(self.f90_directory, "Select F90 Directory", directory=True))
        f90_layout.addRow("Run ID:", self.f90_runid)
        f90_layout.addRow("GEQDSK File:", self._create_file_browser(self.f90_geqdsk, "Select GEQDSK File"))
        self.f90_group.setLayout(f90_layout)

    def _create_transp_group(self):
        self.transp_group = QGroupBox("TRANSP Settings")
        transp_layout = QFormLayout()
        self.transp_directory = QLineEdit("path/to/transp_files")
        self.transp_runid = QLineEdit("TRANS_RUN_ID")
        self.transp_time = QDoubleSpinBox(); self.transp_time.setDecimals(4); self.transp_time.setValue(1.2345)
        self.transp_ion_mass = QDoubleSpinBox(); self.transp_ion_mass.setValue(2.0)
        self.transp_nbi_mass = QDoubleSpinBox(); self.transp_nbi_mass.setValue(2.0)
        self.transp_bt_sign = QSpinBox(); self.transp_bt_sign.setRange(-1,1); self.transp_bt_sign.setSingleStep(2); self.transp_bt_sign.setValue(1)
        self.transp_ip_sign = QSpinBox(); self.transp_ip_sign.setRange(-1,1); self.transp_ip_sign.setSingleStep(2); self.transp_ip_sign.setValue(1)
        self.transp_phi_ran_min = QDoubleSpinBox(); self.transp_phi_ran_min.setRange(0, 2); self.transp_phi_ran_min.setValue(0.0)
        self.transp_phi_ran_max = QDoubleSpinBox(); self.transp_phi_ran_max.setRange(0, 2); self.transp_phi_ran_max.setValue(2.0)
        self.transp_ntheta = QSpinBox(); self.transp_ntheta.setRange(10, 1000); self.transp_ntheta.setValue(200)
        self.transp_nr = QSpinBox(); self.transp_nr.setRange(10, 1000); self.transp_nr.setValue(300)
        self.transp_nz = QSpinBox(); self.transp_nz.setRange(10, 1000); self.transp_nz.setValue(300)
        self.transp_ne_decay = QDoubleSpinBox(); self.transp_ne_decay.setValue(0.1)
        self.transp_te_decay = QDoubleSpinBox(); self.transp_te_decay.setValue(0.1)
        self.transp_ti_decay = QDoubleSpinBox(); self.transp_ti_decay.setValue(0.1)
        self.transp_omega_decay = QDoubleSpinBox(); self.transp_omega_decay.setValue(0.1)
        self.transp_s_new = QLineEdit(); self.transp_s_new.setToolTip("Comma-separated float values for new s-domain")
        self.transp_prof_plot = QCheckBox()
        self.transp_nbi_plot = QCheckBox()

        transp_layout.addRow("Directory:", self._create_file_browser(self.transp_directory, "Select TRANSP Directory", directory=True))
        transp_layout.addRow("Run ID:", self.transp_runid)
        transp_layout.addRow("Time [s]:", self.transp_time)
        transp_layout.addRow("Ion Mass:", self.transp_ion_mass)
        transp_layout.addRow("NBI Mass:", self.transp_nbi_mass)
        transp_layout.addRow("Bt Sign:", self.transp_bt_sign)
        transp_layout.addRow("Ip Sign:", self.transp_ip_sign)
        transp_layout.addRow("Phi Range (x π):", self.transp_phi_ran_min)
        transp_layout.addRow("", self.transp_phi_ran_max)
        transp_layout.addRow("n_theta:", self.transp_ntheta)
        transp_layout.addRow("n_r:", self.transp_nr)
        transp_layout.addRow("n_z:", self.transp_nz)
        transp_layout.addRow("n_e Decay Length:", self.transp_ne_decay)
        transp_layout.addRow("T_e Decay Length:", self.transp_te_decay)
        transp_layout.addRow("T_i Decay Length:", self.transp_ti_decay)
        transp_layout.addRow("Omega Decay Length:", self.transp_omega_decay)
        transp_layout.addRow("New S-Domain:", self.transp_s_new)
        transp_layout.addRow("Plot Profiles:", self.transp_prof_plot)
        transp_layout.addRow("Plot NBI:", self.transp_nbi_plot)
        self.transp_group.setLayout(transp_layout)

    def _create_general_group(self):
        self.general_group = QGroupBox("General Settings"); self.general_group.setCheckable(False)
        layout = QFormLayout()
        self.nmarker = QSpinBox(); self.nmarker.setRange(100, 10000000); self.nmarker.setValue(10000)
        self.seed = QSpinBox(); self.seed.setRange(-1, 1000000); self.seed.setValue(12345)
        self.lambda_min = QDoubleSpinBox(); self.lambda_min.setRange(0,2000); self.lambda_min.setValue(652.0)
        self.lambda_max = QDoubleSpinBox(); self.lambda_max.setRange(0,2000); self.lambda_max.setValue(662.0)
        self.dlam = QDoubleSpinBox(); self.dlam.setDecimals(3); self.dlam.setValue(0.01)
        self.only_pi = QCheckBox()
        layout.addRow("Num Markers:", self.nmarker)
        layout.addRow("Random Seed (-1 for none):", self.seed)
        layout.addRow("λ Min [nm]:", self.lambda_min)
        layout.addRow("λ Max [nm]:", self.lambda_max)
        layout.addRow("dλ [nm]:", self.dlam)
        layout.addRow("Only Pi-Polarized Light:", self.only_pi)
        self.general_group.setLayout(layout)

    def _create_uvw_group(self):
        self.uvw_group = QGroupBox("Grid & UVW Settings"); self.uvw_group.setCheckable(True); self.uvw_group.setChecked(True)
        layout = QFormLayout()
        self.calc_uvw = QCheckBox(); self.calc_uvw.setChecked(True)
        self.grid_drz = QDoubleSpinBox(); self.grid_drz.setValue(1.0)
        self.du = QDoubleSpinBox(); self.du.setValue(1.0)
        self.dvw = QDoubleSpinBox(); self.dvw.setValue(1.0)
        self.u_range_min = QSpinBox(); self.u_range_min.setRange(-2000,2000); self.u_range_min.setValue(-100)
        self.u_range_max = QSpinBox(); self.u_range_max.setRange(-2000,2000); self.u_range_max.setValue(900)
        self.v_width = QDoubleSpinBox(); self.v_width.setRange(0,1000); self.v_width.setValue(120.0)
        self.w_width = QDoubleSpinBox(); self.w_width.setRange(0,1000); self.w_width.setValue(120.0)
        layout.addRow("Calculate UVW Grid:", self.calc_uvw)
        layout.addRow("Grid DRZ [cm]:", self.grid_drz)
        layout.addRow("dU [cm]:", self.du)
        layout.addRow("dVW [cm]:", self.dvw)
        layout.addRow("U Range Min [cm]:", self.u_range_min)
        layout.addRow("U Range Max [cm]:", self.u_range_max)
        layout.addRow("V Width [cm]:", self.v_width)
        layout.addRow("W Width [cm]:", self.w_width)
        self.uvw_group.setLayout(layout)

    def _create_advanced_sim_group(self):
        self.advanced_sim_group = QGroupBox("Advanced Simulation"); self.advanced_sim_group.setCheckable(False)
        layout = QFormLayout()
        self.verbose = QCheckBox(); self.verbose.setChecked(True)
        self.calc_halo = QCheckBox(); self.calc_halo.setChecked(True)
        self.calc_density = QCheckBox(); self.calc_density.setChecked(True)
        self.calc_photon_origin = QCheckBox()
        self.respawn_aperture = QCheckBox(); self.respawn_aperture.setChecked(True)
        self.skip_options = QLineEdit("")
        self.skip_options.setToolTip("e.g. spec,tables,fields")
        layout.addRow("Verbose Logging:", self.verbose)
        layout.addRow("Calculate Halo:", self.calc_halo)
        layout.addRow("Calculate Density (R,Z,phi):", self.calc_density)
        layout.addRow("Calculate Photon Origin:", self.calc_photon_origin)
        layout.addRow("Respawn if Aperture Hit:", self.respawn_aperture)
        layout.addRow("Skip Prep Steps:", self.skip_options)
        self.advanced_sim_group.setLayout(layout)

    def _create_impurity_group(self):
        self.impurity_group = QGroupBox("Impurity & Tables"); self.impurity_group.setCheckable(False)
        layout = QFormLayout()
        self.impurities = QLineEdit("Carbon")
        self.impurities.setToolTip("Enter comma-separated list, e.g., Carbon,Boron,Neon")
        self.path_to_tables = QLineEdit("")
        self.load_raw_data = QCheckBox(); self.load_raw_data.setChecked(True)
        layout.addRow("Impurities:", self.impurities)
        layout.addRow("Path to Tables:", self._create_file_browser(self.path_to_tables, "Select Custom Tables Directory", directory=True))
        layout.addRow("Load Raw Data:", self.load_raw_data)
        self.impurity_group.setLayout(layout)

    def _create_extended_emission_group(self):
        self.extended_emission_group = QGroupBox("Extended Emission"); self.extended_emission_group.setCheckable(True)
        layout = QFormLayout()
        self.trans_n_max = QSpinBox(); self.trans_n_max.setRange(3, 6); self.trans_n_max.setValue(3)
        self.trans_n_min = QSpinBox(); self.trans_n_min.setRange(1, 5); self.trans_n_min.setValue(2)
        layout.addRow("Transition (n_max -> n_min):", self.trans_n_max)
        layout.addRow("", self.trans_n_min)
        self.extended_emission_group.setLayout(layout)
        self.extended_emission_group.setChecked(False) # Default off

    def _connect_signals(self):
        self.mode_selector.currentTextChanged.connect(self.toggle_mode)

    def toggle_mode(self):
        mode = self.mode_selector.currentText()
        is_w7x = (mode == "W7-X")
        is_f90 = (mode == "Fortran FIDASIM")
        is_transp = (mode == "TRANSP")
        self.w7x_group.setVisible(is_w7x)
        self.f90_group.setVisible(is_f90)
        self.transp_group.setVisible(is_transp)

    def get_fidasim_params(self):
        params = {}
        mode = self.mode_selector.currentText()

        # General Settings
        params['nmarker'] = self.nmarker.value()
        params['seed'] = self.seed.value()
        params['lambda_min'] = self.lambda_min.value()
        params['lambda_max'] = self.lambda_max.value()
        params['dlam'] = self.dlam.value()
        params['only_pi'] = self.only_pi.isChecked()
        params["batch_marker"] = params["nmarker"]

        # Grid & UVW Settings
        if self.uvw_group.isChecked():
            params['calc_uvw'] = self.calc_uvw.isChecked()
            params['u_range'] = [self.u_range_min.value(), self.u_range_max.value()]
            params['v_width'] = self.v_width.value()
            params['w_width'] = self.w_width.value()
            params['du'] = self.du.value()
            params['dvw'] = self.dvw.value()
        params['grid_drz'] = self.grid_drz.value() # This is also used outside the group

        # Advanced Simulation Settings
        params['verbose'] = self.verbose.isChecked()
        params['calc_halo'] = self.calc_halo.isChecked()
        params['calc_density'] = self.calc_density.isChecked()
        params['calc_photon_origin'] = self.calc_photon_origin.isChecked()
        params['respawn_if_aperture_is_hit'] = self.respawn_aperture.isChecked()
        skip_text = self.skip_options.text().strip()
        if skip_text:
            params['skip'] = [s.strip() for s in skip_text.split(',')]

        # Impurity Settings
        impurities_text = self.impurities.text().strip()
        if impurities_text:
            params['impurities'] = [s.strip().capitalize() for s in impurities_text.split(',')]
        if self.path_to_tables.text():
            params['path_to_tables'] = self.path_to_tables.text()
        params['load_raw_data'] = self.load_raw_data.isChecked()

        # Extended Emission
        if self.extended_emission_group.isChecked():
            params['calc_extended_emission'] = True
            params['transition'] = [self.trans_n_max.value(), self.trans_n_min.value()]

        # Mode-specific params
        if mode == "W7-X":
            params['machine'] = "W7X"
            # LOS
            params['los_shot'] = self.w7x_los_shot.text()
            params['los_head'] = self.w7x_los_head.text()
            if self.w7x_los_file.text():
                params['los_file'] = self.w7x_los_file.text()
            params['los_default'] = self.w7x_los_default.isChecked()
            params['los_new'] = self.w7x_los_new.isChecked()
            # Equilibrium
            if self.w7x_progID.text():
                params['progID'] = self.w7x_progID.text()
            if self.w7x_vmecID.text():
                params['vmecID'] = self.w7x_vmecID.text()
            if self.w7x_eq_path.text():
                params['eq_path'] = self.w7x_eq_path.text()
            params['extended_vmec_factor'] = self.w7x_extended_vmec_factor.value()
            params['b0_factor'] = self.w7x_b0_factor.value()
            params['drz'] = self.w7x_drz.value()
            params['phi_ran'] = [self.w7x_phi_ran_min.value() * np.pi, self.w7x_phi_ran_max.value() * np.pi]
            # Profiles
            if self.w7x_prof_path.text():
                params['prof_path'] = self.w7x_prof_path.text()
            # NBI
            params['shot_num'] = self.w7x_shot_num.text()
            params['t_start'] = self.w7x_t_start.value()
            params['t_stop'] = self.w7x_t_stop.value()
            nbi_frac_text = self.w7x_nbi_cur_frac.text().strip()
            if nbi_frac_text:
                try:
                    params['nbi_cur_frac'] = [float(s) for s in nbi_frac_text.split(',')]
                except ValueError:
                    print("Warning: Could not parse NBI Current Fractions. Ignoring.") # Or raise an error
            params['nbi_debug'] = self.w7x_nbi_debug.isChecked()
            params['nbi_default'] = self.w7x_nbi_default.isChecked()
            params['ion_mass'] = self.w7x_ion_mass.value()
            params['nbi_mass'] = self.w7x_nbi_mass.value()

        elif mode == "Fortran FIDASIM":
            params['FIDASIM_check'] = True
            params['directory'] = self.f90_directory.text()
            params['runid'] = self.f90_runid.text()
            params['geqdsk'] = self.f90_geqdsk.text()

        elif mode == "TRANSP":
            params['transp'] = True
            params['directory'] = self.transp_directory.text()
            params['runid'] = self.transp_runid.text()
            params['time'] = self.transp_time.value()
            params['ion_mass'] = self.transp_ion_mass.value()
            params['nbi_mass'] = self.transp_nbi_mass.value()
            params['Bt_sign'] = self.transp_bt_sign.value()
            params['Ip_sign'] = self.transp_ip_sign.value()
            params['phi_ran'] = [self.transp_phi_ran_min.value() * np.pi, self.transp_phi_ran_max.value() * np.pi]
            params['ntheta'] = self.transp_ntheta.value()
            params['nr'] = self.transp_nr.value()
            params['nz'] = self.transp_nz.value()
            params['n_decay'] = self.transp_ne_decay.value()
            params['te_decay'] = self.transp_te_decay.value()
            params['ti_decay'] = self.transp_ti_decay.value()
            params['omega_decay'] = self.transp_omega_decay.value()
            s_new_text = self.transp_s_new.text().strip()
            if s_new_text:
                try:
                    params['s_new'] = [float(s) for s in s_new_text.split(',')]
                except ValueError:
                    print("Warning: Could not parse New S-Domain. Ignoring.")
            params['prof_plot'] = self.transp_prof_plot.isChecked()
            params['nbi_plot'] = self.transp_nbi_plot.isChecked()

        return params