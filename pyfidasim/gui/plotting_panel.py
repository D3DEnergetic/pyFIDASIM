# gui/plotting_panel.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QCheckBox, QSpinBox, QLineEdit, QPushButton, QTabWidget, QLabel
)
from PyQt6.QtCore import pyqtSignal
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from .plotting_routines import (
    plot_los, plot_spectra, plot_r_density, plot_u_density, plot_profiles
)

class PlottingPanel(QWidget):
    """
    A widget that contains all plotting controls and displays,
    organized into a tabbed interface.
    """
    # Define signals to request data from the main window/worker
    request_prep_data = pyqtSignal()
    request_full_run_data = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.results = None # To store the latest full simulation results
        self.prep_results = None # To store the latest preparation results

        # Create the tab widget
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Create each tab
        self._create_los_tab()
        self._create_spectra_tab()
        self._create_density_tab()
        self._create_profiles_tab()

        self._connect_signals()

    def _create_los_tab(self):
        """Creates the tab for LOS Geometry plotting."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls Group
        controls_group = QGroupBox("Plotting Options")
        controls_layout = QFormLayout(controls_group)
        self.plot_crossed_cells_cb = QCheckBox()
        self.plot_ssurf_le = QLineEdit("1.0")
        self.plot_ssurf_le.setToolTip("Enter comma-separated s-surface values")
        self.update_los_plot_button = QPushButton("Update Geometry Plot") # Renamed for clarity

        controls_layout.addRow("Plot Crossed Cells:", self.plot_crossed_cells_cb)
        controls_layout.addRow("S-Surfaces:", self.plot_ssurf_le)
        controls_layout.addRow(self.update_los_plot_button)

        # Plot Widget
        self.los_plot_widget = gl.GLViewWidget()
        self.los_plot_widget.opts['distance'] = 400

        layout.addWidget(controls_group)
        layout.addWidget(self.los_plot_widget, 1) # Give it stretch factor
        self.tabs.addTab(tab, "LOS Geometry")

    def _create_spectra_tab(self):
        """Creates the tab for spectra plotting."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls Group
        controls_group = QGroupBox("Plotting Options")
        controls_layout = QFormLayout(controls_group)

        # Energy component checkboxes
        self.spec_full_cb = QCheckBox("Full"); self.spec_full_cb.setChecked(True)
        self.spec_half_cb = QCheckBox("Half"); self.spec_half_cb.setChecked(True)
        self.spec_third_cb = QCheckBox("Third"); self.spec_third_cb.setChecked(True)
        self.spec_halo_cb = QCheckBox("Halo"); self.spec_halo_cb.setChecked(True)
        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.spec_full_cb)
        hbox_labels.addWidget(self.spec_half_cb)
        hbox_labels.addWidget(self.spec_third_cb)
        hbox_labels.addWidget(self.spec_halo_cb)
        controls_layout.addRow("Components:", hbox_labels)

        # LOS selector
        self.ilos_selector = QSpinBox()
        self.ilos_selector.setRange(0, 0)
        self.ilos_selector.setEnabled(False)
        self.los_name_label = QLabel("N/A")
        controls_layout.addRow("Select LOS:", self.ilos_selector)
        controls_layout.addRow("LOS Name:", self.los_name_label)

        # Plot Widget
        self.spectra_plot_widget = pg.PlotWidget()

        layout.addWidget(controls_group)
        layout.addWidget(self.spectra_plot_widget, 1)
        self.tabs.addTab(tab, "Spectra")

    def _create_density_tab(self):
        """Creates the tab for density profile plotting."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- R-Density Plot ---
        r_density_group = QGroupBox("Radial (R) Density")
        r_layout = QVBoxLayout(r_density_group)
        # R-Density Controls
        self.r_dens_full_cb = QCheckBox("Full"); self.r_dens_full_cb.setChecked(True)
        self.r_dens_half_cb = QCheckBox("Half"); self.r_dens_half_cb.setChecked(True)
        self.r_dens_third_cb = QCheckBox("Third"); self.r_dens_third_cb.setChecked(True)
        self.r_dens_halo_cb = QCheckBox("Halo"); self.r_dens_halo_cb.setChecked(True)
        r_hbox_labels = QHBoxLayout()
        r_hbox_labels.addWidget(self.r_dens_full_cb)
        r_hbox_labels.addWidget(self.r_dens_half_cb)
        r_hbox_labels.addWidget(self.r_dens_third_cb)
        r_hbox_labels.addWidget(self.r_dens_halo_cb)
        # R-Density Plot Widget
        self.r_density_plot_widget = pg.PlotWidget()
        r_layout.addLayout(r_hbox_labels)
        r_layout.addWidget(self.r_density_plot_widget)

        # --- U-Density Plot ---
        u_density_group = QGroupBox("Beam-Aligned (U) Density")
        u_layout = QVBoxLayout(u_density_group)
        # U-Density Controls
        u_controls_layout = QFormLayout()
        self.u_dens_full_cb = QCheckBox("Full"); self.u_dens_full_cb.setChecked(True)
        self.u_dens_half_cb = QCheckBox("Half"); self.u_dens_half_cb.setChecked(True)
        self.u_dens_third_cb = QCheckBox("Third"); self.u_dens_third_cb.setChecked(True)
        u_hbox_labels = QHBoxLayout()
        u_hbox_labels.addWidget(self.u_dens_full_cb)
        u_hbox_labels.addWidget(self.u_dens_half_cb)
        u_hbox_labels.addWidget(self.u_dens_third_cb)
        self.eng_lvl_selector = QSpinBox()
        self.eng_lvl_selector.setRange(1, 3); self.eng_lvl_selector.setValue(1) # n=1 to n=3
        u_controls_layout.addRow("Components:", u_hbox_labels)
        u_controls_layout.addRow("Energy Level (n):", self.eng_lvl_selector)
        # U-Density Plot Widget
        self.u_density_plot_widget = pg.PlotWidget()
        u_layout.addLayout(u_controls_layout)
        u_layout.addWidget(self.u_density_plot_widget)

        layout.addWidget(r_density_group)
        layout.addWidget(u_density_group)
        self.tabs.addTab(tab, "Density Profiles")

    def _create_profiles_tab(self):
        """Creates the tab for plotting input plasma profiles."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.update_profiles_plot_button = QPushButton("Update Input Profiles Plot")
        layout.addWidget(self.update_profiles_plot_button)

        self.profiles_plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.profiles_plot_widget, 1)

        self.tabs.addTab(tab, "Input Profiles")

    def _connect_signals(self):
        """Connect widget signals to plotting update slots."""
        # The main window now triggers the data generation.
        # These buttons will just update the plot with the latest data.
        self.update_los_plot_button.clicked.connect(self.update_los_plot)
        self.update_profiles_plot_button.clicked.connect(self.update_input_profiles_plot)

        # Connect all plot option changes to the update method for live updates
        self.ilos_selector.valueChanged.connect(self.update_plots)
        self.spec_full_cb.stateChanged.connect(self.update_plots)
        self.spec_half_cb.stateChanged.connect(self.update_plots)
        self.spec_third_cb.stateChanged.connect(self.update_plots)
        self.spec_halo_cb.stateChanged.connect(self.update_plots)

        self.r_dens_full_cb.stateChanged.connect(self.update_plots)
        self.r_dens_half_cb.stateChanged.connect(self.update_plots)
        self.r_dens_third_cb.stateChanged.connect(self.update_plots)
        self.r_dens_halo_cb.stateChanged.connect(self.update_plots)

        self.u_dens_full_cb.stateChanged.connect(self.update_plots)
        self.u_dens_half_cb.stateChanged.connect(self.update_plots)
        self.u_dens_third_cb.stateChanged.connect(self.update_plots)
        self.eng_lvl_selector.valueChanged.connect(self.update_plots)
    
    def on_prep_results_ready(self, prep_results):
        """Slot to receive preparation results and trigger initial plots."""
        self.prep_results = prep_results
        self.update_los_plot()
        self.update_input_profiles_plot()

    def on_full_results_ready(self, full_results):
        """Slot to receive full results, enable controls, and plot."""
        self.results = full_results
        
        # Enable spectra controls
        nlos = self.results.get('spec', {}).get('nlos', 0)
        if nlos > 0:
            self.ilos_selector.setEnabled(True)
            self.ilos_selector.setRange(0, nlos - 1)
            # Make sure a valid value is selected to trigger initial plot
            self.ilos_selector.setValue(min(self.ilos_selector.value(), nlos - 1))
        else:
            self.ilos_selector.setEnabled(False)
            self.ilos_selector.setRange(0, 0)
        
        # Also update the prep_results so the LOS/Profile plots can be updated
        # with the data from the full run if desired.
        self.prep_results = {
            'sgrid': full_results.get('fields'),
            'grid3d': full_results.get('grid3d'),
            'spec': full_results.get('spec'),
            'nbi': full_results.get('nbi'),
            'profiles': full_results.get('fidasim_params', {}).get('profiles_data') # Assuming profiles are passed back
        }
        
        self.update_plots() # Update all plots with the new data

    def update_los_plot(self):
        if not self.prep_results:
            return
        
        ssurf_text = self.plot_ssurf_le.text()
        try:
            plot_ssurf = [float(s.strip()) for s in ssurf_text.split(',')]
        except (ValueError, AttributeError):
            plot_ssurf = [1.0]

        # Call the plotting routine, passing the widget to be drawn on
        plot_los(
            view_widget=self.los_plot_widget,
            sgrid=self.prep_results.get('sgrid'),
            grid3d=self.prep_results.get('grid3d'),
            spec=self.prep_results.get('spec'),
            nbi=self.prep_results.get('nbi'),
            plot_crossed_cells=self.plot_crossed_cells_cb.isChecked(),
            plot_ssurf=plot_ssurf
        )

    def update_plots(self, _=None): # _=None to accept unused stateChanged arguments
        """Update all plots based on current control values and available data."""
        if not self.results:
            return # No data to plot

        # --- Update Spectra ---
        spec_labels = []
        if self.spec_full_cb.isChecked(): spec_labels.append('full')
        if self.spec_half_cb.isChecked(): spec_labels.append('half')
        if self.spec_third_cb.isChecked(): spec_labels.append('third')
        if self.spec_halo_cb.isChecked(): spec_labels.append('halo')
        
        ilos = self.ilos_selector.value()
        plot_spectra(self.spectra_plot_widget, self.results.get('spec'), ilos, spec_labels)
        if self.results.get('spec') and ilos < self.results['spec'].get('nlos', 0):
            self.los_name_label.setText(self.results['spec']['losname'][ilos])
        
        # --- Update R-Density ---
        r_dens_labels = []
        if self.r_dens_full_cb.isChecked(): r_dens_labels.append('full')
        if self.r_dens_half_cb.isChecked(): r_dens_labels.append('half')
        if self.r_dens_third_cb.isChecked(): r_dens_labels.append('third')
        if self.r_dens_halo_cb.isChecked(): r_dens_labels.append('halo')
        plot_r_density(self.r_density_plot_widget, self.results.get('grid3d'), r_dens_labels)

        # --- Update U-Density ---
        u_dens_labels = []
        if self.u_dens_full_cb.isChecked(): u_dens_labels.append('full')
        if self.u_dens_half_cb.isChecked(): u_dens_labels.append('half')
        if self.u_dens_third_cb.isChecked(): u_dens_labels.append('third')
        eng_lvl = self.eng_lvl_selector.value() - 1 # Convert from 1-based to 0-based
        plot_u_density(self.u_density_plot_widget, self.results.get('grid3d'), u_dens_labels, eng_lvl)
    
    def update_input_profiles_plot(self):
        """Update the input profiles plot."""
        if not self.prep_results or 'profiles' not in self.prep_results:
            # Maybe request prep data if none is available
            print("No profile data available. Run preparation first.")
            return

        plot_profiles(self.profiles_plot_widget, self.prep_results.get('profiles'))