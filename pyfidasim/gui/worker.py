# gui/worker.py
import time
import traceback
from PyQt6.QtCore import QObject, pyqtSignal

from pyfidasim.input_prep import input_prep
from pyfidasim.main import calc_attenuation
import numpy as np

class PrepWorker(QObject):
    """
    Runs only the pyFIDASIM input preparation step.
    This is used to generate the necessary data for pre-simulation plotting,
    such as the line-of-sight geometry.
    """
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    results_ready = pyqtSignal(dict) # Emits prep results for plotting
    finished = pyqtSignal()

    def __init__(self, fidasim_params):
        super().__init__()
        self.fidasim_params = fidasim_params

    def run(self):
        """The input preparation logic."""
        try:
            self.progress.emit("Preparing inputs for plotting...")
            sim_settings, spec, tables, fields, profiles, nbi, grid3d, ncdf, PSF, fbm = input_prep(self.fidasim_params)
            self.progress.emit("Inputs prepared successfully.")

            # --- Special logic for NBI sources (affects NBI geometry in plot) ---
            if not self.fidasim_params.get('FIDASIM_check', False) and self.fidasim_params.get('machine') == 'W7X':
                self.progress.emit("Manually selecting NBI source Q7 for W7X case...")
                nbi['sources'] = ['Q7']
                pop_keys = ("Q1","Q2","Q3","Q4","Q5","Q6","Q8")
                for popper in pop_keys:
                    if popper in nbi:
                        nbi.pop(popper)

            # Bundle initial results into a dictionary formatted for plotting
            prep_results = {
                'sgrid': fields,  # The plot_los function expects 'sgrid', which is the 'fields' dict
                'grid3d': grid3d,
                'spec': spec,
                'nbi': nbi,
            }
            self.results_ready.emit(prep_results)

        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit()


class FullWorker(QObject):
    """
    Runs the full pyFIDASIM computation (preparation + attenuation).
    """
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    results_ready = pyqtSignal(dict) # Emits all final results for plotting
    finished = pyqtSignal()

    def __init__(self, fidasim_params):
        super().__init__()
        self.fidasim_params = fidasim_params

    def run(self):
        """The main simulation logic."""
        try:
            self.progress.emit("Preparing inputs...")
            sim_settings, spec, tables, fields, profiles, nbi, grid3d, ncdf, PSF, fbm = input_prep(self.fidasim_params)
            self.progress.emit("Inputs prepared successfully.")

            # --- Special logic from the original example script ---
            if not self.fidasim_params.get('FIDASIM_check', False) and self.fidasim_params.get('machine') == 'W7X':
                self.progress.emit("Manually selecting NBI source Q7 for W7X case...")
                nbi['sources'] = ['Q7']
                pop_keys = ("Q1","Q2","Q3","Q4","Q5","Q6","Q8")
                for popper in pop_keys:
                    if popper in nbi:
                        nbi.pop(popper)

            # --- Run pyFIDASIM Attenuation ---
            self.progress.emit("Running pyFIDASIM attenuation calculation... (This may take a while)")
            t1 = time.time()
            grid3d_out, spec_out = calc_attenuation(sim_settings, profiles, nbi, spec, fields, grid3d, tables, ncdf, PSF, fbm)
            t2 = time.time()
            self.progress.emit(f"Attenuation calculation finished. Time taken: {t2 - t1:.2f} seconds.")

            # Bundle all results into a dictionary for post-simulation plotting
            results = {
                'fidasim_params': self.fidasim_params,
                'spec': spec_out,
                'grid3d': grid3d_out,
                'fields': fields, # Original fields dict is needed for some plots
                'nbi': nbi,
            }
            self.results_ready.emit(results)

        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit()