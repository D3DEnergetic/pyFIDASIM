# gui/main_window.py
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QPlainTextEdit,
    QSplitter, QMessageBox, QScrollArea, QHBoxLayout
)
from PyQt6.QtCore import QThread, Qt

# Import your project's modules
from .input_panel import InputPanel
from .plotting_panel import PlottingPanel
from .worker import PrepWorker, FullWorker # Correctly import the new worker classes

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyFIDASIM GUI")
        self.setGeometry(100, 100, 1600, 900)

        self.thread = None
        self.worker = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        # Left side: Input Panel in a scroll area
        self.input_panel = InputPanel()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.input_panel)
        scroll_area.setMinimumWidth(450) # Give it a bit more space

        # Right side: Controls, Plotting Panel, and Log
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Top control buttons
        control_layout = QHBoxLayout()
        self.prepare_button = QPushButton("Prepare & Plot Geometry")
        self.prepare_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.run_button = QPushButton("Run Full Simulation")
        self.run_button.setStyleSheet("font-size: 14px; padding: 8px; font-weight: bold;")
        control_layout.addWidget(self.prepare_button)
        control_layout.addWidget(self.run_button)

        # Main plotting panel (contains all tabs and plot controls)
        self.plotting_panel = PlottingPanel()

        # Log output
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("font-family: Consolas, monospaced;")

        # Splitter for plots and log
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self.plotting_panel)
        right_splitter.addWidget(self.log_output)
        right_splitter.setSizes([700, 200])

        right_layout.addLayout(control_layout)
        right_layout.addWidget(right_splitter)

        # Main splitter for input panel and right side
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(scroll_area)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([460, 1140])
        self.setCentralWidget(main_splitter)

    def _connect_signals(self):
        # Connect main action buttons to their respective worker functions
        self.prepare_button.clicked.connect(self.start_preparation)
        self.run_button.clicked.connect(self.start_full_run)

        # The plotting panel emits signals when it needs data
        self.plotting_panel.request_prep_data.connect(self.start_preparation)
        self.plotting_panel.request_full_run_data.connect(self.start_full_run)

    def _start_worker(self, worker_class):
        """Generic method to start a worker thread."""
        self.prepare_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.log_output.appendPlainText("\n" + "="*40)
        
        fidasim_params = self.input_panel.get_fidasim_params()

        self.thread = QThread()
        self.worker = worker_class(fidasim_params)
        self.worker.moveToThread(self.thread)

        self.worker.progress.connect(self.update_log)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.worker_finished)
        
        # Connect the results_ready signal to the correct slot in the plotting panel
        if isinstance(self.worker, PrepWorker):
            self.log_output.appendPlainText("Starting input preparation...")
            self.worker.results_ready.connect(self.plotting_panel.on_prep_results_ready)
        elif isinstance(self.worker, FullWorker):
            self.log_output.appendPlainText("Starting full simulation...")
            self.worker.results_ready.connect(self.plotting_panel.on_full_results_ready)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def start_preparation(self):
        self._start_worker(PrepWorker)

    def start_full_run(self):
        self._start_worker(FullWorker)

    def update_log(self, message):
        self.log_output.appendPlainText(message)

    def handle_error(self, traceback_str):
        self.log_output.appendPlainText(f"\n--- ERROR ---\n{traceback_str}")
        QMessageBox.critical(self, "Worker Error", "An error occurred. Check the log for details.")
        self.worker_finished() # Ensure UI is re-enabled on error

    def worker_finished(self):
        self.prepare_button.setEnabled(True)
        self.run_button.setEnabled(True)
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        self.log_output.appendPlainText("...process finished.")
        self.thread = None
        self.worker = None

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.worker_finished()
        event.accept()