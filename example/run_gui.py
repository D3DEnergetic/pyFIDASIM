# run_gui.py
import sys
from PyQt6.QtWidgets import QApplication
# We need to import pyqtgraph to ensure it's found
import pyqtgraph as pg

# Set pyqtgraph configuration options
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

from pyfidasim.gui.main_window import MainWindow

if __name__ == '__main__':
    # The 'matplotlib.use()' call is no longer necessary as we are not using it for plotting.
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())