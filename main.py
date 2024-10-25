# main.py

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QFile, QTextStream
from src.ui.main_window import MainWindow

def load_stylesheet(app):
    """Load and apply the custom QSS stylesheet"""
    file = QFile("src/ui/styles.qss")
    if file.open(QFile.ReadOnly | QFile.Text):
        stream = QTextStream(file)
        app.setStyleSheet(stream.readAll())
        file.close()

def main():
    """Main entry point for the Audio Visualizer application"""
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Visualizer")

    # Load and apply the stylesheet
    load_stylesheet(app)

    # Initialize and show main window
    main_window = MainWindow()
    main_window.show()

    # Run the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
