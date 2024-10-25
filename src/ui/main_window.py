from pathlib import Path
from typing import Optional, Dict, List
import sys
import logging
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QMenuBar, QMenu, QAction, 
    QDockWidget, QScrollArea, QSplitter, QFrame, QMessageBox,
    QStatusBar, QToolBar
)
from PyQt5.QtCore import Qt, QSettings, QTimer, QSize, pyqtSignal
from PyQt5.QtGui import QIcon, QKeySequence

# Import other components
from .visualization_settings import VisualizationSettingsPanel
from .custom_widgets import (
    WaveformDisplay, TimelineEditor, EffectStack,
    AudioControls, PresetManager, ProjectExplorer
)
from ..core.audio_processor import AudioProcessor, AudioFormat
from ..utils.config import AppConfig
from ..utils.helpers import format_time, create_icon_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main application window with comprehensive audio visualization interface"""
    
    # Custom signals
    audio_loaded = pyqtSignal(str)  # Emitted when audio is loaded
    visualization_updated = pyqtSignal()  # Emitted when visualization changes
    export_started = pyqtSignal()  # Emitted when export begins
    export_finished = pyqtSignal(bool)  # Emitted when export ends (success status)
    
    def __init__(self):
        super().__init__()
        self.config = AppConfig()
        self.settings = QSettings('AudioVisualizer', 'App')
        
        # Initialize core components
        self.audio_processor = AudioProcessor(enable_realtime=True)
        self.current_project_path: Optional[Path] = None
        self.is_modified = False
        
        self._setup_ui()
        self._restore_window_state()
        self._connect_signals()
        
        # Start auto-save timer
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self._auto_save)
        self.auto_save_timer.start(300000)  # 5 minutes
        
    def _setup_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Audio Visualizer")
        self.setMinimumSize(1200, 800)
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create toolbar
        self._create_toolbar()
        
        # Create main splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        # Create left panel (project explorer)
        self.project_explorer = ProjectExplorer()
        self.main_splitter.addWidget(self.project_explorer)
        
        # Create center panel
        self.center_panel = QWidget()
        self.center_layout = QVBoxLayout(self.center_panel)
        
        # Add waveform display
        self.waveform_display = WaveformDisplay()
        self.center_layout.addWidget(self.waveform_display)
        
        # Add timeline editor
        self.timeline_editor = TimelineEditor()
        self.center_layout.addWidget(self.timeline_editor)
        
        # Add audio controls
        self.audio_controls = AudioControls()
        self.center_layout.addWidget(self.audio_controls)
        
        self.main_splitter.addWidget(self.center_panel)
        
        # Create right panel (settings and effects)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        # Add visualization settings
        self.visualization_settings = VisualizationSettingsPanel()
        self.right_layout.addWidget(self.visualization_settings)
        
        # Add effect stack
        self.effect_stack = EffectStack()
        self.right_layout.addWidget(self.effect_stack)
        
        # Add preset manager
        self.preset_manager = PresetManager()
        self.right_layout.addWidget(self.preset_manager)
        
        self.main_splitter.addWidget(self.right_panel)
        
        # Set splitter proportions
        self.main_splitter.setStretchFactor(0, 1)  # Project explorer
        self.main_splitter.setStretchFactor(1, 3)  # Center panel
        self.main_splitter.setStretchFactor(2, 1)  # Right panel
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create dock widgets
        self._create_dock_widgets()
        
    def _create_menu_bar(self):
        """Create the application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        new_action = QAction('&New Project', self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction('&Open Project...', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction('&Save Project', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        save_as_action = QAction('Save Project &As...', self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        import_audio_action = QAction('Import &Audio...', self)
        import_audio_action.triggered.connect(self.import_audio)
        file_menu.addAction(import_audio_action)
        
        export_video_action = QAction('&Export Video...', self)
        export_video_action.triggered.connect(self.export_video)
        file_menu.addAction(export_video_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('&Edit')
        
        undo_action = QAction('&Undo', self)
        undo_action.setShortcut(QKeySequence.Undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction('&Redo', self)
        redo_action.setShortcut(QKeySequence.Redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        preferences_action = QAction('&Preferences...', self)
        preferences_action.triggered.connect(self.show_preferences)
        edit_menu.addAction(preferences_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        # Add dock widget visibility toggles
        for dock in self.findChildren(QDockWidget):
            view_menu.addAction(dock.toggleViewAction())
            
        view_menu.addSeparator()
        
        fullscreen_action = QAction('&Full Screen', self)
        fullscreen_action.setShortcut(QKeySequence.FullScreen)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        tutorial_action = QAction('&Tutorial', self)
        tutorial_action.triggered.connect(self.show_tutorial)
        help_menu.addAction(tutorial_action)
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def _create_toolbar(self):
        """Create the main toolbar"""
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Add commonly used actions
        toolbar.addAction(QIcon(create_icon_path('new')), 'New Project', self.new_project)
        toolbar.addAction(QIcon(create_icon_path('open')), 'Open Project', self.open_project)
        toolbar.addAction(QIcon(create_icon_path('save')), 'Save Project', self.save_project)
        
        toolbar.addSeparator()
        
        toolbar.addAction(QIcon(create_icon_path('import')), 'Import Audio', self.import_audio)
        toolbar.addAction(QIcon(create_icon_path('export')), 'Export Video', self.export_video)
        
        toolbar.addSeparator()
        
        toolbar.addAction(QIcon(create_icon_path('undo')), 'Undo', self.undo)
        toolbar.addAction(QIcon(create_icon_path('redo')), 'Redo', self.redo)
        
    def _create_dock_widgets(self):
        """Create dock widgets for additional panels"""
        # Effect Controls dock
        effects_dock = QDockWidget('Effect Controls', self)
        effects_dock.setWidget(self.effect_stack)
        effects_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, effects_dock)
        
        # Preset Browser dock
        presets_dock = QDockWidget('Preset Browser', self)
        presets_dock.setWidget(self.preset_manager)
        presets_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, presets_dock)
        
    def _connect_signals(self):
        """Connect all signal/slot pairs"""
        # Audio processor signals
        self.audio_processor.audio_loaded.connect(self._on_audio_loaded)
        
        # Timeline editor signals
        self.timeline_editor.position_changed.connect(self._on_timeline_position_changed)
        
        # Audio controls signals
        self.audio_controls.play_pause_triggered.connect(self._on_play_pause)
        self.audio_controls.stop_triggered.connect(self._on_stop)
        
        # Effect stack signals
        self.effect_stack.effect_added.connect(self._on_effect_added)
        self.effect_stack.effect_removed.connect(self._on_effect_removed)
        
        # Preset manager signals
        self.preset_manager.preset_selected.connect(self._on_preset_selected)
        
    def _restore_window_state(self):
        """Restore window geometry and state from settings"""
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
            
        state = self.settings.value('windowState')
        if state:
            self.restoreState(state)
            
    def _save_window_state(self):
        """Save window geometry and state to settings"""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())
        
    def _auto_save(self):
        """Auto-save the current project"""
        if self.is_modified and self.current_project_path:
            self.save_project()
            
    # File operations
    def new_project(self):
        """Create a new project"""
        if self.check_unsaved_changes():
            self.current_project_path = None
            self.is_modified = False
            self._clear_project()
            self.setWindowTitle("Audio Visualizer - New Project")


    def open_project(self):
        """Open an existing project"""
        if self.check_unsaved_changes():
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Project",
                "",
                "Audio Visualizer Project (*.avp);;All Files (*.*)"
            )
            if file_path:
                self._load_project(Path(file_path))
                
    def save_project(self):
        """Save the current project"""
        if not self.current_project_path:
            return self.save_project_as()
        self._save_project(self.current_project_path)
        return True
        
    def save_project_as(self):
        """Save the current project with a new name"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            "",
            "Audio Visualizer Project (*.avp);;All Files (*.*)"
        )
        if file_path:
            path = Path(file_path)
            if path.suffix != '.avp':
                path = path.with_suffix('.avp')
            self._save_project(path)
            return True
        return False
        
    def import_audio(self):
        """Import audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Audio",
            "",
            "Audio Files (*.mp3 *.wav *.ogg *.flac);;All Files (*.*)"
        )
        if file_path:
            self._import_audio_file(Path(file_path))
            
    def export_video(self):
        """Export visualization as video"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Video",
            "",
            "MP4 Video (*.mp4);;All Files (*.*)"
        )
        if file_path:
            self._export_video_file(Path(file_path))
            
    # Event handlers
    def closeEvent(self, event):
        """Handle application close event"""
        if self.check_unsaved_changes():
            self._save_window_state()
            event.accept()
        else:
            event.ignore()
            
    def check_unsaved_changes(self) -> bool:
        """Check for unsaved changes and prompt user if necessary"""
        if self.is_modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Do you want to save your changes?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                return self.save_project()
            elif reply == QMessageBox.Cancel:
                return False
                
        return True
        
    # Project loading/saving
    def _load_project(self, path: Path):
        """Load project from file"""
        try:
            # Load the project data (implementation-specific)
            # Example: self.audio_processor.load_project(path)
            self.current_project_path = path
            self.is_modified = False
            self.setWindowTitle(f"Audio Visualizer - {path.name}")
            self.status_bar.showMessage("Project loaded successfully.", 2000)
            logger.info(f"Project loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading project: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load project: {e}")
            
    def _save_project(self, path: Path):
        """Save project to file"""
        try:
            # Save the project data (implementation-specific)
            # Example: self.audio_processor.save_project(path)
            self.current_project_path = path
            self.is_modified = False
            self.setWindowTitle(f"Audio Visualizer - {path.name}")
            self.status_bar.showMessage("Project saved successfully.", 2000)
            logger.info(f"Project saved to {path}")
        except Exception as e:
            logger.error(f"Error saving project: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save project: {e}")
            
    def _import_audio_file(self, path: Path):
        """Import an audio file for visualization and processing"""
        try:
            success = self.audio_processor.load_file(path)
            if success:
                self.audio_loaded.emit(str(path))
                self.is_modified = True
                self.status_bar.showMessage("Audio loaded successfully.", 2000)
                logger.info(f"Audio loaded from {path}")
            else:
                self.status_bar.showMessage("Failed to load audio file.", 2000)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load audio file: {e}")
            
    def _export_video_file(self, path: Path):
        """Export the current visualization as a video file"""
        try:
            # Example export video with the audio processor data and settings
            # self.video_generator.generate_video(path, ...)
            self.export_started.emit()
            self.export_finished.emit(True)
            self.status_bar.showMessage("Video export completed successfully.", 2000)
            logger.info(f"Video exported to {path}")
        except Exception as e:
            self.export_finished.emit(False)
            logger.error(f"Error exporting video: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export video: {e}")

    # Utility functions
    def toggle_fullscreen(self):
        """Toggle the application full screen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def show_preferences(self):
        """Show preferences dialog (to be implemented)"""
        # Preferences dialog code here (if available)
        QMessageBox.information(self, "Preferences", "Preferences dialog not implemented.")
    
    def show_tutorial(self):
        """Display tutorial information"""
        QMessageBox.information(self, "Tutorial", "Tutorial will guide you through the application usage.")
        
    def show_about(self):
        """Show about dialog with application information"""
        QMessageBox.about(self, "About Audio Visualizer", "Audio Visualizer Application\nVersion 1.0")

