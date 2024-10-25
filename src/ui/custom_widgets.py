# custom_widgets.py

from PyQt5.QtWidgets import QPushButton, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QColor
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen

class ColorButton(QPushButton):
    """A button for selecting and displaying a color"""
    
    def __init__(self, text="Select Color"):
        super().__init__(text)
        self._color = QColor("white")
        self.setFixedSize(80, 30)
    
    def set_color(self, color: str):
        """Set button background to specified color"""
        self._color = QColor(color)
        self.setStyleSheet(f"background-color: {color}; border: 1px solid #ccc;")
    
    def color(self) -> str:
        return self._color.name()

class WaveformDisplay(QWidget):
    """A widget to display the waveform of audio data"""
    
    def __init__(self):
        super().__init__()
        self.audio_data = []
        self.setMinimumHeight(100)

    def paintEvent(self, event):
        if not self.audio_data:
            return
        
        painter = QPainter(self)
        pen = QPen(QColor("#4CAF50"), 2)
        painter.setPen(pen)
        
        width = self.width()
        height = self.height()
        data_len = len(self.audio_data)
        scale_factor = data_len / width
        max_amplitude = max(abs(min(self.audio_data)), max(self.audio_data))
        
        for x in range(width):
            index = int(x * scale_factor)
            y = int((self.audio_data[index] / max_amplitude) * (height / 2))
            painter.drawLine(x, height // 2 - y, x, height // 2 + y)

    def update_audio_data(self, audio_data):
        """Update audio data and repaint"""
        self.audio_data = audio_data
        self.update()

class TimelineEditor(QWidget):
    """A widget to edit and visualize the timeline of the audio"""
    
    position_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.timeline_label = QLabel("Timeline")
        layout.addWidget(self.timeline_label)

        self.timeline_slider = QSlider(Qt.Horizontal)
        layout.addWidget(self.timeline_slider)
        self.timeline_slider.valueChanged.connect(self.position_changed.emit)

class AudioControls(QWidget):
    """Widget to control audio playback (Play/Pause, Stop, etc.)"""
    
    play_pause_triggered = pyqtSignal()
    stop_triggered = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        
        self.play_button = QPushButton("Play/Pause")
        self.play_button.clicked.connect(self.play_pause_triggered.emit)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_triggered.emit)
        
        layout.addWidget(self.play_button)
        layout.addWidget(self.stop_button)
