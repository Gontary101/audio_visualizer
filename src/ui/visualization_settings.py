# visualization_settings.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QSlider, QColorDialog, QPushButton, QFormLayout
)
from PyQt5.QtCore import Qt
from .custom_widgets import ColorButton
from ..core.visualizer import VisualizationType

class VisualizationSettingsPanel(QWidget):
    """Settings panel for adjusting visualization properties"""

    def __init__(self):
        super().__init__()
        
        # Main layout
        layout = QVBoxLayout()
        
        # Visualization Style Dropdown
        self.style_dropdown = QComboBox()
        self.style_dropdown.addItems([vis_type.name for vis_type in VisualizationType])
        self.style_dropdown.currentIndexChanged.connect(self.update_visualization_style)
        
        # Color Scheme Selector
        self.color_button = ColorButton("Select Color")
        self.color_button.clicked.connect(self.select_color)
        
        # Effects and Properties (e.g., bar width, spacing)
        self.bar_width_slider = QSlider(Qt.Horizontal)
        self.bar_width_slider.setMinimum(1)
        self.bar_width_slider.setMaximum(50)
        self.bar_width_slider.setValue(10)
        self.bar_width_slider.setTickInterval(5)
        
        self.spacing_slider = QSlider(Qt.Horizontal)
        self.spacing_slider.setMinimum(1)
        self.spacing_slider.setMaximum(20)
        self.spacing_slider.setValue(5)
        self.spacing_slider.setTickInterval(1)
        
        # Organize into form layout
        form_layout = QFormLayout()
        form_layout.addRow("Style", self.style_dropdown)
        form_layout.addRow("Color Scheme", self.color_button)
        form_layout.addRow("Bar Width", self.bar_width_slider)
        form_layout.addRow("Spacing", self.spacing_slider)
        
        # Apply button
        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self.apply_changes)
        
        # Add to main layout
        layout.addLayout(form_layout)
        layout.addWidget(self.apply_button)
        layout.addStretch()
        
        self.setLayout(layout)

    def update_visualization_style(self):
        """Update visualization style based on dropdown selection"""
        # Implementation for setting visualization style if necessary
        pass
    
    def select_color(self):
        """Open color dialog to select color scheme"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.color_button.set_color(color.name())
    
    def apply_changes(self):
        """Apply changes to the visualization settings"""
        # Implement to emit signal or directly update visualization
        pass
