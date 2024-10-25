# visualizer.py

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import Optional, Dict

class VisualizationType(Enum):
    """Types of visualizations available"""
    WAVEFORM = "waveform"
    SPECTRUM = "spectrum"
    CIRCULAR = "circular"
    PARTICLE = "particle"
    THREE_D = "3d"

class Visualizer:
    """Visualizer class to create different styles of audio visualizations"""

    def __init__(self, visualization_type: VisualizationType = VisualizationType.WAVEFORM, color_scheme: Optional[str] = "default"):
        self.visualization_type = visualization_type
        self.color_scheme = color_scheme
        self.fig, self.ax = plt.subplots()
    
    def set_visualization_type(self, visualization_type: VisualizationType) -> None:
        """Set the type of visualization"""
        self.visualization_type = visualization_type

    def plot(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Plot the selected visualization based on type"""
        self.ax.clear()
        if self.visualization_type == VisualizationType.WAVEFORM:
            self._plot_waveform(audio_data, sample_rate)
        elif self.visualization_type == VisualizationType.SPECTRUM:
            self._plot_spectrum(audio_data, sample_rate)
        elif self.visualization_type == VisualizationType.CIRCULAR:
            self._plot_circular(audio_data)
        elif self.visualization_type == VisualizationType.PARTICLE:
            self._plot_particle(audio_data)
        elif self.visualization_type == VisualizationType.THREE_D:
            self._plot_3d(audio_data)
        else:
            raise ValueError("Unknown visualization type")

        plt.show()

    def _plot_waveform(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Plot the waveform of the audio signal"""
        times = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
        self.ax.plot(times, audio_data, color=self.color_scheme)
        self.ax.set_title("Waveform")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

    def _plot_spectrum(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Plot the frequency spectrum of the audio signal"""
        fft_result = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
        self.ax.plot(freqs[:len(freqs) // 2], np.abs(fft_result[:len(freqs) // 2]), color=self.color_scheme)
        self.ax.set_title("Frequency Spectrum")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude")

    def _plot_circular(self, audio_data: np.ndarray) -> None:
        """Plot a circular visualization (mock implementation)"""
        theta = np.linspace(0, 2 * np.pi, len(audio_data))
        radius = np.abs(audio_data)
        self.ax.plot(radius * np.cos(theta), radius * np.sin(theta), color=self.color_scheme)
        self.ax.set_title("Circular Visualization")

    def _plot_particle(self, audio_data: np.ndarray) -> None:
        """Plot a particle system visualization (placeholder)"""
        self.ax.scatter(np.random.rand(len(audio_data)), np.random.rand(len(audio_data)), alpha=0.6, color=self.color_scheme)
        self.ax.set_title("Particle Visualization")

    def _plot_3d(self, audio_data: np.ndarray) -> None:
        """Plot a 3D visualization (mock implementation)"""
        # A 3D mock plot example
        self.ax = plt.axes(projection="3d")
        z = np.linspace(0, 1, len(audio_data))
        x = np.cos(z * 2 * np.pi) * audio_data
        y = np.sin(z * 2 * np.pi) * audio_data
        self.ax.plot3D(x, y, z, color=self.color_scheme)
        self.ax.set_title("3D Visualization")

    def update_settings(self, settings: Dict) -> None:
        """Update visualizer settings like color scheme or animation speed"""
        self.color_scheme = settings.get("color_scheme", self.color_scheme)
