# video_generator.py

import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import Optional, Tuple
from visualizer import Visualizer, VisualizationType

class VideoGenerator:
    """Generates video files from audio visualizations"""

    def __init__(self, 
                 output_path: str,
                 resolution: Tuple[int, int] = (1280, 720),
                 frame_rate: int = 30,
                 quality: int = 85,
                 visualizer: Optional[Visualizer] = None):
        """
        Initialize VideoGenerator with specified parameters
        
        Args:
            output_path: File path for the output video
            resolution: Resolution of the output video (width, height)
            frame_rate: Frame rate of the output video
            quality: Quality setting (usually between 0 and 100)
            visualizer: Instance of Visualizer for rendering frames
        """
        self.output_path = output_path
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.quality = quality
        self.visualizer = visualizer or Visualizer()
        self.frames = []

    def generate_video(self, audio_data: np.ndarray, sample_rate: int, duration: float) -> None:
        """Generate a video file from audio data and visualizations"""
        # Calculate the total number of frames based on duration and frame rate
        total_frames = int(self.frame_rate * duration)
        samples_per_frame = len(audio_data) // total_frames

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
        video_writer = cv2.VideoWriter(self.output_path, fourcc, self.frame_rate, self.resolution)

        # Generate frames and write to video
        for frame_idx in range(total_frames):
            start_sample = frame_idx * samples_per_frame
            end_sample = start_sample + samples_per_frame
            frame_audio_data = audio_data[start_sample:end_sample]

            # Plot the frame using the visualizer
            frame = self._generate_frame(frame_audio_data, sample_rate)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            video_writer.write(frame_bgr)

        # Release video writer
        video_writer.release()
        print(f"Video saved to {self.output_path}")

    def _generate_frame(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Generate a single frame image for the video based on audio data"""
        fig, ax = plt.subplots(figsize=(self.resolution[0] / 100, self.resolution[1] / 100))
        ax.axis('off')  # Remove axis for a clean frame

        # Use the visualizer to render the frame
        self.visualizer.plot(audio_data, sample_rate)

        # Save the plot to a numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)  # Close the plot to free memory
        return frame

    def set_visualizer_settings(self, visualization_type: VisualizationType, color_scheme: Optional[str] = None) -> None:
        """Set visualization settings for the video generation"""
        self.visualizer.set_visualization_type(visualization_type)
        if color_scheme:
            self.visualizer.update_settings({"color_scheme": color_scheme})

    def set_quality(self, quality: int) -> None:
        """Adjust video quality settings"""
        if 0 <= quality <= 100:
            self.quality = quality
        else:
            raise ValueError("Quality should be between 0 and 100")

    def set_frame_rate(self, frame_rate: int) -> None:
        """Adjust video frame rate"""
        if frame_rate > 0:
            self.frame_rate = frame_rate
        else:
            raise ValueError("Frame rate should be positive")

    def set_resolution(self, resolution: Tuple[int, int]) -> None:
        """Set resolution of the video"""
        self.resolution = resolution
