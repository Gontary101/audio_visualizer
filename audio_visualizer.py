import sys
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QProgressBar, QSlider
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from moviepy.editor import VideoClip, AudioFileClip, CompositeVideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import multiprocessing as mp
from functools import partial

class AudioVisualizer:
    def __init__(self, audio_path, frame_length=1000, fps=30, amplitude_scale=2.0):
        self.audio_path = audio_path
        self.frame_length = frame_length
        self.fps = fps
        self.amplitude_scale = amplitude_scale
        self.y, self.sr = librosa.load(audio_path)
        self.duration = len(self.y) / self.sr

    def create_frame(self, t):
        start_sample = int(t * self.sr)
        end_sample = min(start_sample + self.frame_length, len(self.y))
        frame = self.y[start_sample:end_sample]
        
        if len(frame) < self.frame_length:
            frame = np.pad(frame, (0, self.frame_length - len(frame)))
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_ylim(-self.amplitude_scale, self.amplitude_scale)
        ax.set_xlim(0, self.frame_length)
        ax.axis('off')
        
        bars = ax.bar(range(self.frame_length), frame * self.amplitude_scale, width=1, color='black', align='edge')
        
        img = mplfig_to_npimage(fig)
        plt.close(fig)
        return img

    def create_frames_batch(self, batch):
        return [self.create_frame(t) for t in batch]

    def generate_video(self, output_path, progress_callback=None):
        total_frames = int(self.duration * self.fps)
        time_points = np.linspace(0, self.duration, total_frames, endpoint=False)

        num_cores = mp.cpu_count()
        batches = np.array_split(time_points, num_cores)

        with mp.Pool(num_cores) as pool:
            frames = []
            for i, batch_frames in enumerate(pool.imap(self.create_frames_batch, batches)):
                frames.extend(batch_frames)
                if progress_callback:
                    progress = int((i + 1) / len(batches) * 100)
                    progress_callback.emit(progress)

        def make_frame(t):
            frame_index = min(int(t * self.fps), len(frames) - 1)
            return frames[frame_index]

        video = VideoClip(make_frame, duration=self.duration)
        audio = AudioFileClip(self.audio_path)
        final_video = CompositeVideoClip([video.set_audio(audio)])

        final_video.write_videofile(output_path, fps=self.fps, audio_codec='aac')

class VideoGenerationThread(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, visualizer, output_path):
        super().__init__()
        self.visualizer = visualizer
        self.output_path = output_path

    def run(self):
        self.visualizer.generate_video(self.output_path, self.progress_updated)
        self.finished.emit()

class AudioVisualizerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer")
        self.setGeometry(100, 100, 400, 300)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.input_label = QLabel("No audio file selected")
        layout.addWidget(self.input_label)

        self.select_button = QPushButton("Select Audio File")
        self.select_button.clicked.connect(self.select_audio_file)
        layout.addWidget(self.select_button)

        self.amplitude_slider = QSlider(Qt.Horizontal)
        self.amplitude_slider.setMinimum(10)
        self.amplitude_slider.setMaximum(50)
        self.amplitude_slider.setValue(20)
        self.amplitude_slider.setTickPosition(QSlider.TicksBelow)
        self.amplitude_slider.setTickInterval(10)
        layout.addWidget(QLabel("Amplitude Scale:"))
        layout.addWidget(self.amplitude_slider)

        self.generate_button = QPushButton("Generate Video")
        self.generate_button.clicked.connect(self.generate_video)
        self.generate_button.setEnabled(False)
        layout.addWidget(self.generate_button)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.audio_path = None

    def select_audio_file(self):
        file_dialog = QFileDialog()
        self.audio_path, _ = file_dialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.mp3 *.wav)")
        if self.audio_path:
            self.input_label.setText(f"Selected: {os.path.basename(self.audio_path)}")
            self.generate_button.setEnabled(True)

    def generate_video(self):
        if not self.audio_path:
            return

        output_path, _ = QFileDialog.getSaveFileName(self, "Save Video As", "", "MP4 Files (*.mp4)")
        if not output_path:
            return

        amplitude_scale = self.amplitude_slider.value() / 10
        visualizer = AudioVisualizer(self.audio_path, amplitude_scale=amplitude_scale)
        
        self.generation_thread = VideoGenerationThread(visualizer, output_path)
        self.generation_thread.progress_updated.connect(self.update_progress)
        self.generation_thread.finished.connect(self.generation_finished)
        
        self.generate_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.generation_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def generation_finished(self):
        self.generate_button.setEnabled(True)
        self.progress_bar.setValue(100)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioVisualizerGUI()
    window.show()
    sys.exit(app.exec_())