import sys
import os
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import cupy as cp

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
    QLabel, QVBoxLayout, QHBoxLayout, QWidget, QProgressBar, QSlider, 
    QComboBox, QSpinBox, QColorDialog, QStyle)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from moviepy.editor import VideoClip, AudioFileClip, CompositeVideoClip, TextClip
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.config import change_settings
import multiprocessing as mp
import tempfile
import logging
from skimage.transform import resize
from transformers import pipeline
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Configure MoviePy to use ImageMagick
if os.name == 'nt':  # Windows
    change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.0.10-Q16\magick.exe"})
else:  # Unix/Linux/MacOS
    change_settings({"IMAGEMAGICK_BINARY": "convert"})

# GPU accelerated signal processing functions
def process_audio_gpu(audio_data):
    # Transfer data to GPU
    audio_gpu = cp.asarray(audio_data)
    # Process on GPU
    processed = cp.abs(audio_gpu)
    # Return to CPU
    return cp.asnumpy(processed)

def apply_savgol(data, window_length, polyorder):
    return savgol_filter(data, window_length, polyorder)

def parallel_savgol(data, window_length, polyorder):
    # Split data into chunks for parallel processing
    n_cores = mp.cpu_count()
    chunk_size = len(data) // n_cores
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(
            apply_savgol,
            chunks,
            [window_length] * len(chunks),
            [polyorder] * len(chunks)
        ))
    return np.concatenate(results)

class AudioVisualizer:
    def __init__(self, audio_path, frame_length=2048, fps=30, amplitude_scale=40.0,
                 visualization_type='bars', color='#000000', background_color='#FFFFFF',
                 aspect_ratio='16:9', orientation='horizontal', subsample_factor=1):
        self.audio_path = audio_path
        self.frame_length = frame_length
        self.fps = fps
        self.amplitude_scale = amplitude_scale
        self.visualization_type = visualization_type
        self.color = color
        self.background_color = background_color
        self.aspect_ratio = aspect_ratio
        self.orientation = orientation
        self.subsample_factor = subsample_factor
        
        # Parse aspect ratio
        self.width_ratio, self.height_ratio = map(int, self.aspect_ratio.split(':'))
        if self.orientation == 'horizontal':
            self.target_width = 1920 if self.aspect_ratio == '16:9' else 1440
            self.target_height = self.target_width * self.height_ratio // self.width_ratio
        else:
            self.target_height = 1920 if self.aspect_ratio == '16:9' else 1440
            self.target_width = self.target_height * self.height_ratio // self.width_ratio

        if self.aspect_ratio == '1:1':
            self.target_width = self.target_height = 1080
        
        self.calculate_dimensions()
        
        try:
            # Load and process audio using GPU acceleration
            self.y, self.sr = librosa.load(audio_path, sr=None)
            
            # Process audio on GPU
            self.y = process_audio_gpu(self.y)
            
            # Parallel smoothing using all CPU cores
            window_length = 51
            polyorder = 3
            self.y = parallel_savgol(self.y, window_length, polyorder)
            
            self.duration = len(self.y) / self.sr
            
            # Calculate spectrogram using GPU
            hop_length = self.frame_length // 4
            # Transfer to GPU
            y_gpu = cp.asarray(self.y)
            self.spec = cp.asnumpy(cp.fft.rfft(y_gpu))
            self.spec_db = librosa.amplitude_to_db(np.abs(self.spec))
            
            # Store previous frame data
            self.prev_frame = cp.zeros(self.frame_length)
            
            # Generate subtitles using GPU-accelerated Whisper
            pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3-turbo", 
                chunk_length_s=30,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            self.transcription = pipe(
                audio_path,
                return_timestamps=True,
                generate_kwargs={
                    "task": "transcribe",
                    "language": "en"
                }
            )
            self.subtitles = self.transcription["chunks"]
            
        except Exception as e:
            logging.error(f"Error loading audio file: {e}")
            raise

    def calculate_dimensions(self):
        base_size = 10
        
        if self.orientation == 'horizontal':
            self.fig_width = base_size
            self.fig_height = (base_size * self.height_ratio) / self.width_ratio
        else:
            self.fig_width = (base_size * self.height_ratio) / self.width_ratio
            self.fig_height = base_size
            
        if self.aspect_ratio == '1:1':
            self.effective_amplitude = self.amplitude_scale * 1.5
        else:
            self.effective_amplitude = self.amplitude_scale
            
    def create_frame(self, t):
        start_sample = int(t * self.sr)
        end_sample = min(start_sample + self.frame_length, len(self.y))
        frame = self.y[start_sample:end_sample]
        
        if len(frame) < self.frame_length:
            frame = cp.pad(frame, (0, self.frame_length - len(frame)))
            
        # GPU-accelerated frame interpolation
        frame_gpu = cp.asarray(frame)
        prev_frame_gpu = self.prev_frame
        frame_gpu = 0.7 * frame_gpu + 0.3 * prev_frame_gpu
        self.prev_frame = frame_gpu
        frame = cp.asnumpy(frame_gpu)
        
        plt.ioff()
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height),
                              facecolor=self.background_color)
        
        if self.orientation == 'vertical':
            ax.set_xlim(-self.effective_amplitude, self.effective_amplitude)
            ax.set_ylim(0, self.frame_length)
        else:
            ax.set_ylim(-self.effective_amplitude, self.effective_amplitude)
            ax.set_xlim(0, self.frame_length)
            
        ax.axis('off')
        fig.patch.set_facecolor(self.background_color)
        
        indices = range(0, self.frame_length, self.subsample_factor)
        subsampled_frame = frame[indices]
        
        window_length = min(15, len(subsampled_frame) - 1)
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 3:
            subsampled_frame = savgol_filter(subsampled_frame, window_length, 2)
        
        min_amplitude = 0.01
        subsampled_frame = np.where(np.abs(subsampled_frame) < min_amplitude,
                                  min_amplitude * np.sign(subsampled_frame + 1e-10),
                                  subsampled_frame)
        
        if self.visualization_type == 'bars':
            if self.orientation == 'vertical':
                ax.barh(indices, subsampled_frame * self.effective_amplitude,
                   height=self.subsample_factor * 0.8,
                   color=self.color, align='edge', alpha=0.8)
            else:
                ax.bar(indices, subsampled_frame * self.effective_amplitude,
                  width=self.subsample_factor * 0.8,
                  color=self.color, align='edge', alpha=0.8)
        elif self.visualization_type == 'wave':
            if self.orientation == 'vertical':
                ax.plot(subsampled_frame * self.effective_amplitude, indices,
                   color=self.color, linewidth=2, alpha=0.8)
            else:
                ax.plot(indices, subsampled_frame * self.effective_amplitude,
                   color=self.color, linewidth=2, alpha=0.8)
                if np.max(np.abs(subsampled_frame)) < min_amplitude:
                    ax.axhline(y=0, color=self.color, linewidth=1, alpha=0.5)
        elif self.visualization_type == 'spectrum':
            spec_slice = self.spec_db[:, int(t * self.fps)]
            subsampled_spec = spec_slice[::self.subsample_factor]
            if self.orientation == 'vertical':
                ax.imshow([subsampled_spec], aspect='auto', cmap=plt.cm.get_cmap('viridis'),
                         origin='lower', interpolation='nearest')
            else:
                ax.imshow([[subsampled_spec]], aspect='auto', cmap=plt.cm.get_cmap('viridis'))
        
        ax.text(0.02, 0.98, f"Time: {t:.2f}s", transform=ax.transAxes,
                color=self.color, fontsize=10, alpha=0.7)
        
        img = mplfig_to_npimage(fig)
        plt.close(fig)
        img = resize(img, (self.target_height, self.target_width, 3), anti_aliasing=True)
        img = (img * 255).astype(np.uint8)
        return img

    def create_frames_batch(self, batch):
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self.create_frame, batch))

    def generate_video(self, output_path, progress_callback=None):
        total_frames = int(self.duration * self.fps)
        time_points = np.linspace(0, self.duration, total_frames, endpoint=False)

        if self.orientation == 'horizontal':
            width = 1920 if self.aspect_ratio == '16:9' else 1440
            height = width * self.height_ratio // self.width_ratio
        else:
            height = 1920 if self.aspect_ratio == '16:9' else 1440
            width = height * self.height_ratio // self.width_ratio

        if self.aspect_ratio == '1:1':
            width = height = 1080

        with tempfile.NamedTemporaryFile(suffix='.mmap', delete=False) as tmp:
            tmp_filename = tmp.name
            shape = (total_frames, self.target_height, self.target_width, 3)
            mmap_array = np.memmap(tmp_filename, dtype=np.uint8, mode='w+', shape=shape)

        batch_size = 50
        num_batches = (total_frames + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_frames)
            batch = time_points[start:end]
            
            frames = self.create_frames_batch(batch)
            mmap_array[start:end] = [np.array(frame) for frame in frames]

            if progress_callback:
                progress = int((i + 1) / num_batches * 100)
                progress_callback.emit(progress)
                QApplication.processEvents()

        def make_frame(t):
            frame_index = min(int(t * self.fps), total_frames - 1)
            return mmap_array[frame_index]

        video = VideoClip(make_frame, duration=self.duration)
        audio = AudioFileClip(self.audio_path)
        
        subtitle_clips = []
        for chunk in self.subtitles:
            start = chunk["timestamp"][0]
            end = chunk["timestamp"][1]
            text = chunk["text"]
            
            txt_clip = TextClip(
                text,
                fontsize=36,
                font='Arial-Bold', 
                color=self.color,
                bg_color=self.background_color,
                size=(width * 0.8, None),
                method='caption',
                stroke_color=self.color,
                stroke_width=1
            )
            txt_clip = txt_clip.set_start(start).set_end(end)
            txt_clip = txt_clip.set_position(('center', 0.8), relative=True)
            subtitle_clips.append(txt_clip)
        
        final_video = CompositeVideoClip([video.set_audio(audio)] + subtitle_clips)

        final_video.write_videofile(
            output_path,
            fps=self.fps,
            codec='libx264',
            audio_codec='aac',
            audio_bitrate='128k',
            preset='ultrafast',
            threads=mp.cpu_count()
        )

        del mmap_array
        os.unlink(tmp_filename)
        plt.close('all')

    def create_frames_batch(self,batch):
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            return list(executor.map(self.create_frame, batch))

class VideoGenerationThread(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, visualizer, output_path):
        super().__init__()
        self.visualizer = visualizer
        self.output_path = output_path

    def run(self):
        try:
            self.visualizer.generate_video(self.output_path, self.progress_updated)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class StyleableWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                color: #333333;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
            QProgressBar {
                border: 2px solid #2196F3;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #ffffff;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)

class AudioVisualizerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Audio Visualizer")
        self.setGeometry(100, 100, 600, 400)
        self.initUI()

    def initUI(self):
        main_widget = StyleableWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout()
        
        header = QLabel("Advanced Audio Visualizer")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        file_layout = QHBoxLayout()
        self.input_label = QLabel("No audio file selected")
        self.input_label.setStyleSheet("padding: 8px; background: white; border-radius: 4px;")
        file_layout.addWidget(self.input_label)
        
        self.select_button = QPushButton("Select Audio")
        self.select_button.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        file_layout.addWidget(self.select_button)
        layout.addLayout(file_layout)
        
        settings_layout = QHBoxLayout()
        
        viz_layout = QVBoxLayout()
        viz_layout.addWidget(QLabel("Visualization Type:"))
        self.viz_type = QComboBox()
        self.viz_type.addItems(['bars', 'wave', 'spectrum'])
        viz_layout.addWidget(self.viz_type)
        settings_layout.addLayout(viz_layout)
        
        fps_layout = QVBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_spinner = QSpinBox()
        self.fps_spinner.setRange(1, 60)
        self.fps_spinner.setValue(15)
        fps_layout.addWidget(self.fps_spinner)
        settings_layout.addLayout(fps_layout)
        
        aspect_layout = QVBoxLayout()
        aspect_layout.addWidget(QLabel("Aspect Ratio:"))
        self.aspect_ratio = QComboBox()
        self.aspect_ratio.addItems(['16:9', '4:3', '1:1'])
        aspect_layout.addWidget(self.aspect_ratio)
        settings_layout.addLayout(aspect_layout)
        
        orientation_layout = QVBoxLayout()
        orientation_layout.addWidget(QLabel("Orientation:"))
        self.orientation = QComboBox()
        self.orientation.addItems(['horizontal', 'vertical'])
        orientation_layout.addWidget(self.orientation)
        settings_layout.addLayout(orientation_layout)
        
        colors_layout = QVBoxLayout()
        colors_layout.addWidget(QLabel("Colors:"))
        color_buttons = QHBoxLayout()
        self.color_button = QPushButton("Viz Color")
        self.bg_color_button = QPushButton("BG Color")
        color_buttons.addWidget(self.color_button)
        color_buttons.addWidget(self.bg_color_button)
        colors_layout.addLayout(color_buttons)
        settings_layout.addLayout(colors_layout)
        
        layout.addLayout(settings_layout)
        
        amplitude_layout = QVBoxLayout()
        amplitude_layout.addWidget(QLabel("Amplitude Scale:"))
        self.amplitude_slider = QSlider(Qt.Horizontal)
        self.amplitude_slider.setMinimum(100)
        self.amplitude_slider.setMaximum(1000)
        self.amplitude_slider.setValue(400)
        self.amplitude_slider.setTickPosition(QSlider.TicksBelow)
        self.amplitude_slider.setTickInterval(100)
        amplitude_layout.addWidget(self.amplitude_slider)
        layout.addLayout(amplitude_layout)
        
        self.generate_button = QPushButton("Generate Video")
        self.generate_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.generate_button.setEnabled(False)
        layout.addWidget(self.generate_button)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        main_widget.setLayout(layout)
        
        self.select_button.clicked.connect(self.select_audio_file)
        self.generate_button.clicked.connect(self.generate_video)
        self.color_button.clicked.connect(self.select_color)
        self.bg_color_button.clicked.connect(self.select_bg_color)
        
        self.audio_path = None
        self.viz_color = "#000000"
        self.bg_color = "#FFFFFF"
        
        subsample_layout = QVBoxLayout()
        subsample_layout.addWidget(QLabel("Signal Density:"))
        self.subsample_spinner = QSpinBox()
        self.subsample_spinner.setRange(1, 16)
        self.subsample_spinner.setValue(4)
        self.subsample_spinner.setToolTip("Higher values will show fewer signals (1 = all signals)")
        subsample_layout.addWidget(self.subsample_spinner)
        settings_layout.addLayout(subsample_layout)
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(lambda: QApplication.processEvents())
        self.update_timer.start(100)

    def select_audio_file(self):
        file_dialog = QFileDialog()
        self.audio_path, _ = file_dialog.getOpenFileName(
            self, "Select Audio File", "", 
            "Audio Files (*.mp3 *.wav *.m4a *.ogg *.flac)")
        
        if self.audio_path:
            self.input_label.setText(f"Selected: {os.path.basename(self.audio_path)}")
            self.generate_button.setEnabled(True)
            self.status_label.setText("Ready to generate video")

    def select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.viz_color = color.name()
            self.color_button.setStyleSheet(f"background-color: {self.viz_color};")

    def select_bg_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.bg_color = color.name()
            self.bg_color_button.setStyleSheet(f"background-color: {self.bg_color};")

    def generate_video(self):
        if not self.audio_path:
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Video As", "", "MP4 Files (*.mp4)")
        
        if not output_path:
            return
        
        if not output_path.endswith(".mp4"):
            output_path += ".mp4"

        try:
            visualizer = AudioVisualizer(
                self.audio_path,
                fps=self.fps_spinner.value(),
                amplitude_scale=self.amplitude_slider.value() / 10,
                visualization_type=self.viz_type.currentText(),
                color=self.viz_color,
                background_color=self.bg_color,
                aspect_ratio=self.aspect_ratio.currentText(),
                orientation=self.orientation.currentText(),
                subsample_factor=self.subsample_spinner.value()
            )
            
            self.generation_thread = VideoGenerationThread(visualizer, output_path)
            self.generation_thread.progress_updated.connect(self.update_progress)
            self.generation_thread.finished.connect(self.generation_finished)
            self.generation_thread.error.connect(self.handle_error)
            
            self.generate_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.status_label.setText("Generating video...")
            self.generation_thread.start()
            
        except Exception as e:
            self.handle_error(str(e))

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        QApplication.processEvents()

    def generation_finished(self):
        self.generate_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText("Video generation completed successfully!")

    def handle_error(self, error_message):
        self.generate_button.setEnabled(True)
        self.status_label.setText(f"Error: {error_message}")
        self.progress_bar.setValue(0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        window = AudioVisualizerGUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"Application error: {e}")
        sys.exit(1)
