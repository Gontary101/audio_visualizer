# Advanced Audio Visualizer

A GPU-accelerated Python desktop application for creating dynamic audio visualizations with custom styles and automatic subtitles. The application leverages CUDA acceleration through CuPy for real-time audio processing and PyTorch for GPU-accelerated speech recognition.

## Features

- **Multiple Visualization Types**: Select from bars, wave, and spectrum styles.
- **Frame Rate Control**: Adjustable FPS between 1-60.
- **Customizable Colors**: Set distinct visualization and background colors.
- **Aspect Ratio and Orientation Options**: Supports 16:9, 4:3, and 1:1 aspect ratios with horizontal or vertical orientations.
- **Subtitle Integration**: Automatically generates and integrates subtitles using Whisper for real-time transcriptions, synchronized with the video.
- **Memory-efficient Processing**: Utilizes memory mapping to handle large audio files.
- **Supported Formats**: MP3, WAV, M4A, OGG, and FLAC.
- **GPU Acceleration**: CUDA-powered audio processing for faster rendering
- **Parallel Processing**: Utilizes all CPU cores for enhanced performance
- **Real-time Speech Recognition**: GPU-accelerated Whisper model for subtitle generation
- **Smart Memory Management**: Memory mapping and batch processing for large files
- **Frame Interpolation**: Smooth transitions between frames using GPU acceleration
- **Adaptive Processing**: Falls back to CPU when GPU is unavailable

## Installation

### Prerequisites
- Python 3.7 or higher
- FFmpeg (for video encoding)
- Qt5 libraries (included in PyQt5)
- `libsndfile` (for audio processing)
- `CUDA` (optional, for faster Whisper transcription if using a GPU)

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Gontary101/audio-visualizer.git
    cd audio-visualizer
    ```

2. **Set up a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install system dependencies** (Linux users):
    ```bash
    # Ubuntu/Debian
    sudo apt-get install python3-qt5 libsndfile1 ffmpeg

    # Fedora
    sudo dnf install python3-qt5 libsndfile ffmpeg
    ```

### Additional Notes for Windows Users
Ensure that ImageMagick is installed and `MAGICK_BINARY` is correctly set in `moviepy/config_defaults.py` if needed. Install it from [ImageMagick’s website](https://imagemagick.org) and set the path in the code.

### 1. CUDA Setup (for GPU acceleration)
First, install CUDA Toolkit from NVIDIA's website. Then:

```bash
# Verify CUDA installation
nvidia-smi

# Install CuPy for your CUDA version
# For CUDA 11.x:
pip install cupy-cuda11x
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Additional System Dependencies
For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install python3-qt5 libsndfile1 ffmpeg nvidia-cuda-toolkit
```

For Windows:
- Install CUDA Toolkit from NVIDIA website
- Install ImageMagick and add to PATH
- Install Visual C++ Build Tools

## Usage

1. **Run the application**:
    ```bash
    python audio_visualizer.py
    ```

2. **Select an Audio File**: Click "Select Audio" to load your audio file.
3. **Configure Visualization Settings**:
    - Choose visualization type: bars, wave, or spectrum.
    - Adjust FPS (1-60).
    - Set aspect ratio (16:9, 4:3, or 1:1) and orientation (horizontal or vertical).
    - Customize visualization and background colors.
    - Adjust amplitude scale using the slider.
    - Select the subsampling factor to control signal density.
4. **Generate Video**: Click "Generate Video," choose a save location, and wait for processing. The application will automatically generate and add subtitles to the video, synchronized with the audio content.

## Troubleshooting

### Common Issues

- **Qt Platform Plugin Error on Linux**:
    ```bash
    # Install Wayland support:
    sudo apt-get install qt5-wayland  # Ubuntu/Debian
    sudo dnf install qt5-qtwayland    # Fedora

    # Or set the platform:
    export QT_QPA_PLATFORM=xcb
    ```

- **Audio File Errors**:
    - Check file format compatibility and ensure the file isn’t corrupted.
    - Confirm audio codecs are installed correctly.

- **Memory Issues**:
    - Reduce FPS or try shorter audio files.
    - Close other memory-intensive applications during processing.

### Performance Tips

- Use lower FPS for quicker processing.
- “Bars” visualization processes faster than “spectrum.”
- Running the application on SSD storage for temporary files can improve speed.

## Dependencies

Check `requirements.txt` for the full list of Python dependencies.

## System Requirements

- **Python**: 3.7 or higher.
- **FFmpeg**: Required for video encoding.
- **RAM**: 4GB minimum (8GB recommended for longer files).
- **Whisper**: Optional, requires a compatible GPU for faster transcription.

## Contributing

1. Fork this repository.
2. Create a new branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.



## Acknowledgments

- NVIDIA for CUDA toolkit
- CuPy team for GPU acceleration
- OpenAI for the Whisper model
- MoviePy contributors

## Known Issues and Solutions

### GPU-Related Issues

1. **CUDA Out of Memory**
   - Reduce batch size in video generation (default is 50, try 25 or lower)
   - Lower the frame rate
   - Use a smaller window length for audio processing
   - Close other GPU-intensive applications

2. **CuPy Import Errors**
   ```python
   # Fallback solution in code
   try:
       import cupy as cp
   except ImportError:
       import numpy as cp
       print("GPU acceleration not available, using CPU")
   ```

3. **Whisper Model Loading Issues**
   - Ensure enough VRAM for model loading (at least 4GB recommended)
   - If memory error occurs, try:
     ```python
     import torch
     torch.cuda.empty_cache()  # Clear GPU memory
     ```
   - Consider using smaller Whisper model variants (tiny, base, or small)

4. **GPU Memory Leaks**
   - Clear matplotlib figures after each frame generation
   - Use context managers for GPU operations
   - Monitor GPU memory usage with `nvidia-smi`

### Audio Processing Issues

1. **Long Audio Files**
   - Files longer than 10 minutes may cause memory issues
   - Solution: Split audio into chunks:
     ```python
     # Using pydub
     from pydub import AudioSegment
     audio = AudioSegment.from_file(file_path)
     chunk_length = 10 * 60 * 1000  # 10 minutes in milliseconds
     chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
     ```

2. **High Sample Rate Audio**
   - High sample rates (>48kHz) may cause processing delays
   - Solution: Downsample before processing:
     ```python
     y, sr = librosa.load(audio_path, sr=44100)  # Force 44.1kHz
     ```

3. **Spectrogram Memory Usage**
   - Large spectrograms can consume excessive memory
   - Solutions:
     - Increase subsample_factor (default is 4)
     - Reduce frame_length (default is 2048)
     - Use lower frequency resolution

### GUI Issues

1. **Progress Bar Freezing**
   - Progress bar may appear frozen during heavy processing
   - Solution: Reduce update frequency or use smaller batch sizes
   - Alternative: Implement background worker for progress updates

2. **Color Picker Dialog**
   - May crash on some Linux distributions
   - Workaround: Use hex color codes directly:
     ```python
     self.viz_color = "#000000"  # Black
     self.bg_color = "#FFFFFF"   # White
     ```

3. **Window Scaling Issues**
   - High DPI displays may show incorrect scaling
   - Solution: Add to start of script:
     ```python
     if hasattr(Qt, 'AA_EnableHighDpiScaling'):
         QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
     if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
         QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
     ```

### File Handling Issues

1. **Temporary File Cleanup**
   - Temporary files might not be deleted if process is interrupted
   - Solution: Implement cleanup on exit:
     ```python
     import atexit
     import shutil

     def cleanup_temp_files():
         if os.path.exists(tmp_dir):
             shutil.rmtree(tmp_dir)
     
     atexit.register(cleanup_temp_files)
     ```

2. **FFmpeg Codec Issues**
   - Some systems may not support h264 codec
   - Solution: Provide fallback codec options:
     ```python
     try:
         # Try h264 first
         final_video.write_videofile(output_path, codec='libx264')
     except:
         # Fallback to mpeg4
         final_video.write_videofile(output_path, codec='mpeg4')
     ```

3. **Large Output Files**
   - High FPS and resolution can create very large files
   - Solutions:
     - Add video compression options
     - Implement bitrate control:
       ```python
       final_video.write_videofile(
           output_path,
           bitrate="2000k",
           audio_bitrate="192k"
       )
       ```

### Performance Optimization Tips

1. **Batch Processing**
   - Adjust batch size based on available memory:
     ```python
     batch_size = min(50, total_frames // 100)  # Dynamic batch size
     ```

2. **Memory Management**
   - Implement periodic garbage collection:
     ```python
     import gc
     gc.collect()
     torch.cuda.empty_cache()  # If using GPU
     ```

3. **Multi-threading**
   - Be cautious with thread pool size:
     ```python
     max_workers = min(mp.cpu_count(), 8)  # Limit maximum threads
     ```

4. **Frame Generation**
   - Cache frequently used matplotlib objects
   - Use vectorized operations where possible
   - Consider using OpenGL for real-time visualization
