
# Advanced Audio Visualizer

A Python-based desktop application for creating dynamic audio visualizations with custom styles and subtitles, designed for an interactive user experience. The application leverages `PyQt5` for the GUI and `MoviePy` for video generation, supporting multiple visualization types, real-time progress tracking, custom colors, and aspect ratios.

## Features

- **Multiple Visualization Types**: Select from bars, wave, and spectrum styles.
- **Frame Rate Control**: Adjustable FPS between 1-60.
- **Customizable Colors**: Set distinct visualization and background colors.
- **Aspect Ratio and Orientation Options**: Supports 16:9, 4:3, and 1:1 aspect ratios with horizontal or vertical orientations.
- **Subtitle Integration**: Automatically generates and integrates subtitles using Whisper for real-time transcriptions, synchronized with the video.
- **Memory-efficient Processing**: Utilizes memory mapping to handle large audio files.
- **Supported Formats**: MP3, WAV, M4A, OGG, and FLAC.

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