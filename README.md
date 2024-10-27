# Audio Visualizer

A Python-based desktop application that creates beautiful visualizations from audio files. The application supports multiple visualization types, custom colors, different aspect ratios, and orientations.

## Features

- Multiple visualization types (bars, wave, spectrum)
- Adjustable FPS (1-60)
- Customizable visualization and background colors
- Support for different aspect ratios (16:9, 4:3, 1:1)
- Horizontal and vertical orientations
- Real-time progress tracking
- Support for various audio formats (MP3, WAV, M4A, OGG, FLAC)
- Memory-efficient processing using memory mapping
- Modern, user-friendly GUI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-visualizer.git
cd audio-visualizer
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. On Linux systems, ensure you have the required system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install python3-qt5 libsndfile1 ffmpeg

# Fedora
sudo dnf install python3-qt5 libsndfile ffmpeg
```

## Usage

1. Run the application:
```bash
python audio_visualizer.py
```

2. Click "Select Audio" to choose an audio file
3. Configure your visualization settings:
   - Choose visualization type (bars, wave, spectrum)
   - Adjust FPS (1-60)
   - Select aspect ratio (16:9, 4:3, 1:1)
   - Choose orientation (horizontal, vertical)
   - Set visualization and background colors
   - Adjust amplitude scale using the slider
4. Click "Generate Video" and choose where to save the output file
5. Wait for the processing to complete

## Troubleshooting

### Common Issues

1. **Qt platform plugin error on Linux**:
   ```bash
   # Either install Qt Wayland plugin:
   sudo apt-get install qt5-wayland  # Ubuntu/Debian
   sudo dnf install qt5-qtwayland    # Fedora
   
   # Or set the environment variable:
   export QT_QPA_PLATFORM=xcb
   ```

2. **Audio file loading errors**:
   - Ensure the audio file format is supported
   - Check if the file is corrupted
   - Verify that you have the required audio codecs installed

3. **Memory errors during processing**:
   - Try reducing the FPS
   - Process shorter audio files
   - Close other memory-intensive applications

### Performance Tips

- Lower FPS values will result in faster processing
- The 'bars' visualization type typically processes faster than 'spectrum'
- Closing other applications during video generation can improve performance
- Using SSD storage for temporary files can speed up processing

## Dependencies

See `requirements.txt` for the complete list of Python dependencies.

## System Requirements

- Python 3.7 or higher
- FFmpeg (for video encoding)
- At least 4GB RAM (8GB recommended for longer audio files)
- Qt5 libraries
- libsndfile

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## what's next
1. Adding live subtitles using Whisper

