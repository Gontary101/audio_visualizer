# Audio Visualizer

This project creates a video visualization of an audio file, displaying the amplitude as a bar graph that changes over time. It uses multiprocessing to speed up video generation.

It's useful for generating a video output for an AI podcast, for example (NOTEBOOKLM).

## Features

- **Input:** Accepts MP3 and WAV audio files
- **Output:** Generates MP4 video files
- **Visualization:** Displays audio amplitude as a dynamic bar graph
- **Amplitude Control:** Adjustable amplitude scaling using a slider
- **Progress Bar:** Shows the progress of video generation
- **Multiprocessing:** Utilizes multiple CPU cores for faster processing
- **GUI:** User-friendly interface built with PyQt5

## Requirements

See `requirements.txt` for a list of required Python packages.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Gontary101/audio-visualizer.git
   ```

2. Navigate to the project directory:
   ```bash
   cd audio-visualizer
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python audio_visualizer.py 
   ```

2. Click "Select Audio File" and choose your desired audio file (MP3 or WAV).

3. Adjust the "Amplitude Scale" slider to control the height of the bars in the visualization.

4. Click "Generate Video" and choose a location to save the output MP4 file.

5. The progress bar will display the video generation progress. Once complete, the video will be saved at the specified location.

## Dependencies

This project relies on several external libraries:

- **librosa:** For audio analysis and loading
- **NumPy:** For numerical operations
- **Matplotlib:** For plotting the bar graph
- **PyQt5:** For the graphical user interface
- **moviepy:** For video creation and manipulation
- **multiprocessing:** For parallel processing
