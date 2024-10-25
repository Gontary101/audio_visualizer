# setup.py

from setuptools import setup, find_packages

setup(
    name="audio_visualizer",
    version="1.0.0",
    description="A real-time audio visualization and editing application.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "PyQt5",
        "numpy",
        "librosa",
        "soundfile",
        "matplotlib",
        "opencv-python",
    ],
    entry_points={
        "console_scripts": [
            "audio-visualizer=main:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
