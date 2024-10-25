# helpers.py

from PyQt5.QtGui import QIcon
from pathlib import Path

def format_time(seconds: int) -> str:
    """Format seconds into a string of the form HH:MM:SS"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    return f"{minutes:02}:{seconds:02}"

def create_icon_path(icon_name: str) -> QIcon:
    """Generate the full path to an icon resource by name"""
    # Adjust to the actual path to icons within the project structure
    icon_path = Path(__file__).parent / "resources/icons" / f"{icon_name}.png"
    return QIcon(str(icon_path))
