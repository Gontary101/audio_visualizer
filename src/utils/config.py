# config.py

from PyQt5.QtCore import QSettings

class AppConfig:
    """Application configuration management with persistent settings"""
    
    def __init__(self):
        self.settings = QSettings('AudioVisualizer', 'App')
    
    def set_value(self, key: str, value):
        """Set a configuration value"""
        self.settings.setValue(key, value)
    
    def get_value(self, key: str, default=None):
        """Retrieve a configuration value with an optional default"""
        return self.settings.value(key, default)
    
    def set_window_geometry(self, geometry):
        """Save window geometry settings"""
        self.set_value('window_geometry', geometry)
    
    def get_window_geometry(self):
        """Retrieve window geometry settings"""
        return self.get_value('window_geometry')
    
    def set_window_state(self, state):
        """Save window state settings"""
        self.set_value('window_state', state)
    
    def get_window_state(self):
        """Retrieve window state settings"""
        return self.get_value('window_state')
    
    def set_recent_project(self, project_path: str):
        """Save the path of the last opened project"""
        self.set_value('recent_project', project_path)
    
    def get_recent_project(self):
        """Retrieve the path of the last opened project"""
        return self.get_value('recent_project')
    
    def clear(self):
        """Clear all settings"""
        self.settings.clear()
