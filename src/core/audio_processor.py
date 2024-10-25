from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Dict, List, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioFormat(Enum):
    """Supported audio file formats"""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    
class AnalysisType(Enum):
    """Types of audio analysis available"""
    WAVEFORM = "waveform"
    SPECTRUM = "spectrum"
    BEATS = "beats"
    TEMPO = "tempo"
    PITCH = "pitch"
    RMS = "rms"
    MFCC = "mfcc"

@dataclass
class AudioSegment:
    """Represents a segment of audio data with its properties"""
    data: np.ndarray
    start_time: float
    end_time: float
    sample_rate: int
    
    def duration(self) -> float:
        """Calculate segment duration in seconds"""
        return self.end_time - self.start_time

class AudioEffect:
    """Base class for audio effects"""
    def __init__(self, name: str):
        self.name = name
        
    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio data with the effect"""
        raise NotImplementedError("Effect processing not implemented")
        
    def get_parameters(self) -> Dict:
        """Get current effect parameters"""
        raise NotImplementedError("Parameter getter not implemented")
        
    def set_parameters(self, params: Dict) -> None:
        """Set effect parameters"""
        raise NotImplementedError("Parameter setter not implemented")

class AudioProcessor:
    """Main audio processing class with comprehensive audio manipulation capabilities"""
    
    def __init__(self, 
                 default_sample_rate: int = 44100,
                 buffer_size: int = 2048,
                 num_channels: int = 2,
                 enable_realtime: bool = False):
        """
        Initialize AudioProcessor with specified parameters
        
        Args:
            default_sample_rate: Target sample rate for processing
            buffer_size: Size of audio processing buffer
            num_channels: Number of audio channels to process
            enable_realtime: Enable real-time processing capabilities
        """
        self.default_sample_rate = default_sample_rate
        self.buffer_size = buffer_size
        self.num_channels = num_channels
        self.enable_realtime = enable_realtime
        
        # Audio data storage
        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate: Optional[int] = None
        self._duration: Optional[float] = None
        
        # Real-time processing
        self._processing_queue = queue.Queue() if enable_realtime else None
        self._processing_thread: Optional[threading.Thread] = None
        self._is_processing = False
        
        # Effect chain
        self._effects: List[AudioEffect] = []
        
        # Analysis cache
        self._analysis_cache: Dict = {}
        
        # Thread pool for parallel processing
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
    def load_file(self, 
                  file_path: Union[str, Path], 
                  normalize: bool = True,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None) -> bool:
        """
        Load audio file with optional normalization and time range
        
        Args:
            file_path: Path to audio file
            normalize: Whether to normalize audio data
            start_time: Start time in seconds for partial loading
            end_time: End time in seconds for partial loading
            
        Returns:
            bool: Success status
        """
        try:
            # Load audio file
            y, sr = librosa.load(file_path, 
                               sr=self.default_sample_rate,
                               mono=self.num_channels == 1,
                               offset=start_time if start_time else 0,
                               duration=end_time - start_time if end_time else None)
            
            self._audio_data = y
            self._sample_rate = sr
            self._duration = librosa.get_duration(y=y, sr=sr)
            
            # Normalize if requested
            if normalize:
                self._audio_data = librosa.util.normalize(self._audio_data)
            
            # Clear analysis cache
            self._analysis_cache.clear()
            
            logger.info(f"Successfully loaded audio file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return False
            
    def save_file(self, 
                  file_path: Union[str, Path], 
                  format: AudioFormat = AudioFormat.WAV) -> bool:
        """
        Save processed audio to file
        
        Args:
            file_path: Output file path
            format: Output audio format
            
        Returns:
            bool: Success status
        """
        try:
            if self._audio_data is None:
                raise ValueError("No audio data loaded")
                
            sf.write(file_path, self._audio_data, self._sample_rate)
            logger.info(f"Successfully saved audio to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return False
    
    def analyze(self, 
                analysis_type: AnalysisType,
                start_time: Optional[float] = None,
                end_time: Optional[float] = None,
                **kwargs) -> Dict:
        """
        Perform audio analysis of specified type
        
        Args:
            analysis_type: Type of analysis to perform
            start_time: Start time for analysis
            end_time: End time for analysis
            **kwargs: Additional analysis parameters
            
        Returns:
            Dict containing analysis results
        """
        if self._audio_data is None:
            raise ValueError("No audio data loaded")
            
        # Check cache first
        cache_key = (analysis_type, start_time, end_time, str(kwargs))
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
            
        # Get audio segment for analysis
        segment = self._get_segment(start_time, end_time)
        
        # Perform requested analysis
        result = {}
        
        if analysis_type == AnalysisType.WAVEFORM:
            result['waveform'] = segment.data
            
        elif analysis_type == AnalysisType.SPECTRUM:
            D = librosa.stft(segment.data, **kwargs)
            result['spectrum'] = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
        elif analysis_type == AnalysisType.BEATS:
            tempo, beats = librosa.beat.beat_track(y=segment.data, sr=self._sample_rate)
            result['tempo'] = tempo
            result['beats'] = beats
            
        elif analysis_type == AnalysisType.TEMPO:
            onset_env = librosa.onset.onset_strength(y=segment.data, sr=self._sample_rate)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self._sample_rate)
            result['tempo'] = tempo[0]
            
        elif analysis_type == AnalysisType.PITCH:
            pitches, magnitudes = librosa.piptrack(y=segment.data, sr=self._sample_rate)
            result['pitches'] = pitches
            result['magnitudes'] = magnitudes
            
        elif analysis_type == AnalysisType.RMS:
            result['rms'] = librosa.feature.rms(y=segment.data)[0]
            
        elif analysis_type == AnalysisType.MFCC:
            mfccs = librosa.feature.mfcc(y=segment.data, sr=self._sample_rate, **kwargs)
            result['mfcc'] = mfccs
            
        # Cache results
        self._analysis_cache[cache_key] = result
        return result
    
    def add_effect(self, effect: AudioEffect) -> None:
        """Add an audio effect to the processing chain"""
        self._effects.append(effect)
        
    def remove_effect(self, effect_name: str) -> None:
        """Remove an effect from the processing chain by name"""
        self._effects = [e for e in self._effects if e.name != effect_name]
        
    def apply_effects(self) -> None:
        """Apply all effects in the chain to the audio data"""
        if self._audio_data is None:
            raise ValueError("No audio data loaded")
            
        processed_data = self._audio_data.copy()
        for effect in self._effects:
            processed_data = effect.process(processed_data, self._sample_rate)
        
        self._audio_data = processed_data
        self._analysis_cache.clear()
        
    def start_realtime_processing(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Start real-time audio processing
        
        Args:
            callback: Function to call with processed audio chunks
        """
        if not self.enable_realtime:
            raise ValueError("Real-time processing not enabled")
            
        def processing_loop():
            while self._is_processing:
                try:
                    chunk = self._processing_queue.get(timeout=1.0)
                    processed_chunk = chunk.copy()
                    
                    # Apply effects
                    for effect in self._effects:
                        processed_chunk = effect.process(processed_chunk, self._sample_rate)
                        
                    callback(processed_chunk)
                    
                except queue.Empty:
                    continue
                    
        self._is_processing = True
        self._processing_thread = threading.Thread(target=processing_loop)
        self._processing_thread.start()
        
    def stop_realtime_processing(self) -> None:
        """Stop real-time audio processing"""
        self._is_processing = False
        if self._processing_thread:
            self._processing_thread.join()
            
    def _get_segment(self, 
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None) -> AudioSegment:
        """Get audio segment between specified times"""
        if self._audio_data is None:
            raise ValueError("No audio data loaded")
            
        start_sample = int(start_time * self._sample_rate) if start_time else 0
        end_sample = int(end_time * self._sample_rate) if end_time else len(self._audio_data)
        
        return AudioSegment(
            data=self._audio_data[start_sample:end_sample],
            start_time=start_time if start_time else 0,
            end_time=end_time if end_time else self._duration,
            sample_rate=self._sample_rate
        )
    
    @property
    def duration(self) -> Optional[float]:
        """Get audio duration in seconds"""
        return self._duration
        
    @property
    def sample_rate(self) -> Optional[int]:
        """Get audio sample rate"""
        return self._sample_rate
        
    def __del__(self):
        """Cleanup resources"""
        if self._thread_pool:
            self._thread_pool.shutdown()
        self.stop_realtime_processing()

# Example effect implementations
class NormalizationEffect(AudioEffect):
    """Normalizes audio to a target RMS level"""
    
    def __init__(self, target_rms: float = -20.0):
        super().__init__("normalize")
        self.target_rms = target_rms
        
    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        return librosa.util.normalize(audio_data, rms=self.target_rms)
        
    def get_parameters(self) -> Dict:
        return {"target_rms": self.target_rms}
        
    def set_parameters(self, params: Dict) -> None:
        if "target_rms" in params:
            self.target_rms = params["target_rms"]

class ReverbEffect(AudioEffect):
    """Applies reverb to audio"""
    
    def __init__(self, room_size: float = 0.5, damping: float = 0.5):
        super().__init__("reverb")
        self.room_size = room_size
        self.damping = damping
        
    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        # Simple convolution reverb implementation
        decay = np.exp(-np.linspace(0, self.room_size * 5, int(sample_rate * self.room_size)))
        decay = decay * (1 - self.damping * np.linspace(0, 1, len(decay)))
        
        return np.convolve(audio_data, decay, mode='same')
        
    def get_parameters(self) -> Dict:
        return {
            "room_size": self.room_size,
            "damping": self.damping
        }
        
    def set_parameters(self, params: Dict) -> None:
        if "room_size" in params:
            self.room_size = params["room_size"]
        if "damping" in params:
            self.damping = params["damping"]
