"""
Voice Pipeline Service - STT (Whisper) and TTS (Gemini Native Audio)
"""
import asyncio
import io
import logging
import time
from typing import Optional, Tuple, Any
import numpy as np
import sounddevice as sd
import whisper
import wave
import tempfile
from pathlib import Path

import google.generativeai as genai
from google.generativeai.types import GenerativeModel

from backend.config import settings

logger = logging.getLogger(__name__)


class VoiceService:
    """Handles Speech-to-Text and Text-to-Speech operations"""
    
    def __init__(self):
        # Initialize Whisper for STT
        logger.info(f"Loading Whisper {settings.whisper_model} model...")
        self.whisper_model = whisper.load_model(
            settings.whisper_model,
            device=settings.whisper_device
        )
        logger.info("Whisper model loaded")
        
        # Configure Gemini for TTS
        genai.configure(api_key=settings.gemini_api_key)
        
        # Initialize Gemini Native Audio model for TTS
        self.tts_model = genai.GenerativeModel(
            model_name=settings.tts_model,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                audio_config={
                    "voice": settings.tts_voice,
                    "speed": settings.tts_speed
                }
            )
        )
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        
        logger.info("Voice service initialized")
    
    async def transcribe_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> Tuple[str, float]:
        """
        Transcribe audio to text using Whisper
        
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        start_time = time.time()
        
        try:
            # Ensure audio is in the correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Run transcription
            result = self.whisper_model.transcribe(
                audio_data,
                language="en",
                fp16=False
            )
            
            transcription = result["text"].strip()
            
            # Calculate simple confidence based on no_speech_prob
            confidence = 1.0 - result.get("no_speech_prob", 0.0)
            
            processing_time = time.time() - start_time
            logger.info(f"Transcribed in {processing_time:.2f}s: {transcription}")
            
            return transcription, confidence
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return "", 0.0
    
    async def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None
    ) -> Optional[bytes]:
        """
        Convert text to speech using Gemini Native Audio
        
        Returns:
            Audio data as bytes (WAV format)
        """
        try:
            # Use provided voice/speed or defaults from settings
            voice = voice or settings.tts_voice
            speed = speed or settings.tts_speed
            
            # Create audio generation prompt
            prompt = f"""<audio_generation>
<voice>{voice}</voice>
<speed>{speed}</speed>
<text>{text}</text>
</audio_generation>"""
            
            # Generate audio using Gemini Native Audio
            response = self.tts_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="audio/wav",
                    audio_config={
                        "voice": voice,
                        "speed": speed
                    }
                )
            )
            
            # Extract audio data from response
            if response.audio:
                return response.audio
            else:
                logger.error("No audio data in TTS response")
                return None
                
        except Exception as e:
            logger.error(f"Error in TTS synthesis: {e}")
            return None
    
    async def play_audio(self, audio_data: bytes):
        """Play audio data through speakers"""
        try:
            # Convert bytes to numpy array
            with io.BytesIO(audio_data) as audio_buffer:
                with wave.open(audio_buffer, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                    sample_rate = wav_file.getframerate()
            
            # Play audio
            sd.play(audio_array, sample_rate)
            sd.wait()  # Wait until audio finishes playing
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    async def record_audio(
        self,
        duration: float = 5.0,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """Record audio from microphone"""
        try:
            logger.info(f"Recording {duration} seconds of audio...")
            
            # Record audio
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()  # Wait until recording is finished
            
            return recording.flatten()
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return np.array([])
    
    async def voice_activity_detection(
        self,
        audio_data: np.ndarray,
        threshold: float = 0.01
    ) -> bool:
        """Simple voice activity detection"""
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms > threshold
    
    async def continuous_recording_with_vad(
        self,
        max_duration: float = 10.0,
        silence_duration: float = 1.5,
        energy_threshold: float = 0.01
    ) -> np.ndarray:
        """
        Record audio with voice activity detection
        Stops recording after silence_duration seconds of silence
        """
        sample_rate = self.sample_rate
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(chunk_duration * sample_rate)
        
        recording = []
        silence_chunks = 0
        max_silence_chunks = int(silence_duration / chunk_duration)
        max_chunks = int(max_duration / chunk_duration)
        
        logger.info("Recording with VAD...")
        
        try:
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=chunk_samples
            )
            
            with stream:
                for i in range(max_chunks):
                    chunk, _ = stream.read(chunk_samples)
                    recording.append(chunk)
                    
                    # Check for voice activity
                    if await self.voice_activity_detection(chunk, energy_threshold):
                        silence_chunks = 0
                    else:
                        silence_chunks += 1
                    
                    # Stop if enough silence detected
                    if silence_chunks >= max_silence_chunks and len(recording) > 10:
                        logger.info("Silence detected, stopping recording")
                        break
            
            # Combine all chunks
            full_recording = np.concatenate(recording, axis=0).flatten()
            return full_recording
            
        except Exception as e:
            logger.error(f"Error in continuous recording: {e}")
            return np.array([])
    
    async def process_voice_command(
        self,
        audio_data: np.ndarray
    ) -> Tuple[str, float]:
        """Process a voice command and return transcription"""
        # Check if there's actual audio content
        if len(audio_data) == 0 or np.max(np.abs(audio_data)) < 0.001:
            return "", 0.0
        
        # Transcribe the audio
        text, confidence = await self.transcribe_audio(audio_data)
        
        return text, confidence
    
    async def speak_response(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        play_audio: bool = True
    ) -> Optional[bytes]:
        """
        Generate and optionally play speech from text
        
        Returns:
            Audio data as bytes
        """
        # Generate speech
        audio_data = await self.synthesize_speech(text, voice, speed)
        
        if audio_data and play_audio:
            # Play the audio
            await self.play_audio(audio_data)
        
        return audio_data
    
    def get_available_voices(self) -> list:
        """Get list of available TTS voices"""
        # Gemini Native Audio voices
        return [
            "nova",      # Default female voice
            "echo",      # Male voice
            "fable",     # Storytelling voice
            "onyx",      # Deep male voice
            "shimmer"    # Expressive female voice
        ]
    
    async def test_audio_setup(self) -> bool:
        """Test audio input/output setup"""
        try:
            # Test recording
            logger.info("Testing audio recording...")
            test_recording = await self.record_audio(duration=1.0)
            
            if len(test_recording) == 0:
                logger.error("No audio recorded - check microphone")
                return False
            
            # Test playback with a simple beep
            logger.info("Testing audio playback...")
            t = np.linspace(0, 0.2, int(0.2 * self.sample_rate))
            beep = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz beep
            
            sd.play(beep, self.sample_rate)
            sd.wait()
            
            logger.info("Audio setup test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Audio setup test failed: {e}")
            return False 