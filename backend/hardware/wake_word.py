"""
Wake Word Detection for "Hey Wheatley"
Uses Porcupine for wake word detection (OpenWakeWord can be added later)
"""
import asyncio
import logging
import numpy as np
import sounddevice as sd
from typing import Callable, Optional
import struct
import pvporcupine

from backend.config import settings

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Detects wake word 'Hey Wheatley'"""
    
    def __init__(self, on_wake_callback: Optional[Callable] = None):
        self.on_wake_callback = on_wake_callback
        self.is_listening = False
        self.porcupine = None
        self.stream = None
        
        # Initialize Porcupine
        self._init_porcupine()
        
        logger.info("Wake word detector initialized")
    
    def _init_porcupine(self):
        """Initialize Porcupine wake word engine"""
        try:
            # For demo, using built-in wake words
            # In production, you would train a custom "Hey Wheatley" model
            self.porcupine = pvporcupine.create(
                keywords=["computer", "jarvis"],  # Using built-in keywords for demo
                sensitivities=[0.5, 0.5]
            )
            
            self.sample_rate = self.porcupine.sample_rate
            self.frame_length = self.porcupine.frame_length
            
            logger.info(f"Porcupine initialized: {self.sample_rate}Hz, {self.frame_length} samples/frame")
            
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            logger.info("Wake word detection will be disabled")
            self.porcupine = None
    
    async def start_listening(self):
        """Start listening for wake word"""
        if not self.porcupine:
            logger.error("Porcupine not initialized, cannot start listening")
            return
        
        if self.is_listening:
            logger.warning("Already listening for wake word")
            return
        
        self.is_listening = True
        logger.info("Started listening for wake word...")
        
        # Start audio stream
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.frame_length,
                dtype='int16',
                callback=self._audio_callback
            )
            self.stream.start()
            
            # Keep the detector running
            while self.is_listening:
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for processing audio frames"""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        if not self.is_listening or not self.porcupine:
            return
        
        # Convert audio data to the format expected by Porcupine
        audio_frame = indata.flatten()
        
        try:
            # Process the audio frame
            keyword_index = self.porcupine.process(audio_frame)
            
            if keyword_index >= 0:
                logger.info(f"Wake word detected! (index: {keyword_index})")
                
                # Call the wake callback
                if self.on_wake_callback:
                    asyncio.create_task(self._handle_wake_word())
                    
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
    
    async def _handle_wake_word(self):
        """Handle wake word detection"""
        # Temporarily stop listening to prevent multiple detections
        self.is_listening = False
        
        # Play acknowledgment sound
        await self._play_ding()
        
        # Call the callback
        if self.on_wake_callback:
            await self.on_wake_callback()
        
        # Resume listening after a short delay
        await asyncio.sleep(2.0)
        self.is_listening = True
    
    async def _play_ding(self):
        """Play acknowledgment sound"""
        try:
            # Generate a simple ding sound
            duration = 0.2
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create a pleasant ding sound (two tones)
            frequency1 = 523.25  # C5
            frequency2 = 659.25  # E5
            
            ding = 0.3 * np.sin(2 * np.pi * frequency1 * t) * np.exp(-t * 5)
            ding += 0.2 * np.sin(2 * np.pi * frequency2 * t) * np.exp(-t * 3)
            
            sd.play(ding, sample_rate)
            sd.wait()
            
        except Exception as e:
            logger.error(f"Error playing ding sound: {e}")
    
    def stop_listening(self):
        """Stop listening for wake word"""
        self.is_listening = False
        logger.info("Stopped listening for wake word")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_listening()
        
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None


class SimpleWakeWordDetector:
    """
    Simplified wake word detector using energy-based detection
    (Fallback when Porcupine is not available)
    """
    
    def __init__(self, on_wake_callback: Optional[Callable] = None):
        self.on_wake_callback = on_wake_callback
        self.is_listening = False
        self.sample_rate = 16000
        self.energy_threshold = 0.02
        
        logger.info("Simple wake word detector initialized (energy-based)")
    
    async def start_listening(self):
        """Start listening for loud sounds (simplified wake detection)"""
        self.is_listening = True
        logger.info("Started listening for audio activity...")
        
        chunk_duration = 0.5  # 500ms chunks
        chunk_samples = int(chunk_duration * self.sample_rate)
        
        try:
            while self.is_listening:
                # Record a chunk
                recording = sd.rec(
                    chunk_samples,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=np.float32
                )
                sd.wait()
                
                # Check energy level
                energy = np.sqrt(np.mean(recording ** 2))
                
                if energy > self.energy_threshold:
                    logger.info("Audio activity detected!")
                    
                    if self.on_wake_callback:
                        await self.on_wake_callback()
                    
                    # Wait before listening again
                    await asyncio.sleep(3.0)
                else:
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error in simple wake detection: {e}")
    
    def stop_listening(self):
        """Stop listening"""
        self.is_listening = False
        logger.info("Stopped listening for audio activity")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_listening()


# Factory function to create appropriate wake word detector
def create_wake_word_detector(on_wake_callback: Optional[Callable] = None) -> WakeWordDetector:
    """Create wake word detector, falling back to simple detection if needed"""
    try:
        # Try to create Porcupine-based detector
        detector = WakeWordDetector(on_wake_callback)
        if detector.porcupine:
            return detector
        else:
            # Fall back to simple detector
            detector.cleanup()
            return SimpleWakeWordDetector(on_wake_callback)
    except Exception as e:
        logger.error(f"Failed to create wake word detector: {e}")
        # Return simple detector as fallback
        return SimpleWakeWordDetector(on_wake_callback) 