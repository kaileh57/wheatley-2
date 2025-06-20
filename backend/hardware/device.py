"""
Hardware Device Interface for Raspberry Pi
Handles LEDs, buttons, and other hardware interactions
"""
import asyncio
import logging
from typing import Dict, Any, Callable, Optional
from datetime import datetime
import json

from backend.config import settings

logger = logging.getLogger(__name__)

# Try to import RPi.GPIO, fallback to mock if not on Raspberry Pi
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    logger.warning("RPi.GPIO not available, using mock GPIO")
    HAS_GPIO = False
    
    # Mock GPIO for development/testing
    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        IN = "IN"
        PUD_UP = "PUD_UP"
        FALLING = "FALLING"
        
        @staticmethod
        def setmode(mode):
            pass
        
        @staticmethod
        def setup(pin, mode, pull_up_down=None):
            logger.debug(f"Mock GPIO setup: pin {pin}, mode {mode}")
        
        @staticmethod
        def output(pin, value):
            logger.debug(f"Mock GPIO output: pin {pin}, value {value}")
        
        @staticmethod
        def input(pin):
            return 0
        
        @staticmethod
        def add_event_detect(pin, edge, callback, bouncetime=None):
            logger.debug(f"Mock GPIO event: pin {pin}, edge {edge}")
        
        @staticmethod
        def cleanup():
            logger.debug("Mock GPIO cleanup")
    
    GPIO = MockGPIO()


class LEDController:
    """Controls LED patterns for visual feedback"""
    
    def __init__(self):
        self.led_pins = {
            "status": 17,     # Status LED
            "listening": 27,  # Listening indicator
            "thinking": 22,   # Processing indicator
            "speaking": 23    # Speaking indicator
        }
        
        self.current_pattern = None
        self.pattern_task = None
        
        if settings.led_enabled:
            self._setup_leds()
    
    def _setup_leds(self):
        """Initialize LED pins"""
        GPIO.setmode(GPIO.BCM)
        
        for name, pin in self.led_pins.items():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, False)
            logger.info(f"Initialized LED '{name}' on pin {pin}")
    
    async def set_pattern(self, pattern_name: str):
        """Set LED pattern"""
        if not settings.led_enabled:
            return
        
        # Cancel current pattern
        if self.pattern_task:
            self.pattern_task.cancel()
        
        self.current_pattern = pattern_name
        
        # Start new pattern
        if pattern_name == "listening":
            self.pattern_task = asyncio.create_task(self._breathing_pattern("listening"))
        elif pattern_name == "thinking":
            self.pattern_task = asyncio.create_task(self._spinning_pattern())
        elif pattern_name == "speaking":
            self.pattern_task = asyncio.create_task(self._pulsing_pattern("speaking"))
        elif pattern_name == "idle":
            self.pattern_task = asyncio.create_task(self._idle_pattern())
        elif pattern_name == "off":
            self._all_leds_off()
    
    async def _breathing_pattern(self, led_name: str):
        """Breathing LED pattern"""
        pin = self.led_pins.get(led_name)
        if not pin:
            return
        
        try:
            while True:
                # Fade in
                for i in range(10):
                    GPIO.output(pin, True)
                    await asyncio.sleep(0.01 * i)
                    GPIO.output(pin, False)
                    await asyncio.sleep(0.01 * (10 - i))
                
                # Fade out
                for i in range(10):
                    GPIO.output(pin, True)
                    await asyncio.sleep(0.01 * (10 - i))
                    GPIO.output(pin, False)
                    await asyncio.sleep(0.01 * i)
                
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            GPIO.output(pin, False)
    
    async def _spinning_pattern(self):
        """Spinning pattern across multiple LEDs"""
        pins = list(self.led_pins.values())
        
        try:
            while True:
                for pin in pins:
                    GPIO.output(pin, True)
                    await asyncio.sleep(0.1)
                    GPIO.output(pin, False)
        except asyncio.CancelledError:
            self._all_leds_off()
    
    async def _pulsing_pattern(self, led_name: str):
        """Quick pulsing pattern"""
        pin = self.led_pins.get(led_name)
        if not pin:
            return
        
        try:
            while True:
                GPIO.output(pin, True)
                await asyncio.sleep(0.2)
                GPIO.output(pin, False)
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            GPIO.output(pin, False)
    
    async def _idle_pattern(self):
        """Slow pulse on status LED"""
        pin = self.led_pins.get("status")
        if not pin:
            return
        
        try:
            while True:
                GPIO.output(pin, True)
                await asyncio.sleep(2.0)
                GPIO.output(pin, False)
                await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            GPIO.output(pin, False)
    
    def _all_leds_off(self):
        """Turn off all LEDs"""
        for pin in self.led_pins.values():
            GPIO.output(pin, False)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self.pattern_task:
            self.pattern_task.cancel()
        
        self._all_leds_off()
        
        if HAS_GPIO:
            GPIO.cleanup()


class ButtonController:
    """Handles physical button inputs"""
    
    def __init__(self):
        self.button_pins = {
            "pause_resume": settings.button_pins[0],  # Pause/Resume
            "privacy_mode": settings.button_pins[1],  # Privacy mode toggle
            "quick_note": settings.button_pins[2],     # Quick note
            "timer_stop": settings.button_pins[3]      # Stop timer
        }
        
        self.callbacks: Dict[str, Callable] = {}
        self.privacy_mode = False
        
        if settings.led_enabled:  # Buttons only work with hardware
            self._setup_buttons()
    
    def _setup_buttons(self):
        """Initialize button pins"""
        GPIO.setmode(GPIO.BCM)
        
        for name, pin in self.button_pins.items():
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(
                pin,
                GPIO.FALLING,
                callback=lambda ch, n=name: self._button_pressed(n),
                bouncetime=300
            )
            logger.info(f"Initialized button '{name}' on pin {pin}")
    
    def _button_pressed(self, button_name: str):
        """Handle button press"""
        logger.info(f"Button pressed: {button_name}")
        
        # Handle special buttons
        if button_name == "privacy_mode":
            self.privacy_mode = not self.privacy_mode
            logger.info(f"Privacy mode: {'ON' if self.privacy_mode else 'OFF'}")
        
        # Call registered callback
        if button_name in self.callbacks:
            asyncio.create_task(self.callbacks[button_name]())
    
    def register_callback(self, button_name: str, callback: Callable):
        """Register a callback for a button"""
        if button_name in self.button_pins:
            self.callbacks[button_name] = callback
            logger.info(f"Registered callback for button '{button_name}'")
    
    def is_privacy_mode(self) -> bool:
        """Check if privacy mode is enabled"""
        return self.privacy_mode
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if HAS_GPIO:
            GPIO.cleanup()


class HardwareDevice:
    """Main hardware device controller"""
    
    def __init__(self):
        self.led_controller = LEDController()
        self.button_controller = ButtonController()
        self.device_state = {
            "listening": False,
            "processing": False,
            "speaking": False,
            "privacy_mode": False,
            "last_interaction": None
        }
        
        logger.info("Hardware device initialized")
    
    async def initialize(self):
        """Initialize hardware components"""
        # Set idle pattern
        await self.led_controller.set_pattern("idle")
        
        # Register default button callbacks
        self.button_controller.register_callback(
            "privacy_mode",
            self._toggle_privacy_mode
        )
    
    async def _toggle_privacy_mode(self):
        """Toggle privacy mode"""
        self.device_state["privacy_mode"] = self.button_controller.is_privacy_mode()
        
        if self.device_state["privacy_mode"]:
            await self.led_controller.set_pattern("off")
            logger.info("Privacy mode enabled - LEDs off")
        else:
            await self.led_controller.set_pattern("idle")
            logger.info("Privacy mode disabled - resuming normal operation")
    
    async def set_listening(self, is_listening: bool):
        """Set listening state"""
        self.device_state["listening"] = is_listening
        
        if not self.device_state["privacy_mode"]:
            if is_listening:
                await self.led_controller.set_pattern("listening")
            else:
                await self.led_controller.set_pattern("idle")
    
    async def set_processing(self, is_processing: bool):
        """Set processing state"""
        self.device_state["processing"] = is_processing
        
        if not self.device_state["privacy_mode"]:
            if is_processing:
                await self.led_controller.set_pattern("thinking")
    
    async def set_speaking(self, is_speaking: bool):
        """Set speaking state"""
        self.device_state["speaking"] = is_speaking
        
        if not self.device_state["privacy_mode"]:
            if is_speaking:
                await self.led_controller.set_pattern("speaking")
            else:
                await self.led_controller.set_pattern("idle")
    
    def register_button_callback(self, button_name: str, callback: Callable):
        """Register a button callback"""
        self.button_controller.register_callback(button_name, callback)
    
    def get_device_state(self) -> Dict[str, Any]:
        """Get current device state"""
        return {
            **self.device_state,
            "timestamp": datetime.now().isoformat()
        }
    
    def cleanup(self):
        """Clean up hardware resources"""
        self.led_controller.cleanup()
        self.button_controller.cleanup()
        logger.info("Hardware device cleaned up") 