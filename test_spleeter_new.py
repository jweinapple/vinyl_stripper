#!/usr/bin/env python3
"""
Spleeter Test Script - Built from scratch with TensorFlow compatibility fixes
Uses Universal Audio (ua) as input and Mac speakers as output

IMPORTANT: On Raspberry Pi, Python 3.11 is required due to TensorFlow compatibility.
Python 3.13+ will cause "Unsupported object type NoneType" errors.
See PYTHON_VERSION_REQUIREMENT.md for setup instructions.
"""

import sys
import os

# Check Python version on Raspberry Pi
if os.path.exists('/proc/device-tree/model'):
    try:
        with open('/proc/device-tree/model', 'r') as f:
            if 'raspberry pi' in f.read().lower():
                python_version = sys.version_info
                if python_version.major == 3 and python_version.minor >= 13:
                    print("=" * 60)
                    print("WARNING: Python 3.13+ detected on Raspberry Pi")
                    print("=" * 60)
                    print("Spleeter requires Python 3.11 due to TensorFlow compatibility.")
                    print("You may encounter 'Unsupported object type NoneType' errors.")
                    print("")
                    print("To fix:")
                    print("  1. Run: ./setup_python311_pi.sh")
                    print("  2. Use: source venv311/bin/activate")
                    print("")
                    print("See PYTHON_VERSION_REQUIREMENT.md for details.")
                    print("=" * 60)
                    print("")
                    # Don't exit - let user decide if they want to proceed
    except:
        pass

# TensorFlow compatibility fix - must be done before importing tensorflow
try:
    import tensorflow as tf
    import os
    
    # Optimize TensorFlow for maximum performance
    num_cores = os.cpu_count() or 4
    # Set number of threads (use all CPU cores)
    os.environ['TF_NUM_INTEROP_THREADS'] = str(num_cores)
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(num_cores)
    
    # Enable TensorFlow optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations
    os.environ['TF_DISABLE_MKL'] = '0'  # Enable MKL if available
    
    # Disable TensorFlow warnings for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    
    # Optimize TensorFlow session config
    try:
        # Configure session for better performance
        tf.config.threading.set_inter_op_parallelism_threads(num_cores)
        tf.config.threading.set_intra_op_parallelism_threads(num_cores)
        
        # Try to enable Metal GPU acceleration on Mac
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ Enabled {len(gpus)} GPU device(s) for acceleration")
        except:
            pass
    except:
        pass
    
    print(f"✓ TensorFlow optimized: {num_cores} threads, oneDNN enabled")
    
    # Fix compat.v1.keras import issue for Spleeter
    # Spleeter tries to import 'tensorflow.compat.v1.keras.initializers' as a module
    # We need to create the module structure in sys.modules BEFORE Spleeter imports
    try:
        import sys
        import types
        
        # Ensure compat.v1 exists
        if not hasattr(tf.compat, 'v1'):
            tf.compat.v1 = types.ModuleType('tensorflow.compat.v1')
        
        # Create the module structure that Spleeter expects
        if 'tensorflow.compat.v1.keras' not in sys.modules:
            # Import actual keras
            import tensorflow.keras as tf_keras
            
            # Create module structure
            keras_module = types.ModuleType('tensorflow.compat.v1.keras')
            keras_module.initializers = tf_keras.initializers
            keras_module.layers = tf_keras.layers
            keras_module.models = tf_keras.models
            keras_module.backend = tf_keras.backend
            
            # Also set as attribute
            tf.compat.v1.keras = keras_module
            
            # Add to sys.modules so importlib can find it
            sys.modules['tensorflow.compat.v1.keras'] = keras_module
            sys.modules['tensorflow.compat.v1.keras.initializers'] = tf_keras.initializers
            sys.modules['tensorflow.compat.v1.keras.layers'] = tf_keras.layers
            
            print("✓ Patched tensorflow.compat.v1.keras module structure")
        else:
            print("✓ tensorflow.compat.v1.keras already patched")
    except Exception as e:
        print(f"Warning: Could not patch tensorflow.compat.v1.keras: {e}")
        import traceback
        traceback.print_exc()
    
    # Try to add estimator if missing
    if not hasattr(tf, 'estimator'):
        try:
            import tensorflow_estimator
            # Monkey patch tf.estimator
            import tensorflow_estimator.python.estimator.estimator as estimator_module
            tf.estimator = estimator_module
            # Also add RunConfig
            from tensorflow_estimator.python.estimator import run_config
            tf.estimator.RunConfig = run_config.RunConfig
            print("✓ Patched tf.estimator from tensorflow_estimator")
        except Exception as e:
            print(f"Warning: Could not patch tf.estimator: {e}")
            # Try direct import
            try:
                from tensorflow.python.estimator import estimator as tf_estimator
                tf.estimator = tf_estimator
                print("✓ Using tf.python.estimator")
            except:
                pass
except ImportError:
    print("Error: TensorFlow not installed")
    sys.exit(1)

import numpy as np
import sounddevice as sd
import time
import logging
from collections import deque
import queue
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Platform detection
IS_RASPBERRY_PI = os.path.exists('/proc/device-tree/model') and \
                  'raspberry pi' in open('/proc/device-tree/model').read().lower() \
                  if os.path.exists('/proc/device-tree/model') else False

# Platform-specific defaults
if IS_RASPBERRY_PI:
    DEFAULT_CHUNK_DURATION = 2.5  # Pi needs smaller chunks
    DEFAULT_MIN_BUFFER = 15.0      # Pi needs larger buffer
    DEFAULT_LOW_THRESHOLD = 5.0   # Pi needs higher threshold
else:
    DEFAULT_CHUNK_DURATION = 3.0   # Mac can handle larger chunks
    DEFAULT_MIN_BUFFER = 8.0       # Mac needs smaller buffer
    DEFAULT_LOW_THRESHOLD = 2.0    # Mac can use lower threshold


def find_universal_audio_device():
    """Auto-detect Universal Audio Thunderbolt interface."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if 'apollo' in name or 'universal audio' in name or 'ua ' in name:
            logger.info(f"Found Universal Audio device: {i} - {device['name']}")
            return i
    return None


def find_mac_speakers():
    """Auto-detect Mac built-in speakers."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if device['max_output_channels'] >= 2:
            if 'built-in' in name or ('macbook' in name and 'speaker' in name) or \
               ('mac' in name and 'output' in name):
                if device['max_input_channels'] == 0:
                    logger.info(f"Found Mac speakers: {i} - {device['name']}")
                    return i
    # Fallback: try default output device
    try:
        default_output = sd.query_devices(kind='output')
        logger.info(f"Using default output device: {default_output['index']} - {default_output['name']}")
        return default_output['index']
    except:
        pass
    return None


def find_usb_audio_device():
    """Auto-detect USB audio interface (for Pi) - UFO202 from Soundburger."""
    devices = sd.query_devices()
    # First, look for UFO202/UCA202 specifically (Soundburger vinyl player)
    # These typically show as "USB Audio" with input only
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if ('ufo202' in name or 'uca202' in name) and device['max_input_channels'] > 0:
            logger.info(f"Found UFO202/UCA202 input device: {i} - {device['name']}")
            return i
    # Look for USB Audio device with input but no output (typical UFO202 pattern)
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if 'usb' in name and 'audio' in name and device['max_input_channels'] > 0 and device['max_output_channels'] == 0:
            logger.info(f"Found USB Audio input device (likely UFO202): {i} - {device['name']}")
            return i
    # Then look for stereo USB devices (preferred)
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if 'usb' in name and device['max_input_channels'] >= 2:
            logger.info(f"Found stereo USB audio device: {i} - {device['name']}")
            return i
    # Then look for any USB device with input (mono is OK, we'll handle it)
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if 'usb' in name and device['max_input_channels'] > 0:
            logger.info(f"Found USB audio device (mono): {i} - {device['name']}")
            return i
    # Fallback: use first device with input
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            logger.info(f"Using device {i} as fallback: {device['name']}")
            return i
    return None


def find_pi_output_device():
    """Auto-detect Pi output device (USB to 3.5mm adapter, HiFi DAC HAT, USB CODEC, or headphone jack)."""
    devices = sd.query_devices()
    # Look for USB Audio CODEC with output but no input (JSAUX USB to 3.5mm adapter pattern)
    # This is typically device 1 when UFO202 is device 0
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if device['max_output_channels'] >= 2 and device['max_input_channels'] == 0:
            if 'usb' in name and ('codec' in name or 'audio' in name):
                logger.info(f"Found USB audio adapter output (JSAUX): {i} - {device['name']}")
                return i
    # Look for USB audio devices with output (but not UFO202 input)
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if device['max_output_channels'] >= 2:
            # Look for USB audio devices that are NOT the UFO202 input
            if 'usb' in name and ('codec' in name or 'audio' in name):
                # Check if this is NOT the input device (UFO202 typically has 'ufo202' or 'uca202' in name)
                if 'ufo202' not in name and 'uca202' not in name:
                    logger.info(f"Found USB audio adapter output: {i} - {device['name']}")
                    return i
    # Look for HiFi DAC HAT
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if device['max_output_channels'] >= 2:
            if ('hifiberry' in name and 'dac' in name) or 'bcm2835' in name:
                logger.info(f"Found Pi output device: {i} - {device['name']}")
                return i
    # Fallback: first stereo output device that's not the input
    for i, device in enumerate(devices):
        if device['max_output_channels'] >= 2:
            name = device['name'].lower()
            # Skip if it's the UFO202 input device
            if 'ufo202' not in name and 'uca202' not in name:
                logger.info(f"Using output device: {i} - {device['name']}")
                return i
    # Last resort: default output
    try:
        default_output = sd.query_devices(kind='output')
        logger.info(f"Using default output device: {default_output['index']} - {default_output['name']}")
        return default_output['index']
    except:
        pass
    return None


class SpleeterProcessor:
    """Spleeter-based real-time audio processor."""
    
    def __init__(self, input_device, output_device, remove_vocals=True):
        self.input_device = input_device
        self.output_device = output_device
        self.remove_vocals = remove_vocals
        self.sample_rate = 44100
        self.chunk_duration = DEFAULT_CHUNK_DURATION  # Platform-specific chunk size
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        self.overlap_samples = int(0.1 * self.sample_rate)  # 100ms overlap for smooth transitions
        self.hop_samples = self.chunk_samples - self.overlap_samples
        
        # Initialize Spleeter
        logger.info("Initializing Spleeter...")
        try:
            from spleeter.separator import Separator
            
            model_name = "spleeter:2stems" if remove_vocals else "spleeter:4stems"
            logger.info(f"Using model: {model_name}")
            
            # Create separator with optimizations
            # multiprocess=False is faster for single-threaded processing
            self.separator = Separator(model_name, multiprocess=False)
            self.stems = 2 if remove_vocals else 4
            
            # Optimize Spleeter's internal configuration if possible
            try:
                # Set TensorFlow session config for Spleeter
                # This will be used when Spleeter creates its estimator
                import tensorflow as tf
                # Configure for better performance
                session_config = tf.compat.v1.ConfigProto(
                    intra_op_parallelism_threads=os.cpu_count() or 4,
                    inter_op_parallelism_threads=os.cpu_count() or 4,
                    allow_soft_placement=True,
                    log_device_placement=False,
                )
                # Enable optimizations
                session_config.graph_options.optimizer_options.opt_level = tf.compat.v1.OptimizerOptions.L1
                session_config.graph_options.rewrite_options.constant_folding = tf.compat.v1.ConfigProto.ON
                # Store for potential use
                self._session_config = session_config
            except:
                self._session_config = None
            
            # Pre-load model to avoid reloading on each chunk
            logger.info("Pre-loading Spleeter model (this may take a moment)...")
            try:
                # Warm up the model with a dummy audio chunk
                dummy_audio = np.zeros((self.chunk_samples, 2), dtype=np.float32)
                _ = self.separator.separate(dummy_audio)
                logger.info("✓ Model pre-loaded and warmed up")
            except Exception as e:
                logger.warning(f"Model pre-load warning: {e}")
                # Try alternative pre-load method
                try:
                    self.separator._get_prediction_generator()
                    logger.info("✓ Model pre-loaded via prediction generator")
                except Exception as e2:
                    logger.warning(f"Alternative pre-load also failed: {e2}")
            
            logger.info("Spleeter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Spleeter: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Buffers
        self.input_buffer = deque(maxlen=self.chunk_samples * 20)
        self.output_buffer = np.zeros((0, 2), dtype=np.float32)
        self.passthrough_buffer = deque(maxlen=int(self.sample_rate * 10))
        
        # Processing queue (larger for better throughput)
        self.processing_queue = queue.Queue(maxsize=30)  # Larger queue for better throughput
        self.output_queue = queue.Queue(maxsize=30)
        
        # State
        self.running = False
        self.use_passthrough = True
        self.prebuffer_complete = False
        self.min_buffer_samples = int(DEFAULT_MIN_BUFFER * self.sample_rate)  # Platform-specific buffer
        self.low_buffer_threshold = int(DEFAULT_LOW_THRESHOLD * self.sample_rate)  # Platform-specific threshold
        
        # Processing stats
        self.processing_times = deque(maxlen=20)
        self.chunk_counter = 0
        self.processed_chunks = 0
        
        # Thread lock for Spleeter (it's not fully thread-safe)
        self.processing_lock = threading.Lock()
    
    def process_chunk(self, audio):
        """Process a chunk of audio with Spleeter."""
        start_time = time.time()
        
        try:
            # Ensure correct format: (samples, channels) as float32
            if audio.shape[1] != 2:
                logger.warning(f"Expected stereo audio, got shape: {audio.shape}")
                return np.zeros_like(audio)
            
            audio = audio.astype(np.float32)
            
            # Optimize audio format for faster processing
            # Ensure contiguous array (faster processing)
            if not audio.flags['C_CONTIGUOUS']:
                audio = np.ascontiguousarray(audio)
            
            # Normalize audio to prevent clipping and improve processing
            audio_max = np.abs(audio).max()
            if audio_max > 0:
                audio = audio / max(audio_max, 1.0)
            
            # Spleeter expects (samples, channels) format
            # Separate stems (model should be pre-loaded, so this should be fast)
            # Use lock to ensure thread-safe access to Spleeter
            with self.processing_lock:
                prediction = self.separator.separate(audio)
            
            # Denormalize if needed
            if audio_max > 0:
                for key in prediction:
                    prediction[key] = prediction[key] * audio_max
            
            # Extract the stems we want
            if self.stems == 2:
                # 2-stem model: vocals, accompaniment
                if self.remove_vocals:
                    output = prediction.get('accompaniment', np.zeros_like(audio))
                else:
                    output = prediction.get('vocals', np.zeros_like(audio))
            else:
                # 4-stem model: vocals, drums, bass, other
                if self.remove_vocals:
                    output = prediction.get('drums', np.zeros_like(audio)) + \
                             prediction.get('bass', np.zeros_like(audio)) + \
                             prediction.get('other', np.zeros_like(audio))
                else:
                    output = prediction.get('vocals', np.zeros_like(audio))
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.processed_chunks += 1
            
            logger.info(f"✓ Processed chunk {self.chunk_counter} in {processing_time:.2f}s "
                       f"(real-time ratio: {processing_time/self.chunk_duration:.2f}x)")
            
            return output
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros_like(audio)
    
    def processing_worker(self):
        """Background worker for processing chunks (optimized for speed)."""
        logger.info("Processing worker started")
        while self.running:
            try:
                chunk_id, chunk = self.processing_queue.get(timeout=1.0)
                if chunk_id is None:
                    break
                
                # Process immediately (no batching overhead)
                processed = self.process_chunk(chunk)
                self.output_queue.put((chunk_id, processed))
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
        logger.info("Processing worker stopped")
    
    def output_collector(self):
        """Collect processed chunks in order with crossfading."""
        pending_chunks = {}
        next_chunk_id = 0
        
        while self.running:
            try:
                chunk_id, processed = self.output_queue.get(timeout=1.0)
                pending_chunks[chunk_id] = processed
                
                # Add chunks in order with crossfading
                while next_chunk_id in pending_chunks:
                    chunk = pending_chunks.pop(next_chunk_id)
                    
                    # Crossfade overlap region if we have existing buffer
                    if self.output_buffer.shape[0] >= self.overlap_samples and chunk.shape[0] >= self.overlap_samples:
                        # Crossfade the overlap region
                        old_overlap = self.output_buffer[-self.overlap_samples:]
                        new_overlap = chunk[:self.overlap_samples]
                        
                        # Create fade curves
                        fade_out = np.linspace(1.0, 0.0, self.overlap_samples).reshape(-1, 1)
                        fade_in = np.linspace(0.0, 1.0, self.overlap_samples).reshape(-1, 1)
                        
                        # Crossfade
                        crossfaded = old_overlap * fade_out + new_overlap * fade_in
                        self.output_buffer[-self.overlap_samples:] = crossfaded
                        
                        # Add rest of chunk
                        if chunk.shape[0] > self.overlap_samples:
                            self.output_buffer = np.vstack([
                                self.output_buffer,
                                chunk[self.overlap_samples:]
                            ])
                    else:
                        # No overlap, just append
                        self.output_buffer = np.vstack([self.output_buffer, chunk])
                    
                    next_chunk_id += 1
                    
                    # Check if prebuffer is ready
                    if not self.prebuffer_complete:
                        buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
                        if self.output_buffer.shape[0] >= self.min_buffer_samples:
                            self.prebuffer_complete = True
                            logger.info(f"✓ Pre-buffer complete: {buffer_seconds:.1f}s ready")
                            logger.info("Switching to processed audio...")
                            self.use_passthrough = False
            except queue.Empty:
                continue
    
    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Audio callback for real-time processing."""
        if status:
            logger.debug(f"Audio status: {status}")
        
        # Convert mono to stereo if needed
        if indata.shape[1] == 1:
            # Mono input: duplicate to stereo
            indata_stereo = np.repeat(indata, 2, axis=1)
        else:
            indata_stereo = indata
        
        # Store passthrough
        self.passthrough_buffer.extend(indata_stereo.copy())
        
        # Add to input buffer
        self.input_buffer.extend(indata_stereo.copy())
        
        # Queue chunks for processing (with hop to create overlap)
        while len(self.input_buffer) >= self.chunk_samples:
            chunk = np.array(list(self.input_buffer)[:self.chunk_samples])
            
            # Remove hop_samples from buffer (keep overlap)
            for _ in range(self.hop_samples):
                if self.input_buffer:
                    self.input_buffer.popleft()
            
            # Queue for processing
            try:
                self.processing_queue.put_nowait((self.chunk_counter, chunk))
                self.chunk_counter += 1
            except queue.Full:
                logger.warning("Processing queue full - dropping chunk")
        
        # Output logic with buffer monitoring
        buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
        
        # Check if buffer is getting too low
        if not self.use_passthrough and buffer_seconds < self.low_buffer_threshold / self.sample_rate:
            logger.warning(f"Buffer low ({buffer_seconds:.1f}s) - switching to passthrough")
            self.use_passthrough = True
        
        if not self.use_passthrough and self.output_buffer.shape[0] >= frames:
            # Use processed audio
            outdata[:] = self.output_buffer[:frames]
            self.output_buffer = self.output_buffer[frames:]
        elif len(self.passthrough_buffer) >= frames:
            # Passthrough mode
            passthrough_data = np.array(list(self.passthrough_buffer)[:frames])
            outdata[:] = passthrough_data
            for _ in range(frames):
                if self.passthrough_buffer:
                    self.passthrough_buffer.popleft()
        else:
            # Silence fallback (shouldn't happen often)
            outdata[:] = 0
    
    def run(self):
        """Start real-time processing."""
        self.running = True
        
        # Start worker threads
        worker_thread = threading.Thread(target=self.processing_worker, daemon=True)
        worker_thread.start()
        
        collector_thread = threading.Thread(target=self.output_collector, daemon=True)
        collector_thread.start()
        
        logger.info("=" * 60)
        logger.info("Spleeter Test - Starting audio stream")
        logger.info(f"Input:  Device {self.input_device}")
        logger.info(f"Output: Device {self.output_device}")
        logger.info(f"Mode:   {'Remove vocals' if self.remove_vocals else 'Keep vocals only'}")
        logger.info("=" * 60)
        logger.info("Building buffer... (audio will play through, then switch to processed)")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        # Detect input/output channel counts and sample rate
        input_info = sd.query_devices(self.input_device)
        output_info = sd.query_devices(self.output_device)
        input_channels = input_info['max_input_channels']
        output_channels = output_info['max_output_channels']
        
        # Use actual channel counts (handle mono input)
        input_ch = min(input_channels, 1) if input_channels > 0 else 1
        output_ch = min(output_channels, 2) if output_channels > 0 else 2
        
        # Use device's preferred sample rate if different from default
        device_sample_rate = int(input_info.get('default_samplerate', self.sample_rate))
        if device_sample_rate != self.sample_rate:
            logger.info(f"Device sample rate ({device_sample_rate} Hz) differs from default ({self.sample_rate} Hz)")
            logger.info(f"Using device sample rate: {device_sample_rate} Hz")
            # Update sample rate and recalculate chunk sizes
            self.sample_rate = device_sample_rate
            self.chunk_samples = int(self.chunk_duration * self.sample_rate)
            self.overlap_samples = int(0.1 * self.sample_rate)
            self.hop_samples = self.chunk_samples - self.overlap_samples
            self.min_buffer_samples = int(DEFAULT_MIN_BUFFER * self.sample_rate)
            self.low_buffer_threshold = int(DEFAULT_LOW_THRESHOLD * self.sample_rate)
        
        logger.info(f"Stream config: {input_ch} input channel(s), {output_ch} output channel(s), {self.sample_rate} Hz")
        
        try:
            with sd.Stream(
                samplerate=self.sample_rate,
                channels=(input_ch, output_ch),  # Different channel counts for input/output
                dtype=np.float32,
                blocksize=2048,  # Smaller blocksize for lower latency
                latency='high',
                device=(self.input_device, self.output_device),
                callback=self.audio_callback,
            ):
                last_status_time = time.time()
                while self.running:
                    sd.sleep(100)
                    
                    # Periodic status every 5 seconds
                    current_time = time.time()
                    if current_time - last_status_time >= 5.0:
                        if self.processing_times:
                            avg_time = np.mean(self.processing_times)
                            buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
                            queue_size = self.processing_queue.qsize()
                            mode = "PROCESSED" if not self.use_passthrough else "passthrough"
                            logger.info(
                                f"[{mode}] Buffer: {buffer_seconds:.1f}s | "
                                f"Queue: {queue_size} | "
                                f"Processed: {self.processed_chunks} chunks | "
                                f"Avg: {avg_time:.2f}s/chunk"
                            )
                        last_status_time = current_time
        
        except KeyboardInterrupt:
            logger.info("\nStopping...")
        finally:
            self.running = False
            self.processing_queue.put((None, None))
            worker_thread.join(timeout=2)
            collector_thread.join(timeout=2)
            logger.info(f"Stopped. Processed {self.processed_chunks} chunks total.")


def main():
    """Main entry point."""
    logger.info("Spleeter Test Script (New)")
    logger.info("=" * 60)
    
    # Find devices (platform-specific)
    logger.info("Detecting audio devices...")
    if IS_RASPBERRY_PI:
        logger.info("Platform: Raspberry Pi")
        input_device = find_usb_audio_device()
        if input_device is None:
            logger.error("USB audio device (UFO202) not found!")
            logger.info("\nAvailable input devices:")
            devices = sd.query_devices()
            input_found = False
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    logger.info(f"  {i}: {device['name']}")
                    input_found = True
            if not input_found:
                logger.error("\nNo input devices found! Please ensure:")
                logger.error("  1. UFO202 is connected via USB")
                logger.error("  2. USB devices are properly recognized by the system")
                logger.error("  3. Run 'lsusb' to verify USB devices are detected")
            sys.exit(1)
        
        output_device = find_pi_output_device()
        if output_device is None:
            logger.error("Pi output device not found!")
            logger.info("\nAvailable output devices:")
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    logger.info(f"  {i}: {device['name']}")
            sys.exit(1)
    else:
        logger.info("Platform: Mac")
        input_device = find_universal_audio_device()
        if input_device is None:
            logger.error("Universal Audio device not found!")
            logger.info("\nAvailable input devices:")
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    logger.info(f"  {i}: {device['name']}")
            sys.exit(1)
        
        output_device = find_mac_speakers()
        if output_device is None:
            logger.error("Mac speakers not found!")
            logger.info("\nAvailable output devices:")
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    logger.info(f"  {i}: {device['name']}")
            sys.exit(1)
    
    # Create and run processor
    try:
        processor = SpleeterProcessor(
            input_device=input_device,
            output_device=output_device,
            remove_vocals=True
        )
        processor.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

