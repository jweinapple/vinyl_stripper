#!/usr/bin/env python3
"""
Spleeter Test Script - Built from scratch with TensorFlow compatibility fixes
Uses Universal Audio (ua) as input and Mac speakers as output
"""

import sys
import os

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


class SpleeterProcessor:
    """Spleeter-based real-time audio processor."""
    
    def __init__(self, input_device, output_device, remove_vocals=True):
        self.input_device = input_device
        self.output_device = output_device
        self.remove_vocals = remove_vocals
        self.sample_rate = 44100
        self.chunk_duration = 3.0  # Larger chunks = less overhead per second (optimized for efficiency)
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
        self.min_buffer_samples = int(8.0 * self.sample_rate)  # 8 second buffer (reduced since faster processing)
        self.low_buffer_threshold = int(2.0 * self.sample_rate)  # Fallback to passthrough if below 2s
        
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
        
        # Store passthrough
        self.passthrough_buffer.extend(indata.copy())
        
        # Add to input buffer
        self.input_buffer.extend(indata.copy())
        
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
        
        try:
            with sd.Stream(
                samplerate=self.sample_rate,
                channels=2,
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
    
    # Find devices
    logger.info("Detecting audio devices...")
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

