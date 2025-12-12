#!/usr/bin/env python3
"""
Vinyl Stem Stripper - Real-time vocal/drum removal for vinyl playback
Optimized for Raspberry Pi 5 with USB audio interface

Usage:
    python3 vinyl_stripper.py --list-devices
    python3 vinyl_stripper.py --input 1 --output 1 --remove vocals
    python3 vinyl_stripper.py --remove vocals drums
    
Available stems: vocals, drums, bass, other
"""

import argparse
import sys
import threading
import queue
import numpy as np
from collections import deque
import logging
import os
import traceback
from datetime import datetime

# Check dependencies before importing
def check_dependencies():
    missing = []
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import demucs
    except ImportError:
        missing.append("demucs")
    
    if missing:
        print("Missing dependencies. Install them with:")
        print(f"  pip3 install {' '.join(missing)}")
        if "demucs" in missing:
            print("  # For demucs specifically:")
            print("  pip3 install demucs")
        sys.exit(1)

check_dependencies()

import sounddevice as sd
import torch

# Demucs imports
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Setup logging
def setup_logging():
    """Setup logging to both file and console."""
    log_dir = os.path.expanduser("~/vinyl_stripper/logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"vinyl_stripper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create logger
    logger = logging.getLogger('vinyl_stripper')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (info and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log startup info
    logger.info(f"Logging initialized - log file: {log_file}")
    
    return logger

logger = setup_logging()

def log_memory_status(stage=""):
    """Log current memory status."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        logger.info(f"Memory status {stage}:")
        logger.info(f"  RAM: {mem.used / (1024**3):.2f}GB / {mem.total / (1024**3):.2f}GB ({mem.percent}% used)")
        logger.info(f"  Available: {mem.available / (1024**3):.2f}GB")
        logger.info(f"  Swap: {swap.used / (1024**3):.2f}GB / {swap.total / (1024**3):.2f}GB ({swap.percent}% used)")
        return mem.available, mem.percent
    except ImportError:
        # Fallback to /proc/meminfo on Linux
        try:
            if os.path.exists('/proc/meminfo'):
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                for line in meminfo.split('\n'):
                    if 'MemAvailable' in line:
                        available_kb = int(line.split()[1])
                        logger.info(f"Memory status {stage}: Available: {available_kb / (1024**2):.2f}GB")
                        return available_kb * 1024, None
        except:
            pass
        # Final fallback
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            logger.info(f"Memory usage {stage}: {usage.ru_maxrss / 1024:.2f}MB")
        except:
            logger.warning("Could not determine memory status")
        return None, None


class VinylStripper:
    def __init__(
        self,
        input_device: int,
        output_device: int,
        remove_stems: list[str],
        chunk_duration: float = 0.25,  # seconds per chunk (very small for fastest processing)
        overlap: float = 0.03,  # overlap for smoother transitions (increased to prevent gaps)
        sample_rate: int = 44100,
        model_name: str = None,  # Model name (default: htdemucs_ft)
    ):
        self.input_device = input_device
        self.output_device = output_device
        self.remove_stems = remove_stems
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.model_name = model_name
        
        # Calculate sizes
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(self.chunk_samples * overlap)
        self.hop_samples = self.chunk_samples - self.overlap_samples
        
        self.input_buffer = deque(maxlen=self.chunk_samples * 16)
        self.output_queue = queue.Queue(maxsize=100)  # Increased buffer size
        self.output_buffer = np.zeros((0, 2), dtype=np.float32)
        self.passthrough_buffer = deque(maxlen=int(sample_rate * 5))  # Increased passthrough buffer
        self.processing_queue = queue.Queue(maxsize=30)  # Increased processing queue
        self.running = False
        self.buffer_filling = True
        self.use_passthrough = False  # Disabled - only play processed audio
        self.last_buffer_log = 0
        self.buffer_log_interval = 5.0  # Log buffer status every 5 seconds
        self.min_buffer_before_switch = int(sample_rate * 3.0)  # Need at least 3s before switching from passthrough (reduced from 5s)
        
        # Load model with device-specific optimization
        # Reference torch before any local imports to avoid UnboundLocalError
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = self.model_name if self.model_name else "htdemucs_ft"
        
        logger.info(f"Loading model: {model_name} (device: {self.device})...")
        
        # Check memory before loading
        available_mem, mem_percent = log_memory_status("before model load")
        if available_mem and available_mem < 1.0 * (1024**3):  # Less than 1GB available
            logger.warning(f"Low memory available ({available_mem / (1024**3):.2f}GB). Model loading may fail.")
            logger.warning("Consider using lighter model (htdemucs) or increasing swap space.")
        
        try:
            # Set environment variables for better CPU performance
            # Use more threads for better parallel processing
            import multiprocessing
            num_threads = min(multiprocessing.cpu_count(), 4)  # Use up to 4 threads
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            logger.info(f"Using {num_threads} CPU threads for processing")
            
            logger.info("Fetching model from demucs...")
            self.model = get_model(model_name)
            logger.info("Model downloaded successfully")
            
            # Check memory after download
            log_memory_status("after model download")
            
            logger.info(f"Moving model to device: {self.device}")
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded and ready")
            
        except MemoryError as e:
            logger.error(f"Out of memory while loading model: {e}")
            log_memory_status("OOM error")
            logger.error("Model loading failed due to insufficient memory.")
            logger.error("Solutions:")
            logger.error("  1. Use lighter model: --model htdemucs")
            logger.error("  2. Increase swap space: sudo fallocate -l 2G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile")
            logger.error("  3. Close other applications to free memory")
            raise
        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.error("\nAvailable models: htdemucs, htdemucs_ft, htdemucs_6s")
            raise
        
        # Optimize model for inference
        try:
            from demucs.apply import BagOfModels
            is_bag_of_models = isinstance(self.model, BagOfModels)
        except ImportError:
            is_bag_of_models = False
        
        if not is_bag_of_models:
            if hasattr(torch, 'compile') and self.device != 'cpu':
                try:
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                    logger.info("✓ Model compiled")
                except Exception as e:
                    logger.debug(f"Model compilation skipped: {e}")
                    pass
            
            if self.device == "cpu":
                try:
                    from torch import quantization as torch_quantization
                    self.model = torch_quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("✓ Model quantized")
                except Exception as e:
                    logger.debug(f"Model quantization skipped: {e}")
                    pass
        
        # Stem indices in demucs output: drums, bass, other, vocals
        self.stem_indices = {
            "drums": 0,
            "bass": 1,
            "other": 2,
            "vocals": 3,
        }
        
        logger.info(f"Removing: {', '.join(remove_stems)}")
        logger.info(f"Stem indices: {self.stem_indices}")
        logger.info(f"Will keep: {[s for s in self.stem_indices.keys() if s not in remove_stems]}")
        logger.info(f"Chunk: {chunk_duration}s, Overlap: {overlap*100}%, Latency: ~{chunk_duration + 1:.1f}s")
        
        # Final memory check
        log_memory_status("after initialization")

    def process_chunk(self, audio: np.ndarray) -> np.ndarray:
        """Run stem separation on a chunk and return audio with stems removed."""
        # Convert to torch tensor: (batch, channels, samples)
        audio_tensor = torch.from_numpy(audio.T).unsqueeze(0).float().to(self.device)
        
        # Normalize
        ref = audio_tensor.mean(0)
        ref_std = ref.std()
        # Avoid division by zero
        if ref_std < 1e-8:
            ref_std = torch.tensor(1.0, device=self.device)
        audio_tensor = (audio_tensor - ref.mean()) / ref_std
        
        try:
            with torch.no_grad():
                # Apply model - returns (batch, stems, channels, samples)
                # Optimized for longer chunks and better memory management
                chunk_length = audio_tensor.shape[-1]
                chunk_duration = chunk_length / self.sample_rate
                
                # Configure segment size based on device and chunk size
                # For CPU, use larger segments to process more audio at once
                # For CUDA, can process entire chunk
                if self.device == "cuda":
                    # GPU can handle larger chunks
                    segment_size = None  # Process entire chunk
                    use_split = False
                else:
                    # CPU: Use larger segments for better throughput
                    # Process in 4-8 second segments for better efficiency
                    if chunk_duration <= 4.0:
                        # Small chunk - process whole thing
                        segment_size = None
                        use_split = False
                    else:
                        # Larger chunk - use 6 second segments
                        segment_size = int(self.sample_rate * 6.0)  # 6 second segments
                        use_split = True
                
                logger.debug(f"Processing chunk: {chunk_duration:.2f}s, segment={segment_size}, split={use_split}, device={self.device}")
                
                sources = apply_model(
                    self.model, 
                    audio_tensor, 
                    device=self.device,
                    segment=segment_size,  # None for whole chunk, or segment size
                    split=use_split,  # Enable splitting for large chunks
                    shifts=1,  # Single shift for speed
                    overlap=0.25,  # Small overlap for smoother transitions
                    progress=False,  # Disable progress bar
                )
                
                logger.debug(f"Model returned sources: shape={sources.shape}")
        except Exception as e:
            logger.error(f"Error in apply_model: {type(e).__name__}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise
        
        # sources shape: (1, 4, 2, samples)
        # stems are: drums, bass, other, vocals
        
        # Debug: Check if we're getting valid output from model
        if sources.shape[1] != 4:
            logger.error(f"Unexpected number of stems: {sources.shape[1]} (expected 4)")
        
        # Sum all stems except the ones we want to remove
        output = torch.zeros_like(audio_tensor)
        stems_kept = []
        stems_removed = []
        for stem_name, idx in self.stem_indices.items():
            if stem_name not in self.remove_stems:
                output += sources[0, idx]
                stems_kept.append(stem_name)
            else:
                stems_removed.append(stem_name)
        
        # Debug logging (first chunk only)
        if not hasattr(self, '_logged_first_chunk'):
            logger.info(f"Processing first chunk: Keeping {stems_kept}, Removing {stems_removed}")
            logger.info(f"Output shape: {output.shape}, Mean: {output.mean().item():.6f}, Std: {output.std().item():.6f}")
            logger.info(f"Sources stats - Drums: {sources[0, 0].mean().item():.6f}, Bass: {sources[0, 1].mean().item():.6f}, Other: {sources[0, 2].mean().item():.6f}, Vocals: {sources[0, 3].mean().item():.6f}")
            self._logged_first_chunk = True
        
        # Denormalize - use original reference stats
        ref_std = ref.std()
        ref_mean = ref.mean()
        # Avoid division by zero in denormalization
        if ref_std < 1e-8:
            ref_std = torch.tensor(1.0, device=self.device)
        
        output = output * ref_std + ref_mean
        
        # Check for silence (all zeros or very quiet) - only log first time
        if not hasattr(self, '_checked_silence'):
            output_std = output.std().item()
            output_mean = output.mean().item()
            logger.info(f"Output stats after denormalization: Mean={output_mean:.6f}, Std={output_std:.6f}")
            if output_std < 1e-6:
                logger.warning(f"⚠ Output is very quiet (std: {output_std:.2e}) - possible processing issue")
            self._checked_silence = True
        
        # Convert back to numpy (samples, channels)
        result = output.squeeze(0).T.cpu().numpy()
        
        # Final check - ensure we have valid audio
        if np.allclose(result, 0, atol=1e-6):
            logger.error("⚠ Output is all zeros - processing failed!")
        
        return result

    def processing_worker(self):
        """Background thread that processes audio chunks."""
        chunks_processed = 0
        chunks_failed = 0
        import time as time_module
        
        logger.info("Processing worker started")
        while self.running:
            try:
                chunk = self.processing_queue.get(timeout=0.5)
            except queue.Empty:
                # Log if queue is backing up
                queue_size = self.processing_queue.qsize()
                if queue_size > 10 and chunks_processed == 0:
                    logger.warning(f"⚠ Processing queue backing up ({queue_size} queued) but no chunks processed yet - worker may be stuck")
                continue
            
            if chunk is None:
                break
                
            try:
                # Time the processing to see how long it takes
                start_time = time_module.time()
                logger.debug(f"Starting to process chunk (queue size: {self.processing_queue.qsize()})")
                
                processed = self.process_chunk(chunk)
                processing_time = time_module.time() - start_time
                
                # Check if output is valid
                if processed is None or processed.size == 0:
                    logger.error(f"⚠ Processed chunk is None or empty!")
                    chunks_failed += 1
                    continue
                
                self.output_queue.put(processed)
                chunks_processed += 1
                
                # Log first few chunks and then periodically
                if chunks_processed <= 5 or chunks_processed % 5 == 0:
                    chunk_duration = len(chunk) / self.sample_rate
                    realtime_factor = chunk_duration / processing_time if processing_time > 0 else 0
                    logger.info(f"✓ Processed chunk {chunks_processed}: {processing_time:.2f}s for {chunk_duration:.2f}s audio (RTF: {realtime_factor:.2f}x)")
                    if realtime_factor < 1.0:
                        logger.warning(f"⚠ Processing slower than real-time! Need {1.0/realtime_factor:.1f}x speedup")
                
                if chunks_processed % 10 == 0:
                    logger.debug(f"Processed {chunks_processed} chunks (output queue: {self.output_queue.qsize()}, failed: {chunks_failed})")
            except KeyboardInterrupt:
                logger.info("Processing worker interrupted")
                break
            except Exception as e:
                chunks_failed += 1
                logger.error(f"✗ Processing error on chunk {chunks_processed + chunks_failed}: {type(e).__name__}: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                # Don't break - keep trying to process other chunks
                if chunks_failed > 10:
                    logger.error("Too many processing failures - stopping worker")
                    break
        
        logger.info(f"Processing worker stopped (processed {chunks_processed} chunks, failed {chunks_failed})")

    def audio_callback(self, indata, outdata, frames, time, status):
        """Called by sounddevice for each audio block."""
        if status:
            logger.debug(f"Audio status: {status}")
        
        # Periodic buffer health monitoring
        import time as time_module
        current_time = time_module.time()
        if current_time - self.last_buffer_log > self.buffer_log_interval:
            buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
            queue_size = self.processing_queue.qsize()
            output_queue_size = self.output_queue.qsize()
            mode = "passthrough" if self.use_passthrough else "processed"
            min_needed = self.min_buffer_before_switch / self.sample_rate
            logger.info(f"Buffer status: {buffer_seconds:.1f}s ({mode}), need {min_needed:.1f}s to switch | Processing: {queue_size} queued, {output_queue_size} ready")
            self.last_buffer_log = current_time
        
        # Passthrough buffer disabled - not storing raw audio
        # self.passthrough_buffer.extend(indata.copy())  # Disabled
        
        # Add input to buffer
        self.input_buffer.extend(indata.copy())
        
        # If we have enough samples, queue a chunk for processing
        if len(self.input_buffer) >= self.chunk_samples:
            chunk = np.array(list(self.input_buffer)[:self.chunk_samples])
            
            # Remove hop_samples from the front (keep overlap for next chunk)
            for _ in range(self.hop_samples):
                if self.input_buffer:
                    self.input_buffer.popleft()
            
            # Queue for processing (non-blocking, drop if full)
            # More aggressive queuing - keep processing pipeline busy
            if self.processing_queue.qsize() < self.processing_queue.maxsize * 0.8:  # Even more aggressive
                try:
                    self.processing_queue.put_nowait(chunk)
                except queue.Full:
                    pass  # Skip this chunk if we're backed up
            else:
                # Queue is getting full - log warning occasionally
                if self.processing_queue.qsize() == int(self.processing_queue.maxsize * 0.8):
                    logger.debug(f"Processing queue at {self.processing_queue.qsize()}/{self.processing_queue.maxsize}")
        
        # Try to get processed audio for output
        # Keep filling buffer aggressively to prevent skipping
        target_buffer_size = int(self.sample_rate * 15.0)  # 15 second target buffer for more headroom
        
        # Always try to fill buffer if we have processed chunks available
        # Process multiple chunks per callback to keep buffer full
        chunks_added = 0
        max_chunks_per_callback = 10  # Increased to fill buffer faster
        while chunks_added < max_chunks_per_callback:
            try:
                processed = self.output_queue.get_nowait()
                
                # Handle overlap correctly - only add non-overlapping portion to prevent gaps
                if self.output_buffer.shape[0] > 0:
                    # Skip overlap samples - only add new audio
                    hop_portion = processed[self.overlap_samples:]
                    if hop_portion.shape[0] > 0:
                        self.output_buffer = np.vstack([self.output_buffer, hop_portion])
                else:
                    # First chunk - add entire chunk
                    self.output_buffer = processed
                
                chunks_added += 1
                if self.buffer_filling and self.output_buffer.shape[0] >= target_buffer_size:
                    self.buffer_filling = False
                    logger.info(f"✓ Buffer ready: {self.output_buffer.shape[0] / self.sample_rate:.1f}s")
            except queue.Empty:
                break
        
        # Passthrough mode is disabled - only use processed audio
        # Log buffer status for monitoring
        current_buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
        min_buffer_seconds = self.min_buffer_before_switch / self.sample_rate
        
        if self.buffer_filling and self.output_buffer.shape[0] >= self.min_buffer_before_switch:
            # Buffer is ready
            self.buffer_filling = False
            logger.info(f"✓ Buffer ready ({current_buffer_seconds:.1f}s >= {min_buffer_seconds:.1f}s)")
        elif self.buffer_filling:
            # Still filling - log progress occasionally
            if int(current_buffer_seconds) % 2 == 0 and current_buffer_seconds > 0:
                logger.debug(f"Buffer filling: {current_buffer_seconds:.1f}s / {min_buffer_seconds:.1f}s (processing queue: {self.processing_queue.qsize()}, output queue: {self.output_queue.qsize()})")
        
        # Output only processed audio (passthrough disabled)
        # Check buffer level before outputting
        buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
        min_safe_buffer = 3.0  # Minimum safe buffer in seconds (increased)
        
        if self.output_buffer.shape[0] >= frames:
            # Use processed audio - ensure continuous output
            outdata[:] = self.output_buffer[:frames]
            self.output_buffer = self.output_buffer[frames:]
            
            # Log if buffer is getting low (but not every time to avoid spam)
            new_buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
            if new_buffer_seconds < min_safe_buffer and not self.buffer_filling:
                if not hasattr(self, '_last_low_buffer_log') or (time_module.time() - self._last_low_buffer_log) > 1.0:
                    logger.warning(f"⚠ Buffer low: {new_buffer_seconds:.2f}s remaining (processing: {self.processing_queue.qsize()} queued, {self.output_queue.qsize()} ready)")
                    self._last_low_buffer_log = time_module.time()
        elif self.output_buffer.shape[0] > 0:
            # Partial buffer - use what we have and smoothly fade/extend
            available = self.output_buffer.shape[0]
            outdata[:available] = self.output_buffer
            # Crossfade/extend last samples to avoid clicks
            if available >= 100:  # If we have enough samples, fade out
                fade_length = min(100, available)
                fade_curve = np.linspace(1.0, 0.0, fade_length).reshape(-1, 1)
                outdata[available-fade_length:available] = self.output_buffer[-fade_length:] * fade_curve
            else:
                # Repeat last sample smoothly
                last_sample = self.output_buffer[-1:]
                outdata[available:] = last_sample
            self.output_buffer = np.zeros((0, 2), dtype=np.float32)
            if not hasattr(self, '_last_underrun_log') or (time_module.time() - self._last_underrun_log) > 1.0:
                logger.warning(f"⚠ Buffer underrun - partial output ({buffer_seconds:.2f}s available, need {frames/self.sample_rate:.2f}s)")
                self._last_underrun_log = time_module.time()
        else:
            # No processed audio available - output silence (will cause skipping)
            outdata[:] = 0
            if not self.buffer_filling:
                # Only log occasionally to avoid spam
                if not hasattr(self, '_last_silence_log') or (time_module.time() - self._last_silence_log) > 2.0:
                    logger.warning(f"⚠ No processed audio - skipping (processing: {self.processing_queue.qsize()} queued, {self.output_queue.qsize()} ready)")
                    self._last_silence_log = time_module.time()

    def prefill_buffer(self, duration=10.0):
        """Pre-fill output buffer before starting playback to prevent dropouts."""
        logger.info(f"Pre-filling buffer ({duration}s of audio)...")
        
        try:
            # Record audio for pre-fill duration
            prefill_samples = int(duration * self.sample_rate)
            prefill_audio = sd.rec(
                frames=prefill_samples,
                samplerate=self.sample_rate,
                channels=2,
                device=self.input_device,
                dtype=np.float32
            )
            sd.wait()
            
            # Process the pre-fill audio in chunks
            chunks_to_process = []
            for i in range(0, len(prefill_audio), self.chunk_samples):
                chunk = prefill_audio[i:i+self.chunk_samples]
                if len(chunk) == self.chunk_samples:
                    chunks_to_process.append(chunk)
            
            logger.info(f"Processing {len(chunks_to_process)} pre-fill chunks...")
            
            # Process chunks and fill buffer
            processed_count = 0
            for chunk in chunks_to_process:
                try:
                    processed = self.process_chunk(chunk)
                    hop_portion = processed[:self.hop_samples]
                    self.output_buffer = np.vstack([self.output_buffer, hop_portion]) if self.output_buffer.shape[0] > 0 else hop_portion
                    processed_count += 1
                    if processed_count % 5 == 0:
                        buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
                        logger.info(f"  Processed {processed_count}/{len(chunks_to_process)} chunks ({buffer_seconds:.1f}s buffer)")
                except Exception as e:
                    logger.warning(f"Error processing pre-fill chunk: {e}")
            
            buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
            logger.info(f"✓ Pre-fill complete: {buffer_seconds:.1f}s buffer ready")
        except Exception as e:
            logger.warning(f"Pre-fill failed: {e}. Will start with empty buffer.")

    def run(self):
        """Start the real-time processing."""
        self.running = True
        
        # Start processing thread
        process_thread = threading.Thread(target=self.processing_worker, daemon=True)
        process_thread.start()
        
        logger.info("Starting audio stream...")
        logger.info("Note: Only processed audio will be played (passthrough disabled)")
        logger.info("Audio will be silent until buffer is ready")
        
        logger.info("Press Ctrl+C to stop")
        
        try:
            with sd.Stream(
                device=(self.input_device, self.output_device),
                samplerate=self.sample_rate,
                channels=2,
                dtype=np.float32,
                blocksize=8192,
                latency='high',
                callback=self.audio_callback,
            ):
                while self.running:
                    sd.sleep(100)
        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            self.running = False
            self.processing_queue.put(None)
            process_thread.join(timeout=2)


def list_devices():
    """Print available audio devices."""
    print("\nAvailable audio devices:\n")
    devices = sd.query_devices()
    print(devices)
    print("\nUse the device index numbers with --input and --output")
    
    # Auto-detect USB audio (UCA202, etc.)
    usb_device = find_usb_audio_device()
    if usb_device is not None:
        print(f"\nAuto-detected USB audio: Device {usb_device} - {devices[usb_device]['name']}")
        print(f"  → Use: python3 vinyl_stripper.py --input {usb_device} --output {usb_device}")
    
    # Auto-detect Sound Burger setup
    input_dev, output_dev = find_sound_burger_setup()
    if input_dev is not None or output_dev is not None:
        print("\nAuto-detection (Sound Burger setup):")
        if input_dev is not None:
            print(f"  → Input:  Device {input_dev} - {devices[input_dev]['name']}")
        if output_dev is not None:
            print(f"  → Output: Device {output_dev} - {devices[output_dev]['name']}")


def find_universal_audio_device():
    """Auto-detect Universal Audio Thunderbolt interface."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if 'apollo' in name or 'universal audio' in name or 'ua ' in name:
            return i
    return None


def find_soundid_reference_device():
    """Auto-detect SoundID Reference output device."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if 'soundid' in name or 'sid' in name:
            if device['max_output_channels'] > 0:
                return i
    return None


def find_usb_audio_device():
    """Auto-detect USB audio interface (UCA202, etc.)."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        name = device['name'].lower()
        # Look for USB audio devices (UCA202 shows as "USB Audio Device" in ALSA)
        if ('usb audio' in name or 'uca202' in name or 'behringer' in name or 
            'scarlett' in name or 'focusrite' in name):
            # UCA202 has 2 in, 2 out
            if device['max_input_channels'] >= 2 and device['max_output_channels'] >= 2:
                return i
    return None


def resolve_device(device_arg):
    """
    Resolve device argument to device index.
    Accepts: integer (device index), 'ua' (Universal Audio), 'sid' (SoundID), 'usb' (USB audio), or None (auto-detect).
    """
    if device_arg is None:
        return None
    
    # If it's already an integer, return it
    if isinstance(device_arg, int):
        return device_arg
    
    # Try to parse as integer
    try:
        return int(device_arg)
    except (ValueError, TypeError):
        pass
    
    # Check for nicknames
    device_str = str(device_arg).lower()
    
    if device_str == 'ua' or device_str == 'apollo':
        dev = find_universal_audio_device()
        if dev is None:
            raise ValueError(f"Universal Audio device not found. Use --list-devices to see available devices.")
        return dev
    
    if device_str == 'sid' or device_str == 'soundid':
        dev = find_soundid_reference_device()
        if dev is None:
            raise ValueError(f"SoundID Reference device not found. Use --list-devices to see available devices.")
        return dev
    
    if device_str == 'usb':
        dev = find_usb_audio_device()
        if dev is None:
            raise ValueError(f"USB audio device not found. Use --list-devices to see available devices.")
        return dev
    
    # Unknown nickname
    raise ValueError(f"Unknown device nickname: '{device_arg}'. Use integer index, 'ua', 'sid', 'usb', or --list-devices")




def find_sound_burger_setup():
    """
    Auto-detect Sound Burger → Raspberry Pi → Headphones setup.
    Uses GPIO-based ADC for input and innomaker HiFi DAC HAT (or Pi headphone jack) for output.
    Returns (input_device, output_device) or (None, None) if not found.
    """
    devices = sd.query_devices()
    input_device = None
    output_device = None
    
    # Look for GPIO-based ADC boards (HiFiBerry ADC+, IQaudio DigiAMP+, etc.)
    for i, device in enumerate(devices):
        name = device['name'].lower()
        # GPIO/I2S audio input boards
        if device['max_input_channels'] >= 2:
            if 'hifiberry' in name and 'adc' in name:
                input_device = i
                break
            elif 'adc' in name or 'iqaudio' in name or 'i2s' in name:
                input_device = i
                break
    
    # Fallback: Look for USB audio dongle or other input devices
    if input_device is None:
        for i, device in enumerate(devices):
            if device['max_input_channels'] >= 2:
                # Prefer devices that aren't just microphones
                name = device['name'].lower()
                if 'usb' in name or ('line' in name and 'mic' not in name):
                    input_device = i
                    break
    
    # Find innomaker HiFi DAC HAT (shows up as hifiberry-dac) - HIGHEST PRIORITY
    for i, device in enumerate(devices):
        name = device['name'].lower()
        if device['max_output_channels'] >= 2:
            # Prefer HiFi DAC HAT (innomaker, HiFiBerry DAC, etc.)
            if 'hifiberry' in name and 'dac' in name and 'adc' not in name:
                output_device = i
                break
    
    # Fallback: Raspberry Pi built-in headphone jack
    if output_device is None:
        for i, device in enumerate(devices):
            name = device['name'].lower()
            if device['max_output_channels'] >= 2:
                if 'bcm2835' in name or 'headphone' in name:
                    output_device = i
                    break
    
    # Final fallback: Any stereo output device
    if output_device is None:
        for i, device in enumerate(devices):
            if device['max_output_channels'] >= 2:
                output_device = i
                break
    
    return input_device, output_device


def main():
    parser = argparse.ArgumentParser(
        description="Real-time vinyl stem stripper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 vinyl_stripper.py --list-devices
    python3 vinyl_stripper.py --input 1 --output 1 --remove vocals
    python3 vinyl_stripper.py --input ua --output sid --remove vocals
    python3 vinyl_stripper.py --model htdemucs --remove vocals drums
    python3 vinyl_stripper.py --remove all
    
Device nicknames: ua (Universal Audio), sid (SoundID Reference), usb (USB audio)
Available models: htdemucs, htdemucs_ft (default), htdemucs_6s
Available stems: vocals, drums, bass, other
        """
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("-i", "--input", type=str, help="Input device index or nickname (ua, sid, usb)")
    parser.add_argument("-o", "--output", type=str, help="Output device index or nickname (ua, sid, usb)")
    parser.add_argument(
        "--remove",
        nargs="+",
        choices=["vocals", "drums", "bass", "other", "all"],
        default=["vocals"],
        help="Which stems to remove (can specify multiple, e.g., --remove vocals drums). Options: vocals, drums, bass, other, all"
    )
    parser.add_argument(
        "--chunk",
        type=float,
        default=2.5,
        help="Chunk duration in seconds (default: 2.5, balanced for efficiency and latency)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="htdemucs_ft",
        help="Demucs model name (default: htdemucs_ft). Available: htdemucs, htdemucs_ft, htdemucs_6s"
    )
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_devices()
        return
    
    # Resolve device nicknames to indices
    try:
        if args.input is not None:
            args.input = resolve_device(args.input)
        if args.output is not None:
            args.output = resolve_device(args.output)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Auto-detect devices if not specified
    if args.input is None or args.output is None:
        input_dev = None
        output_dev = None
        
        # Try USB audio device (UCA202, etc.) first
        usb_device = find_usb_audio_device()
        if usb_device is not None:
            if devices[usb_device]['max_input_channels'] >= 2 and devices[usb_device]['max_output_channels'] >= 2:
                input_dev = usb_device
                output_dev = usb_device
                print(f"✓ Auto-detected USB audio interface:")
                print(f"  Input/Output:  Device {usb_device} - {devices[usb_device]['name']}")
        
        # Fall back to Sound Burger setup if USB not found
        if input_dev is None or output_dev is None:
            input_dev, output_dev = find_sound_burger_setup()
            if input_dev is not None and output_dev is not None:
                print(f"✓ Auto-detected Sound Burger setup:")
                print(f"  Input:  Device {input_dev} - {devices[input_dev]['name']}")
                print(f"  Output: Device {output_dev} - {devices[output_dev]['name']}")
        
        # Set detected devices
        if input_dev is not None and output_dev is not None:
            args.input = input_dev
            args.output = output_dev
        else:
            print("Error: --input and --output device indices are required")
            print("Run with --list-devices to see available devices")
            print("\nFor USB audio (UCA202, etc.):")
            print("  → Connect USB audio interface")
            print("\nFor Sound Burger setup:")
            print("  → Install HiFiBerry ADC+ (GPIO) or USB audio dongle for input")
            print("  → Install innomaker HiFi DAC HAT or use Pi headphone jack for output")
            sys.exit(1)
    
    # Parse stems to remove
    if isinstance(args.remove, str):
        args.remove = [args.remove]
    
    if "all" in args.remove:
        remove_stems = ["vocals", "drums", "bass", "other"]
    else:
        remove_stems = args.remove
    
    # Create and run
    try:
        stripper = VinylStripper(
            input_device=args.input,
            output_device=args.output,
            remove_stems=remove_stems,
            chunk_duration=args.chunk,
            model_name=args.model,
        )
        stripper.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except MemoryError as e:
        logger.error(f"Out of memory error: {e}")
        log_memory_status("final OOM")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        # Ensure logs are flushed
        logging.shutdown()


if __name__ == "__main__":
    main()
