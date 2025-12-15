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
        input_device: int | str,  # Can be device index (int) or ALSA string (str)
        output_device: int | str,  # Can be device index (int) or ALSA string (str)
        remove_stems: list[str],
        chunk_duration: float = 0.20,  # seconds per chunk (very small for fastest processing)
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
        
        self.input_buffer = deque(maxlen=self.chunk_samples * 32)  # Increased buffer size
        self.output_queue = queue.Queue(maxsize=200)  # Larger output queue to prevent blocking
        self.output_buffer = np.zeros((0, 2), dtype=np.float32)
        self.passthrough_buffer = deque(maxlen=int(sample_rate * 3))
        self.processing_queue = queue.Queue(maxsize=100)  # Larger processing queue
        self.running = False
        self.buffer_filling = True
        self.use_passthrough = True  # Start with passthrough for immediate audio
        self.last_buffer_log = 0
        self.buffer_log_interval = 5.0  # Log buffer status every 5 seconds
        self.min_buffer_before_switch = int(sample_rate * 0.2)  # Start playing with just 0.2s buffer (very low latency)
        self.crossfade_samples = int(sample_rate * 0.01)  # 10ms crossfade for smooth transition
        self.switching_to_processed = False  # Track if we're in transition
        
        # Load model with device-specific optimization
        # Reference torch before any local imports to avoid UnboundLocalError
        # Priority: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        model_name = self.model_name if self.model_name else "htdemucs_ft"
        
        logger.info(f"Loading model: {model_name} (device: {self.device})...")
        
        # Check memory before loading
        available_mem, mem_percent = log_memory_status("before model load")
        if available_mem and available_mem < 1.0 * (1024**3):  # Less than 1GB available
            logger.warning(f"Low memory available ({available_mem / (1024**3):.2f}GB). Model loading may fail.")
            logger.warning("Consider using lighter model (htdemucs) or increasing swap space.")
        
        try:
            # Set environment variables for better performance
            if self.device == "cpu":
                # Use more threads for better parallel processing on CPU
                import multiprocessing
                num_threads = min(multiprocessing.cpu_count(), 4)  # Use up to 4 threads
                os.environ['OMP_NUM_THREADS'] = str(num_threads)
                logger.info(f"Using {num_threads} CPU threads for processing")
            elif self.device == "cuda":
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                logger.info("Using CUDA GPU acceleration")
            elif self.device == "mps":
                logger.info("Using Apple Silicon GPU (Metal Performance Shaders)")
            
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
            # torch.compile not supported on MPS yet, only CUDA
            if hasattr(torch, 'compile') and self.device == 'cuda':
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
        
        # Normalize - compute mean and std across all samples and channels
        ref_mean = audio_tensor.mean()
        ref_std = audio_tensor.std()
        # Avoid division by zero
        if ref_std < 1e-8:
            ref_std = torch.tensor(1.0, device=self.device)
        audio_tensor = (audio_tensor - ref_mean) / ref_std
        
        with torch.no_grad():
            # Apply model - returns (batch, stems, channels, samples)
            # Use simpler approach for small chunks (like reference)
            sources = apply_model(
                self.model, 
                audio_tensor, 
                device=self.device,
                segment=None,  # Process entire chunk at once (faster for small chunks)
                split=False,  # Disable splitting for speed (we're already chunking)
                shifts=1,  # Single shift for speed (default is 1)
                overlap=0.0,  # No overlap needed since we handle it in chunking
            )
        
        # sources shape: (1, 4, 2, samples)
        # stems are: drums, bass, other, vocals
        
        # Verify sources shape
        if sources.shape[1] != 4:
            logger.error(f"Unexpected sources shape: {sources.shape}, expected (batch, 4, channels, samples)")
        
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
        
        # Debug: Log occasionally to verify processing
        import time as time_module
        if not hasattr(self, '_last_process_log') or (time_module.time() - self._last_process_log) > 5.0:
            logger.info(f"✓ Processing active: Removing {stems_removed}, Keeping {stems_kept}")
            input_energy = audio_tensor.abs().mean().item()
            output_energy = output.abs().mean().item()
            reduction = ((input_energy - output_energy) / input_energy * 100) if input_energy > 0 else 0
            logger.info(f"  Energy: Input={input_energy:.4f}, Output={output_energy:.4f} ({reduction:.1f}% reduction)")
            self._last_process_log = time_module.time()
        
        # Denormalize
        output = output * ref_std + ref_mean
        
        # Convert back to numpy (samples, channels)
        result = output.squeeze(0).T.cpu().numpy()
        
        return result

    def processing_worker(self):
        """Background thread that processes audio chunks."""
        while self.running:
            try:
                chunk = self.processing_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            if chunk is None:
                break
                
            try:
                processed = self.process_chunk(chunk)
                self.output_queue.put(processed)
            except Exception as e:
                logger.error(f"Processing error: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")

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
        
        # Store raw audio for passthrough fallback (enabled for immediate audio)
        self.passthrough_buffer.extend(indata.copy())
        
        # Resample input if needed
        input_audio = indata.copy()
        if hasattr(self, 'needs_input_resampling') and self.needs_input_resampling:
            if self.resample_func:
                # Resample from input_sr to output_sr
                target_samples = int(len(input_audio) * self.output_sample_rate / self.input_sample_rate)
                input_audio = self.resample_func(input_audio, target_samples, axis=0)
            else:
                # Simple linear interpolation fallback
                from numpy import linspace, interp
                old_indices = np.arange(len(input_audio))
                new_length = int(len(input_audio) * self.output_sample_rate / self.input_sample_rate)
                new_indices = linspace(0, len(input_audio) - 1, new_length)
                input_audio = np.array([interp(new_indices, old_indices, input_audio[:, ch]) for ch in range(input_audio.shape[1])]).T
        
        # Add input to buffer
        self.input_buffer.extend(input_audio)
        
        # If we have enough samples, queue a chunk for processing
        if len(self.input_buffer) >= self.chunk_samples:
            chunk = np.array(list(self.input_buffer)[:self.chunk_samples])
            
            # Remove hop_samples from the front (keep overlap for next chunk)
            for _ in range(self.hop_samples):
                if self.input_buffer:
                    self.input_buffer.popleft()
            
            # Queue for processing (non-blocking, drop if full)
            # Keep processing pipeline busy - queue aggressively
            try:
                self.processing_queue.put_nowait(chunk)
            except queue.Full:
                # If queue is full, drop oldest chunk and add new one to prevent lag
                try:
                    _ = self.processing_queue.get_nowait()  # Remove oldest
                    self.processing_queue.put_nowait(chunk)  # Add new
                except:
                    pass  # Skip if still can't queue
        
        # Try to get processed audio for output
        # Keep filling buffer aggressively to prevent skipping
        target_buffer_size = int(self.sample_rate * 1.0)  # Target 1 second buffer (reduced for lower latency)
        
        # Always try to fill buffer if we have processed chunks available
        # Process multiple chunks per callback to keep buffer full
        chunks_added = 0
        max_chunks_per_callback = 20  # Process up to 20 chunks per callback
        
        while chunks_added < max_chunks_per_callback:
            try:
                processed = self.output_queue.get_nowait()
                
                # Handle overlap with proper crossfading to avoid clicks/pops
                if self.output_buffer.shape[0] > 0:
                    # We have existing buffer - need to handle overlap
                    if self.output_buffer.shape[0] >= self.overlap_samples:
                        # Crossfade the overlap region
                        overlap_region_old = self.output_buffer[-self.overlap_samples:]
                        overlap_region_new = processed[:self.overlap_samples]
                        
                        # Create crossfade window (fade out old, fade in new)
                        fade_out = np.linspace(1.0, 0.0, self.overlap_samples).reshape(-1, 1)
                        fade_in = np.linspace(0.0, 1.0, self.overlap_samples).reshape(-1, 1)
                        
                        # Crossfade the overlap
                        crossfaded = overlap_region_old * fade_out + overlap_region_new * fade_in
                        
                        # Replace overlap region with crossfaded version
                        self.output_buffer[-self.overlap_samples:] = crossfaded
                        
                        # Add the rest of the new chunk (after overlap)
                        if processed.shape[0] > self.overlap_samples:
                            new_portion = processed[self.overlap_samples:]
                            self.output_buffer = np.vstack([self.output_buffer, new_portion])
                    else:
                        # Not enough buffer yet - just append (will handle overlap later)
                        self.output_buffer = np.vstack([self.output_buffer, processed])
                else:
                    # First chunk - add entire chunk
                    self.output_buffer = processed
                
                chunks_added += 1
                if self.buffer_filling and self.output_buffer.shape[0] >= target_buffer_size:
                    self.buffer_filling = False
                    logger.info(f"✓ Buffer ready ({self.output_buffer.shape[0] / self.sample_rate:.1f}s)")
            except queue.Empty:
                break
        
        # Check if we should switch from passthrough to processed audio
        current_buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
        min_buffer_seconds = self.min_buffer_before_switch / self.sample_rate
        
        # Switch to processed audio when buffer is ready (with crossfade)
        if self.use_passthrough and self.output_buffer.shape[0] >= self.min_buffer_before_switch:
            if not self.switching_to_processed:
                self.switching_to_processed = True
                self.buffer_filling = False
                logger.info(f"✓ Starting transition to processed audio ({current_buffer_seconds:.1f}s buffer ready)")
        elif self.buffer_filling and self.output_buffer.shape[0] >= self.min_buffer_before_switch:
            self.buffer_filling = False
        
        # Note: Crossfade completion is handled in the crossfade output block
        
        # Output logic: prefer processed audio, fall back to passthrough
        buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
        
        # Handle crossfade transition
        if self.switching_to_processed and self.output_buffer.shape[0] >= frames and len(self.passthrough_buffer) >= frames:
            # Crossfade between passthrough and processed over the crossfade duration
            passthrough_data = np.array(list(self.passthrough_buffer)[:frames])
            processed_data = self.output_buffer[:frames]
            
            # Create crossfade window (fade out passthrough, fade in processed)
            crossfade_len = min(frames, self.crossfade_samples)
            fade_out = np.linspace(1.0, 0.0, crossfade_len).reshape(-1, 1)
            fade_in = np.linspace(0.0, 1.0, crossfade_len).reshape(-1, 1)
            
            # Apply crossfade to the beginning of the frame
            outdata[:crossfade_len] = passthrough_data[:crossfade_len] * fade_out + processed_data[:crossfade_len] * fade_in
            
            # Rest of frame uses processed audio
            if frames > crossfade_len:
                outdata[crossfade_len:] = processed_data[crossfade_len:]
            
            # Update buffers
            self.output_buffer = self.output_buffer[frames:]
            for _ in range(frames):
                if self.passthrough_buffer:
                    self.passthrough_buffer.popleft()
            
            # Track crossfade progress
            if not hasattr(self, '_crossfade_samples_used'):
                self._crossfade_samples_used = 0
            self._crossfade_samples_used += frames
            
            # Complete transition after crossfade duration
            if self._crossfade_samples_used >= self.crossfade_samples:
                self.use_passthrough = False
                self.switching_to_processed = False
                self._crossfade_samples_used = 0
                logger.info("✓ Crossfade complete - fully using processed audio")
        elif not self.use_passthrough and self.output_buffer.shape[0] >= frames:
            # Use processed audio - ensure continuous output
            outdata[:] = self.output_buffer[:frames]
            self.output_buffer = self.output_buffer[frames:]
            
            # Switch back to passthrough if buffer runs low
            new_buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
            if new_buffer_seconds < 0.1 and not self.buffer_filling:
                self.use_passthrough = True
                if not hasattr(self, '_last_low_buffer_log') or (time_module.time() - self._last_low_buffer_log) > 1.0:
                    logger.warning(f"⚠ Buffer low - switching to passthrough ({new_buffer_seconds:.2f}s remaining)")
                    self._last_low_buffer_log = time_module.time()
        elif self.use_passthrough and len(self.passthrough_buffer) >= frames:
            # Passthrough mode - output raw audio immediately
            passthrough_data = np.array(list(self.passthrough_buffer)[:frames])
            outdata[:] = passthrough_data
            # Remove used samples from passthrough buffer
            for _ in range(frames):
                if self.passthrough_buffer:
                    self.passthrough_buffer.popleft()
        elif not self.use_passthrough and self.output_buffer.shape[0] > 0:
            # Partial processed buffer - use what we have, fill rest with passthrough
            available = min(self.output_buffer.shape[0], frames)
            outdata[:available] = self.output_buffer[:available]
            self.output_buffer = self.output_buffer[available:]
            
            # Fill rest with passthrough if available
            if available < frames and len(self.passthrough_buffer) >= (frames - available):
                passthrough_needed = frames - available
                passthrough_data = np.array(list(self.passthrough_buffer)[:passthrough_needed])
                outdata[available:] = passthrough_data
                for _ in range(passthrough_needed):
                    if self.passthrough_buffer:
                        self.passthrough_buffer.popleft()
            elif available < frames:
                # Pad with last sample
                last_sample = outdata[available-1:available] if available > 0 else np.zeros((1, 2), dtype=np.float32)
                outdata[available:] = np.tile(last_sample, (frames - available, 1))
        elif self.use_passthrough and len(self.passthrough_buffer) > 0:
            # Partial passthrough buffer - use what we have
            available = min(len(self.passthrough_buffer), frames)
            passthrough_data = np.array(list(self.passthrough_buffer)[:available])
            outdata[:available] = passthrough_data
            for _ in range(available):
                if self.passthrough_buffer:
                    self.passthrough_buffer.popleft()
            # Pad with last sample
            if available < frames:
                last_sample = passthrough_data[-1:] if available > 0 else np.zeros((1, 2), dtype=np.float32)
                outdata[available:] = np.tile(last_sample, (frames - available, 1))
        else:
            # No audio available - output silence (shouldn't happen with passthrough)
            outdata[:] = 0

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
            
            # Process chunks and fill buffer with proper overlap handling
            processed_count = 0
            for chunk in chunks_to_process:
                try:
                    processed = self.process_chunk(chunk)
                    
                    # Handle overlap with proper crossfading (same as main callback)
                    if self.output_buffer.shape[0] > 0:
                        if self.output_buffer.shape[0] >= self.overlap_samples:
                            # Crossfade the overlap region
                            overlap_region_old = self.output_buffer[-self.overlap_samples:]
                            overlap_region_new = processed[:self.overlap_samples]
                            
                            fade_out = np.linspace(1.0, 0.0, self.overlap_samples).reshape(-1, 1)
                            fade_in = np.linspace(0.0, 1.0, self.overlap_samples).reshape(-1, 1)
                            
                            crossfaded = overlap_region_old * fade_out + overlap_region_new * fade_in
                            self.output_buffer[-self.overlap_samples:] = crossfaded
                            
                            if processed.shape[0] > self.overlap_samples:
                                new_portion = processed[self.overlap_samples:]
                                self.output_buffer = np.vstack([self.output_buffer, new_portion])
                        else:
                            self.output_buffer = np.vstack([self.output_buffer, processed])
                    else:
                        # First chunk - add entire chunk
                        self.output_buffer = processed
                    
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
        
        # Skip pre-fill for low latency - start immediately with passthrough
        # Audio will play immediately via passthrough, then switch to processed when ready
        logger.info("Starting audio stream immediately (passthrough mode)...")
        logger.info("Will switch to processed audio when buffer is ready (~0.2s)")
        logger.info("Press Ctrl+C to stop")
        
        try:
            # Auto-detect sample rates for input and output devices
            input_sr = self.sample_rate
            output_sr = self.sample_rate
            try:
                if isinstance(self.input_device, int):
                    input_dev_info = sd.query_devices(self.input_device)
                    if 'default_samplerate' in input_dev_info and input_dev_info['default_samplerate']:
                        input_sr = int(input_dev_info['default_samplerate'])
                if isinstance(self.output_device, int):
                    output_dev_info = sd.query_devices(self.output_device)
                    if 'default_samplerate' in output_dev_info and output_dev_info['default_samplerate']:
                        output_sr = int(output_dev_info['default_samplerate'])
                
                # PortAudio requires same sample rate for duplex streams
                # Use output device's rate and resample input if needed
                actual_sample_rate = output_sr
                self.input_sample_rate = input_sr
                self.output_sample_rate = output_sr
                self.needs_input_resampling = (input_sr != output_sr)
                
                if self.needs_input_resampling:
                    logger.info(f"Sample rate mismatch: input={input_sr} Hz, output={output_sr} Hz")
                    logger.info(f"Using {actual_sample_rate} Hz for stream, will resample input from {input_sr} Hz")
                    # Import resampling function
                    try:
                        from scipy import signal
                        self.resample_func = signal.resample
                    except ImportError:
                        logger.warning("scipy not available, using numpy-based resampling")
                        self.resample_func = None
                else:
                    logger.info(f"Using sample rate: {actual_sample_rate} Hz")
                
                # Update sample rate for processing
                self.sample_rate = actual_sample_rate
                # Recalculate chunk sizes
                self.chunk_samples = int(self.chunk_duration * actual_sample_rate)
                self.overlap_samples = int(self.chunk_samples * self.overlap)
                self.hop_samples = self.chunk_samples - self.overlap_samples
            except Exception as e:
                logger.debug(f"Could not auto-detect sample rate: {e}")
                self.needs_resampling = False
                self.output_sample_rate = self.sample_rate
            
            # Stream configuration
            stream_kwargs = {
                'samplerate': actual_sample_rate,
                'channels': 2,
                'dtype': np.float32,
                'blocksize': 8192,
                'latency': 'high',
                'callback': self.audio_callback,
            }
            
            # Find ALSA host API for better device handling
            try:
                for i in range(sd.query_hostapis()):
                    api = sd.query_hostapis(i)
                    if 'alsa' in api['name'].lower():
                        stream_kwargs['hostapi'] = i
                        logger.info(f"Using ALSA host API: {api['name']}")
                        break
            except:
                pass
            
            stream_kwargs['device'] = (self.input_device, self.output_device)
            
            with sd.Stream(**stream_kwargs):
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
    """Auto-detect USB audio interface (UCA202/UFO202, etc.)."""
    # Try to enumerate devices first
    try:
        devices = sd.query_devices()
    except:
        devices = []
    
    # If devices found via PortAudio enumeration, check them
    if len(devices) > 0:
        for i, device in enumerate(devices):
            name = device['name'].lower()
            # Look for USB audio devices (UCA202/UFO202 shows as "USB Audio CODEC" or "USB Audio Device" in ALSA)
            has_usb = 'usb' in name
            has_codec = 'codec' in name
            has_other = any(x in name for x in ['uca202', 'ufo202', 'behringer', 'scarlett', 'focusrite'])
            
            if (has_usb and has_codec) or has_usb or has_other:
                # UCA202/UFO202 has 2 in, 2 out
                if device['max_input_channels'] >= 2 and device['max_output_channels'] >= 2:
                    return i
    
    # Fallback: PortAudio enumeration failed, try querying by index
    # Check common USB audio device indices (0-10)
    for i in range(11):
        try:
            device_info = sd.query_devices(i)
            name = device_info['name'].lower()
            # Look for USB audio devices - check for "usb" and "codec" separately
            has_usb = 'usb' in name
            has_codec = 'codec' in name
            has_other = any(x in name for x in ['uca202', 'ufo202', 'behringer', 'scarlett', 'focusrite'])
            
            if (has_usb and has_codec) or has_usb or has_other:
                if device_info['max_input_channels'] >= 2 and device_info['max_output_channels'] >= 2:
                    return i
        except:
            continue
    
    # Final fallback: If PortAudio completely fails, use ALSA device string directly
    # Parse arecord output to find USB Audio CODEC card number
    try:
        import subprocess
        import re
        # Check if USB audio device exists via arecord
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True, timeout=2)
        if 'USB Audio' in result.stdout or 'CODEC' in result.stdout:
            # Parse card number from output: "card 2: CODEC [USB Audio CODEC]"
            match = re.search(r'card (\d+):.*USB Audio.*CODEC', result.stdout)
            if match:
                card_num = match.group(1)
                # When PortAudio fails, we can't use ALSA strings directly with sounddevice
                # Instead, return a special marker that will trigger default device usage
                # The card number is stored for potential ALSA configuration
                return f"_alsa_card_{card_num}"
    except Exception as e:
        logger.debug(f"ALSA fallback failed: {e}")
    
    return None


def resolve_device(device_arg):
    """
    Resolve device argument to device index or ALSA device string.
    Accepts: integer (device index), ALSA string (e.g., 'hw:2,0'), 'ua' (Universal Audio), 'sid' (SoundID), 'usb' (USB audio), or None (auto-detect).
    Returns: int (device index) or str (ALSA device string)
    """
    if device_arg is None:
        return None
    
    # If it's already an integer, return it
    if isinstance(device_arg, int):
        return device_arg
    
    # If it's a string, check if it's an ALSA device string or numeric
    if isinstance(device_arg, str):
        # Check if it's an ALSA device string (starts with 'hw:' or 'plughw:')
        if device_arg.startswith('hw:') or device_arg.startswith('plughw:'):
            return device_arg
        # Try to parse as integer if it's a numeric string
        try:
            return int(device_arg)
        except (ValueError, TypeError):
            pass  # Not numeric, continue to nickname checking
    
    # Try to parse as integer
    try:
        return int(device_arg)
    except (ValueError, TypeError):
        pass
    
    # Check if it's an ALSA device string
    device_str = str(device_arg)
    if device_str.startswith('hw:') or device_str.startswith('plughw:'):
        return device_str
    
    # Check for nicknames
    device_str = device_str.lower()
    
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
    
    if device_str == 'ufo202' or device_str == 'usb':
        dev = find_usb_audio_device()
        if dev is None:
            raise ValueError(f"USB audio device (UFO202) not found. Use --list-devices to see available devices.")
        return dev
    
    # Unknown nickname
    raise ValueError(f"Unknown device nickname: '{device_arg}'. Use integer index, ALSA string (hw:X,Y), 'ua', 'sid', 'usb', or --list-devices")




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
    python3 vinyl_stripper.py --input ufo202 --output ufo202 --remove vocals
    python3 vinyl_stripper.py --model htdemucs --remove vocals drums
    python3 vinyl_stripper.py --remove all
    
Device nicknames: ua (Universal Audio), sid (SoundID Reference), ufo202 (UFO202/USB audio), usb (USB audio - alias for ufo202)
Available models: htdemucs, htdemucs_ft (default), htdemucs_6s
Available stems: vocals, drums, bass, other
        """
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("-i", "--input", type=str, help="Input device index or nickname (ua, sid, ufo202, usb)")
    parser.add_argument("-o", "--output", type=str, help="Output device index or nickname (ua, sid, ufo202, usb)")
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
        default=0.2,
        help="Chunk duration in seconds (default: 0.2, optimized for real-time processing)"
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
        device_detected = False  # Track if we successfully detected a device
        
        # Get devices list for checking capabilities
        devices = sd.query_devices()
        
        # Try USB audio device (UCA202/UFO202, etc.) first
        usb_device = find_usb_audio_device()
        
        # Note: find_usb_audio_device() now returns ALSA string if PortAudio fails
        
        if usb_device is not None:
            # Check if usb_device is a special ALSA marker
            if isinstance(usb_device, str) and usb_device.startswith("_alsa_card_"):
                # PortAudio enumeration failed - use default device
                # ALSA will use the default card, which should be the USB device if configured
                input_dev = None  # Use default device
                output_dev = None  # Use default device
                device_detected = True  # We detected the device, even if using default
                card_num = usb_device.replace("_alsa_card_", "")
                print(f"✓ Auto-detected USB audio interface (UFO202/UCA202):")
                print(f"  ALSA card: {card_num} (hw:{card_num},0)")
                print(f"  Note: Using default audio device (PortAudio enumeration unavailable)")
                print(f"  If this doesn't work, configure ALSA defaults:")
                print(f"    sudo nano /etc/asound.conf")
                print(f"    Add: defaults.pcm.card {card_num}")
                print(f"         defaults.ctl.card {card_num}")
            elif isinstance(usb_device, str):
                # ALSA device string - use directly (shouldn't happen with current code)
                input_dev = usb_device
                output_dev = usb_device
                print(f"✓ Auto-detected USB audio interface (UFO202/UCA202):")
                print(f"  Input/Output:  {usb_device} (ALSA device string)")
                print(f"  Note: Using ALSA device directly (PortAudio enumeration unavailable)")
            else:
                # Integer device index - try to get info
                try:
                    if len(devices) > 0 and usb_device < len(devices):
                        usb_device_info = devices[usb_device]
                    else:
                        usb_device_info = sd.query_devices(usb_device)
                    
                    if usb_device_info['max_input_channels'] >= 2 and usb_device_info['max_output_channels'] >= 2:
                        input_dev = usb_device
                        output_dev = usb_device
                        device_detected = True
                        print(f"✓ Auto-detected USB audio interface (UFO202/UCA202):")
                        print(f"  Input/Output:  Device {usb_device} - {usb_device_info.get('name', 'USB Audio CODEC')}")
                        print(f"  Channels: {usb_device_info['max_input_channels']} in, {usb_device_info['max_output_channels']} out")
                except Exception as e:
                    # If we can't query device info but USB exists, use it anyway
                    input_dev = usb_device
                    output_dev = usb_device
                    device_detected = True
                    print(f"✓ Auto-detected USB audio interface (UFO202/UCA202):")
                    print(f"  Input/Output:  Device {usb_device} (USB Audio CODEC)")
                    print(f"  Note: Using device {usb_device} for both input and output")
        
        # Fall back to Sound Burger setup if USB not found
        if not device_detected and (input_dev is None or output_dev is None):
            input_dev, output_dev = find_sound_burger_setup()
            if input_dev is not None and output_dev is not None:
                print(f"✓ Auto-detected Sound Burger setup:")
                print(f"  Input:  Device {input_dev} - {devices[input_dev]['name']}")
                print(f"  Output: Device {output_dev} - {devices[output_dev]['name']}")
        
        # Set detected devices (None is valid - uses default device)
        args.input = input_dev
        args.output = output_dev
        
        # Only error if we couldn't detect anything and user didn't specify
        if not device_detected:
            print("Error: Could not auto-detect audio devices")
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
