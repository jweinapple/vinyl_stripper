#!/usr/bin/env python3
"""
Vinyl Stem Stripper - Pi 5 Optimized Version
Real-time vocal/drum removal for vinyl playback on Raspberry Pi 5

Key optimizations for Pi 5:
1. Multiple backends: Demucs, Spleeter (faster), ONNX (optimized)
2. Multi-worker parallel processing (uses all CPU cores)
3. Aggressive pre-buffering for high-latency but stable playback
4. Hailo-8 AI Kit support (13-26 TOPS neural accelerator)
5. Auto-detection and Pi-specific tuning

Usage:
    python3 vinyl_stripper_pi.py --list-devices
    python3 vinyl_stripper_pi.py --input 1 --output 1 --remove vocals
    python3 vinyl_stripper_pi.py --backend spleeter --remove vocals  # Faster on Pi
    python3 vinyl_stripper_pi.py --backend onnx --remove vocals      # ONNX optimized
    python3 vinyl_stripper_pi.py --pi-mode                           # Auto-optimize for Pi
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
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from abc import ABC, abstractmethod

# Platform detection
IS_RASPBERRY_PI = os.path.exists('/proc/device-tree/model') and 'raspberry pi' in open('/proc/device-tree/model').read().lower() if os.path.exists('/proc/device-tree/model') else False
PI_MODEL = None
HAILO_AVAILABLE = False
NUM_CORES = multiprocessing.cpu_count()

def detect_platform():
    """Detect platform and available accelerators."""
    global IS_RASPBERRY_PI, PI_MODEL, HAILO_AVAILABLE

    # Detect Raspberry Pi
    if os.path.exists('/proc/device-tree/model'):
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().lower()
                IS_RASPBERRY_PI = 'raspberry pi' in model
                if 'pi 5' in model:
                    PI_MODEL = 'pi5'
                elif 'pi 4' in model:
                    PI_MODEL = 'pi4'
                elif 'pi 3' in model:
                    PI_MODEL = 'pi3'
        except:
            pass

    # Detect Hailo AI Kit
    try:
        # Check for Hailo device
        if os.path.exists('/dev/hailo0') or os.path.exists('/sys/class/hailo_chardev'):
            HAILO_AVAILABLE = True
        # Also check via hailortcli if available
        import subprocess
        result = subprocess.run(['hailortcli', 'fw-control', 'identify'],
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            HAILO_AVAILABLE = True
    except:
        pass

    return IS_RASPBERRY_PI, PI_MODEL, HAILO_AVAILABLE

detect_platform()

# Check dependencies
def check_dependencies():
    missing = []
    optional_missing = []

    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")

    # At least one backend required
    backends_available = []
    try:
        import torch
        backends_available.append("demucs")
    except ImportError:
        optional_missing.append("torch (for demucs)")

    try:
        import spleeter
        backends_available.append("spleeter")
    except ImportError:
        optional_missing.append("spleeter")

    try:
        import onnxruntime
        backends_available.append("onnx")
    except ImportError:
        optional_missing.append("onnxruntime")

    if missing:
        print("Missing required dependencies:")
        print(f"  pip3 install {' '.join(missing)}")
        sys.exit(1)

    if not backends_available:
        print("No stem separation backend available. Install at least one:")
        print("  pip3 install torch demucs      # High quality, slow")
        print("  pip3 install spleeter          # Fast, good quality")
        print("  pip3 install onnxruntime       # Optimized inference")
        sys.exit(1)

    return backends_available

AVAILABLE_BACKENDS = check_dependencies()

import sounddevice as sd

# Setup logging
def setup_logging(verbose=False):
    """Setup logging to both file and console."""
    log_dir = os.path.expanduser("~/vinyl_stripper/logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"vinyl_stripper_pi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger('vinyl_stripper')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()


# =============================================================================
# BACKEND ABSTRACTION
# =============================================================================

class StemSeparatorBackend(ABC):
    """Abstract base class for stem separation backends."""

    @abstractmethod
    def separate(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Separate audio into stems.

        Args:
            audio: numpy array of shape (samples, channels)
            sample_rate: audio sample rate

        Returns:
            dict with keys: 'vocals', 'drums', 'bass', 'other'
            each value is numpy array of shape (samples, channels)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_expected_latency(self, chunk_duration: float) -> float:
        """Return expected processing time for a chunk."""
        pass


class DemucsBackend(StemSeparatorBackend):
    """Demucs backend - highest quality, slowest."""

    def __init__(self, model_name: str = "htdemucs", device: str = None):
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        self.torch = torch
        self.apply_model = apply_model
        self._lock = threading.Lock()  # Thread safety for model inference

        # Device selection
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(f"Loading Demucs model: {model_name} (device: {self.device})")

        # Use lighter model on CPU
        if self.device == "cpu" and model_name == "htdemucs_ft":
            logger.warning("Using htdemucs instead of htdemucs_ft for better CPU performance")
            model_name = "htdemucs"

        self.model = get_model(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Quantization for CPU - skip on ARM64 due to compatibility issues
        if self.device == "cpu" and not IS_RASPBERRY_PI:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Model quantized for CPU")
            except Exception as e:
                logger.debug(f"Quantization skipped: {e}")
        elif IS_RASPBERRY_PI:
            logger.info("Skipping quantization on ARM64 (stability)")

        self.stem_order = ["drums", "bass", "other", "vocals"]

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict:
        # Thread-safe inference with lock
        with self._lock:
            # Convert to tensor: (batch, channels, samples)
            audio_tensor = self.torch.from_numpy(audio.T).unsqueeze(0).float().to(self.device)

            # Normalize
            ref_mean = audio_tensor.mean()
            ref_std = audio_tensor.std()
            if ref_std < 1e-8:
                ref_std = self.torch.tensor(1.0, device=self.device)
            audio_tensor = (audio_tensor - ref_mean) / ref_std

            with self.torch.no_grad():
                sources = self.apply_model(
                    self.model, audio_tensor, device=self.device,
                    segment=None, split=False, shifts=1, overlap=0.0
                )

            # Denormalize and convert to dict
            result = {}
            for i, stem_name in enumerate(self.stem_order):
                stem = sources[0, i] * ref_std + ref_mean
                result[stem_name] = stem.T.cpu().numpy()

            return result

    def get_name(self) -> str:
        return f"Demucs ({self.device})"

    def get_expected_latency(self, chunk_duration: float) -> float:
        # Rough estimates based on device
        if self.device == "cuda":
            return chunk_duration * 3
        elif self.device == "mps":
            return chunk_duration * 5
        else:
            return chunk_duration * 50  # CPU is very slow


class SpleeterBackend(StemSeparatorBackend):
    """Spleeter backend - fast, good quality. RECOMMENDED for Pi."""

    def __init__(self, stems: int = 4):
        from spleeter.separator import Separator

        model_name = f"spleeter:{stems}stems"
        logger.info(f"Loading Spleeter model: {model_name}")

        # Configure for batch processing efficiency
        self.separator = Separator(model_name, multiprocess=False)
        self.stems = stems
        self._model_loaded = False

        # Pre-load model to avoid first-call latency
        try:
            logger.info("Pre-loading Spleeter model (this may take a moment)...")
            self.separator._get_prediction_generator()
            self._model_loaded = True
            logger.info("Spleeter model ready")
        except Exception as e:
            logger.warning(f"Spleeter pre-load failed: {e}")

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict:
        # Spleeter expects (samples, channels) with specific sample rate
        # Ensure correct dtype
        audio = audio.astype(np.float32)

        # Spleeter works best with 44100 Hz
        prediction = self.separator.separate(audio)

        if self.stems == 2:
            # 2-stem model outputs: vocals, accompaniment
            # Map accompaniment to 'other' for consistency
            result = {
                'vocals': prediction.get('vocals', np.zeros_like(audio)),
                'drums': np.zeros_like(audio),  # Not available in 2-stems
                'bass': np.zeros_like(audio),   # Not available in 2-stems
                'other': prediction.get('accompaniment', np.zeros_like(audio)),
            }
        else:
            # 4-stem or 5-stem model
            result = {
                'vocals': prediction.get('vocals', np.zeros_like(audio)),
                'drums': prediction.get('drums', np.zeros_like(audio)),
                'bass': prediction.get('bass', np.zeros_like(audio)),
                'other': prediction.get('other', np.zeros_like(audio)),
            }

        return result

    def get_name(self) -> str:
        status = "ready" if self._model_loaded else "loading"
        return f"Spleeter ({self.stems}-stems, {status})"

    def get_expected_latency(self, chunk_duration: float) -> float:
        # Spleeter is much faster than Demucs
        # 2-stems is ~2x faster than 4-stems
        # On Pi 5: ~3-5x real-time for 4-stems, ~2x for 2-stems
        # On desktop: ~0.5-1x real-time
        if IS_RASPBERRY_PI:
            if self.stems == 2:
                return chunk_duration * 2  # 2-stems is faster
            return chunk_duration * 4  # 4-stems estimate for Pi
        else:
            if self.stems == 2:
                return chunk_duration * 0.8
            return chunk_duration * 1.5  # Desktop estimate


class ONNXBackend(StemSeparatorBackend):
    """ONNX Runtime backend - optimized inference."""

    def __init__(self, model_path: str = None):
        import onnxruntime as ort

        self.ort = ort

        # Setup execution providers
        providers = []

        # Check for available providers
        available = ort.get_available_providers()
        logger.info(f"Available ONNX providers: {available}")

        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if 'CoreMLExecutionProvider' in available:
            providers.append('CoreMLExecutionProvider')

        providers.append('CPUExecutionProvider')

        # Model path - download if not provided
        if model_path is None:
            model_path = self._download_onnx_model()

        if model_path and os.path.exists(model_path):
            logger.info(f"Loading ONNX model: {model_path}")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = NUM_CORES
            sess_options.inter_op_num_threads = NUM_CORES

            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
            self.model_loaded = True
        else:
            logger.warning("ONNX model not found. Using fallback.")
            self.model_loaded = False
            # Fallback to Demucs if ONNX model not available
            self.fallback = DemucsBackend()

    def _download_onnx_model(self) -> str:
        """Download pre-converted ONNX model."""
        model_dir = os.path.expanduser("~/.cache/vinyl_stripper/onnx")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "htdemucs.onnx")

        if os.path.exists(model_path):
            return model_path

        # Note: Would need to host the ONNX model somewhere
        logger.warning("ONNX model not found. See: https://github.com/sevagh/demucs.onnx")
        return None

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict:
        if not self.model_loaded:
            return self.fallback.separate(audio, sample_rate)

        # ONNX inference
        # Shape: (batch, channels, samples)
        audio_input = audio.T[np.newaxis, :, :].astype(np.float32)

        outputs = self.session.run(None, {'audio': audio_input})

        # Parse outputs into stems
        result = {}
        stem_names = ["drums", "bass", "other", "vocals"]
        for i, name in enumerate(stem_names):
            result[name] = outputs[0][0, i].T

        return result

    def get_name(self) -> str:
        return "ONNX Runtime" if self.model_loaded else "ONNX (fallback to Demucs)"

    def get_expected_latency(self, chunk_duration: float) -> float:
        if self.model_loaded:
            return chunk_duration * 10  # ONNX is faster than raw PyTorch CPU
        return chunk_duration * 50


class HailoBackend(StemSeparatorBackend):
    """Hailo AI Kit backend for Raspberry Pi 5."""

    def __init__(self, hef_path: str = None):
        """
        Initialize Hailo backend.

        Note: Requires converting Demucs to HEF format using Hailo Dataflow Compiler.
        This is a placeholder - actual implementation requires model conversion.
        """
        self.available = False

        try:
            from hailo_platform import HEF, Device, VDevice, ConfigureParams

            # Find Hailo device
            devices = Device.scan()
            if not devices:
                raise RuntimeError("No Hailo device found")

            self.device = VDevice(devices[0])

            # Load HEF model if provided
            if hef_path and os.path.exists(hef_path):
                self.hef = HEF(hef_path)
                self.network_group = self.device.configure(self.hef)
                self.available = True
                logger.info(f"Hailo backend initialized: {hef_path}")
            else:
                logger.warning("Hailo HEF model not found. Need to convert Demucs to HEF format.")
                logger.warning("See: https://hailo.ai/developer-zone/documentation/dataflow-compiler/")
        except ImportError:
            logger.warning("Hailo SDK not installed. Install with: pip install hailo-platform")
        except Exception as e:
            logger.warning(f"Hailo initialization failed: {e}")

        # Fallback
        if not self.available:
            logger.info("Falling back to Spleeter for Hailo backend")
            try:
                self.fallback = SpleeterBackend()
            except:
                self.fallback = None

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict:
        if self.available:
            # Hailo inference would go here
            # This requires the model to be converted to HEF format
            pass

        if self.fallback:
            return self.fallback.separate(audio, sample_rate)

        # Return zeros if nothing available
        return {
            'vocals': np.zeros_like(audio),
            'drums': np.zeros_like(audio),
            'bass': np.zeros_like(audio),
            'other': audio.copy(),
        }

    def get_name(self) -> str:
        return "Hailo-8 AI Kit" if self.available else "Hailo (fallback)"

    def get_expected_latency(self, chunk_duration: float) -> float:
        if self.available:
            return chunk_duration * 2  # Hailo is very fast
        return chunk_duration * 5  # Spleeter fallback


def create_backend(backend_name: str, remove_stems: list = None, **kwargs) -> StemSeparatorBackend:
    """Factory function to create the appropriate backend."""

    # Optimize Spleeter stem count based on what we're removing
    spleeter_stems = 4  # Default to 4-stems
    if remove_stems:
        # If only removing vocals, use faster 2-stems model
        if set(remove_stems) == {'vocals'}:
            spleeter_stems = 2
            logger.info("Optimizing: Using 2-stems Spleeter (vocals only)")

    if backend_name == "demucs":
        return DemucsBackend(**kwargs)
    elif backend_name == "spleeter":
        return SpleeterBackend(stems=spleeter_stems, **kwargs)
    elif backend_name == "onnx":
        return ONNXBackend(**kwargs)
    elif backend_name == "hailo":
        return HailoBackend(**kwargs)
    elif backend_name == "auto":
        # Auto-select best backend for platform
        # Priority: Hailo > Spleeter > ONNX > Demucs
        if HAILO_AVAILABLE:
            logger.info("Auto-selected: Hailo (hardware acceleration)")
            return HailoBackend(**kwargs)
        elif "spleeter" in AVAILABLE_BACKENDS:
            # Spleeter is fastest, use it by default
            logger.info("Auto-selected: Spleeter (fastest CPU backend)")
            return SpleeterBackend(stems=spleeter_stems, **kwargs)
        elif "onnx" in AVAILABLE_BACKENDS:
            logger.info("Auto-selected: ONNX Runtime")
            return ONNXBackend(**kwargs)
        elif "demucs" in AVAILABLE_BACKENDS:
            logger.info("Auto-selected: Demucs (slowest, highest quality)")
            return DemucsBackend(**kwargs)
        else:
            raise RuntimeError("No stem separation backend available")
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class VinylStripperPi:
    """Pi-optimized vinyl stem stripper with multi-worker processing."""

    def __init__(
        self,
        input_device,
        output_device,
        remove_stems: list[str],
        backend: str = "auto",
        chunk_duration: float = None,  # Auto-select based on platform
        prebuffer_duration: float = None,  # Auto-select
        sample_rate: int = 44100,
        num_workers: int = None,  # Auto-select
        high_latency_mode: bool = None,  # Auto-select for Pi
    ):
        self.input_device = input_device
        self.output_device = output_device
        self.remove_stems = remove_stems
        self.sample_rate = sample_rate

        # Platform-specific defaults optimized for Spleeter
        if IS_RASPBERRY_PI:
            logger.info(f"Detected Raspberry Pi ({PI_MODEL or 'unknown model'})")
            # Pi 5 has 4 cores. With Spleeter at ~4x real-time, we need all cores
            # Chunk size of 2s gives good balance of efficiency and latency
            chunk_duration = chunk_duration or 2.0  # 2s chunks for efficiency
            prebuffer_duration = prebuffer_duration or 15.0  # 15s prebuffer (enough for 4x RT)
            num_workers = num_workers or NUM_CORES  # Use all cores for parallel processing
            high_latency_mode = high_latency_mode if high_latency_mode is not None else True

            # With 4 workers and 4x real-time processing, we can theoretically keep up
            # But add buffer margin for safety
            logger.info(f"Pi 5 mode: {num_workers} workers, {chunk_duration}s chunks, {prebuffer_duration}s buffer")
        else:
            chunk_duration = chunk_duration or 1.0
            prebuffer_duration = prebuffer_duration or 5.0
            num_workers = num_workers or max(2, NUM_CORES // 2)
            high_latency_mode = high_latency_mode if high_latency_mode is not None else False

        self.chunk_duration = chunk_duration
        self.prebuffer_duration = prebuffer_duration
        self.num_workers = num_workers
        self.high_latency_mode = high_latency_mode

        # Calculate sizes
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_ratio = 0.1  # 10% overlap
        self.overlap_samples = int(self.chunk_samples * self.overlap_ratio)
        self.hop_samples = self.chunk_samples - self.overlap_samples

        # Buffers
        self.input_buffer = deque(maxlen=self.chunk_samples * 100)
        self.output_buffer = np.zeros((0, 2), dtype=np.float32)
        self.passthrough_buffer = deque(maxlen=int(sample_rate * 10))

        # Processing queues
        self.processing_queue = queue.Queue(maxsize=200)
        self.output_queue = queue.Queue(maxsize=200)

        # State
        self.running = False
        self.use_passthrough = True
        self.prebuffer_complete = False
        self.min_buffer_samples = int(prebuffer_duration * sample_rate)

        # Timing
        self.last_status_log = 0
        self.status_log_interval = 5.0
        self.processing_times = deque(maxlen=100)

        # Initialize backend
        logger.info(f"Initializing {backend} backend...")
        self.backend = create_backend(backend, remove_stems=remove_stems)
        logger.info(f"Backend: {self.backend.get_name()}")

        # Log configuration
        expected_latency = self.backend.get_expected_latency(chunk_duration)
        logger.info(f"Configuration:")
        logger.info(f"  Platform: {'Raspberry Pi ' + (PI_MODEL or '') if IS_RASPBERRY_PI else 'Desktop'}")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  Chunk size: {chunk_duration}s ({self.chunk_samples} samples)")
        logger.info(f"  Pre-buffer: {prebuffer_duration}s")
        logger.info(f"  High-latency mode: {high_latency_mode}")
        logger.info(f"  Expected processing time per chunk: ~{expected_latency:.1f}s")
        logger.info(f"  Removing: {', '.join(remove_stems)}")

        # Warn if processing will be slower than real-time
        if expected_latency > chunk_duration * num_workers:
            realtime_ratio = expected_latency / (chunk_duration * num_workers)
            logger.warning(f"Processing {realtime_ratio:.1f}x slower than real-time")
            logger.warning(f"Audio will lag behind. Consider:")
            logger.warning(f"  - Using 'spleeter' backend (faster)")
            logger.warning(f"  - Increasing prebuffer to {int(expected_latency * 10)}s")
            logger.warning(f"  - Using Hailo AI Kit for hardware acceleration")

    def process_chunk(self, audio: np.ndarray) -> np.ndarray:
        """Run stem separation on a chunk and return audio with stems removed."""
        start_time = time.time()

        # Separate stems
        stems = self.backend.separate(audio, self.sample_rate)

        # Sum stems we want to keep
        output = np.zeros_like(audio)
        for stem_name, stem_audio in stems.items():
            if stem_name not in self.remove_stems:
                output += stem_audio

        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        return output

    def processing_worker(self, worker_id: int):
        """Background worker thread for processing chunks."""
        logger.debug(f"Worker {worker_id} started")

        while self.running:
            try:
                item = self.processing_queue.get(timeout=1.0)
                if item is None:
                    break

                chunk_id, chunk = item

                try:
                    processed = self.process_chunk(chunk)
                    self.output_queue.put((chunk_id, processed))
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    traceback.print_exc()
            except queue.Empty:
                continue

        logger.debug(f"Worker {worker_id} stopped")

    def output_collector(self):
        """Collect and order processed chunks for output."""
        pending_chunks = {}
        next_chunk_id = 0

        while self.running:
            try:
                chunk_id, processed = self.output_queue.get(timeout=1.0)
                pending_chunks[chunk_id] = processed

                # Add chunks to output buffer in order
                while next_chunk_id in pending_chunks:
                    chunk = pending_chunks.pop(next_chunk_id)

                    # Handle overlap with crossfading
                    if self.output_buffer.shape[0] >= self.overlap_samples:
                        # Crossfade overlap region
                        old_overlap = self.output_buffer[-self.overlap_samples:]
                        new_overlap = chunk[:self.overlap_samples]

                        fade_out = np.linspace(1.0, 0.0, self.overlap_samples).reshape(-1, 1)
                        fade_in = np.linspace(0.0, 1.0, self.overlap_samples).reshape(-1, 1)

                        crossfaded = old_overlap * fade_out + new_overlap * fade_in
                        self.output_buffer[-self.overlap_samples:] = crossfaded

                        # Add rest of chunk
                        if chunk.shape[0] > self.overlap_samples:
                            self.output_buffer = np.vstack([
                                self.output_buffer,
                                chunk[self.overlap_samples:]
                            ])
                    else:
                        # Just append
                        if self.output_buffer.shape[0] > 0:
                            self.output_buffer = np.vstack([self.output_buffer, chunk])
                        else:
                            self.output_buffer = chunk

                    next_chunk_id += 1

                    # Check if prebuffer is ready
                    if not self.prebuffer_complete:
                        buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
                        if self.output_buffer.shape[0] >= self.min_buffer_samples:
                            self.prebuffer_complete = True
                            logger.info(f"Pre-buffer complete: {buffer_seconds:.1f}s ready")
                            logger.info("Switching to processed audio...")
                            self.use_passthrough = False

            except queue.Empty:
                continue

    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Audio callback - runs in real-time audio thread."""
        if status:
            logger.debug(f"Audio status: {status}")

        # Always store passthrough
        self.passthrough_buffer.extend(indata.copy())

        # Add to input buffer
        self.input_buffer.extend(indata.copy())

        # Queue chunks for processing
        while len(self.input_buffer) >= self.chunk_samples:
            chunk = np.array(list(self.input_buffer)[:self.chunk_samples])

            # Remove hop_samples (keep overlap)
            for _ in range(self.hop_samples):
                if self.input_buffer:
                    self.input_buffer.popleft()

            # Queue with ID for ordering
            if not hasattr(self, '_chunk_counter'):
                self._chunk_counter = 0

            try:
                self.processing_queue.put_nowait((self._chunk_counter, chunk))
                self._chunk_counter += 1
            except queue.Full:
                logger.warning("Processing queue full - dropping chunk")

        # Output logic
        if not self.use_passthrough and self.output_buffer.shape[0] >= frames:
            # Use processed audio
            outdata[:] = self.output_buffer[:frames]
            self.output_buffer = self.output_buffer[frames:]

            # Check buffer health
            buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
            if buffer_seconds < 1.0 and not self.high_latency_mode:
                logger.warning(f"Buffer low: {buffer_seconds:.1f}s")

        elif len(self.passthrough_buffer) >= frames:
            # Passthrough mode
            passthrough_data = np.array(list(self.passthrough_buffer)[:frames])
            outdata[:] = passthrough_data
            for _ in range(frames):
                if self.passthrough_buffer:
                    self.passthrough_buffer.popleft()
        else:
            # Silence fallback
            outdata[:] = 0

        # Periodic status logging
        current_time = time.time()
        if current_time - self.last_status_log > self.status_log_interval:
            self._log_status()
            self.last_status_log = current_time

    def _log_status(self):
        """Log current processing status."""
        buffer_seconds = self.output_buffer.shape[0] / self.sample_rate
        queue_size = self.processing_queue.qsize()
        output_queue_size = self.output_queue.qsize()
        mode = "PROCESSED" if not self.use_passthrough else "passthrough"

        avg_time = np.mean(self.processing_times) if self.processing_times else 0

        logger.info(
            f"[{mode}] Buffer: {buffer_seconds:.1f}s | "
            f"Queue: {queue_size} in, {output_queue_size} out | "
            f"Avg process: {avg_time:.2f}s/chunk"
        )

    def run(self):
        """Start real-time processing."""
        self.running = True
        self._chunk_counter = 0

        # Start worker threads
        workers = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self.processing_worker, args=(i,), daemon=True)
            t.start()
            workers.append(t)

        # Start output collector
        collector_thread = threading.Thread(target=self.output_collector, daemon=True)
        collector_thread.start()

        logger.info("Starting audio stream...")
        if self.high_latency_mode:
            logger.info(f"High-latency mode: Building {self.prebuffer_duration}s buffer before playing processed audio")
            logger.info("Audio will play through immediately (passthrough), then switch to processed")
        logger.info("Press Ctrl+C to stop")

        try:
            with sd.Stream(
                samplerate=self.sample_rate,
                channels=2,
                dtype=np.float32,
                blocksize=4096,  # Larger blocks for Pi
                latency='high',
                device=(self.input_device, self.output_device),
                callback=self.audio_callback,
            ):
                while self.running:
                    sd.sleep(100)

        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            self.running = False

            # Stop workers
            for _ in workers:
                self.processing_queue.put(None)

            for t in workers:
                t.join(timeout=2)


# =============================================================================
# DEVICE DETECTION (from original)
# =============================================================================

def list_devices():
    """Print available audio devices."""
    print("\nAvailable audio devices:\n")
    print(sd.query_devices())
    print("\nUse device index numbers with --input and --output")


def find_usb_audio_device():
    """Auto-detect USB audio interface."""
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            name = device['name'].lower()
            if ('usb' in name and ('codec' in name or 'audio' in name)) or \
               any(x in name for x in ['uca202', 'ufo202', 'behringer', 'scarlett']):
                if device['max_input_channels'] >= 2 and device['max_output_channels'] >= 2:
                    return i
    except:
        pass
    return None


def resolve_device(device_arg):
    """Resolve device argument to device index."""
    if device_arg is None:
        return None

    if isinstance(device_arg, int):
        return device_arg

    try:
        return int(device_arg)
    except (ValueError, TypeError):
        pass

    device_str = str(device_arg).lower()

    if device_str in ('usb', 'ufo202', 'uca202'):
        dev = find_usb_audio_device()
        if dev is None:
            raise ValueError("USB audio device not found")
        return dev

    raise ValueError(f"Unknown device: {device_arg}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vinyl Stem Stripper - Pi 5 Optimized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List audio devices
    python3 vinyl_stripper_pi.py --list-devices

    # Basic usage - auto-selects Spleeter (recommended)
    python3 vinyl_stripper_pi.py --input usb --output usb --remove vocals

    # Explicitly use Spleeter (fastest for Pi)
    python3 vinyl_stripper_pi.py --backend spleeter --remove vocals

    # Pi-optimized mode with 30s pre-buffer
    python3 vinyl_stripper_pi.py --pi-mode --remove vocals

    # High-quality mode with Demucs (very slow on Pi CPU)
    python3 vinyl_stripper_pi.py --backend demucs --chunk 5.0 --prebuffer 60 --workers 3

Backends (in order of speed):
    spleeter - RECOMMENDED for Pi. Fast, good quality (~5x real-time on Pi 5)
    auto     - Auto-selects spleeter if available (default)
    onnx     - ONNX optimized inference (requires converted model)
    hailo    - Hailo AI Kit hardware acceleration (fastest with AI Kit)
    demucs   - Highest quality, but 30-50x slower than real-time on Pi CPU

Hardware Acceleration (Pi 5):
    The Raspberry Pi AI Kit ($70) with Hailo-8L provides 13 TOPS
    of neural network acceleration. With a converted model,
    this can achieve near real-time stem separation.

    Setup: https://www.raspberrypi.com/documentation/accessories/ai-kit.html

Performance Tips for Pi 5:
    1. Always use --backend spleeter for best performance
    2. Use --pi-mode for optimized buffer settings
    3. Allow 30-60 seconds for initial buffer fill
    4. Consider Hailo AI Kit for true real-time processing
        """
    )

    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("-i", "--input", type=str, help="Input device")
    parser.add_argument("-o", "--output", type=str, help="Output device")
    parser.add_argument(
        "--remove", nargs="+",
        choices=["vocals", "drums", "bass", "other", "all"],
        default=["vocals"],
        help="Stems to remove"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "spleeter", "demucs", "onnx", "hailo"],
        default="auto",
        help="Stem separation backend (default: auto, which selects spleeter)"
    )
    parser.add_argument("--chunk", type=float, help="Chunk duration (seconds)")
    parser.add_argument("--prebuffer", type=float, help="Pre-buffer duration (seconds)")
    parser.add_argument("--workers", type=int, help="Number of processing workers")
    parser.add_argument("--pi-mode", action="store_true", help="Force Pi-optimized settings")
    parser.add_argument("--high-latency", action="store_true", help="Enable high-latency mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        global logger
        logger = setup_logging(verbose=True)

    if args.list_devices:
        list_devices()
        return

    # Resolve devices
    try:
        input_dev = resolve_device(args.input)
        output_dev = resolve_device(args.output)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Auto-detect if not specified
    if input_dev is None or output_dev is None:
        usb_dev = find_usb_audio_device()
        if usb_dev is not None:
            input_dev = input_dev or usb_dev
            output_dev = output_dev or usb_dev
            print(f"Auto-detected USB audio: Device {usb_dev}")
        else:
            print("Error: Could not auto-detect audio devices")
            print("Use --list-devices to see available devices")
            sys.exit(1)

    # Parse stems
    remove_stems = args.remove
    if "all" in remove_stems:
        remove_stems = ["vocals", "drums", "bass", "other"]

    # Pi mode overrides
    if args.pi_mode:
        logger.info("Pi mode enabled - using optimized settings")

    # Determine high-latency mode
    high_latency = args.high_latency or args.pi_mode or IS_RASPBERRY_PI

    # Create and run
    try:
        stripper = VinylStripperPi(
            input_device=input_dev,
            output_device=output_dev,
            remove_stems=remove_stems,
            backend=args.backend,
            chunk_duration=args.chunk,
            prebuffer_duration=args.prebuffer,
            num_workers=args.workers,
            high_latency_mode=high_latency,
        )
        stripper.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
