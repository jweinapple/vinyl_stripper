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


class VinylStripper:
    def __init__(
        self,
        input_device: int,
        output_device: int,
        remove_stems: list[str],
        chunk_duration: float = 0.25,  # seconds per chunk (very small for fastest processing)
        overlap: float = 0.1,  # minimal overlap for faster processing
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
        self.output_queue = queue.Queue(maxsize=50)
        self.output_buffer = np.zeros((0, 2), dtype=np.float32)
        self.passthrough_buffer = deque(maxlen=int(sample_rate * 3))
        self.processing_queue = queue.Queue(maxsize=20)
        self.running = False
        self.buffer_filling = True
        self.use_passthrough = False
        
        # Load model with device-specific optimization
        # Reference torch before any local imports to avoid UnboundLocalError
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = self.model_name if self.model_name else "htdemucs_ft"
        print(f"Loading model: {model_name} (device: {self.device})...")
        
        try:
            self.model = get_model(model_name)
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            print("\nAvailable models: htdemucs, htdemucs_ft, htdemucs_6s")
            raise
        self.model.to(self.device)
        self.model.eval()
        
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
                    print("✓ Model compiled")
                except Exception:
                    pass
            
            if self.device == "cpu":
                try:
                    from torch import quantization as torch_quantization
                    self.model = torch_quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    print("✓ Model quantized")
                except Exception:
                    pass
        
        # Stem indices in demucs output: drums, bass, other, vocals
        self.stem_indices = {
            "drums": 0,
            "bass": 1,
            "other": 2,
            "vocals": 3,
        }
        
        print(f"Removing: {', '.join(remove_stems)}")
        print(f"Chunk: {chunk_duration}s, Overlap: {overlap*100}%, Latency: ~{chunk_duration + 1:.1f}s\n")

    def process_chunk(self, audio: np.ndarray) -> np.ndarray:
        """Run stem separation on a chunk and return audio with stems removed."""
        # Convert to torch tensor: (batch, channels, samples)
        audio_tensor = torch.from_numpy(audio.T).unsqueeze(0).float().to(self.device)
        
        # Normalize
        ref = audio_tensor.mean(0)
        audio_tensor = (audio_tensor - ref.mean()) / ref.std()
        
        with torch.no_grad():
            # Apply model - returns (batch, stems, channels, samples)
            # Use smaller segment size for faster processing (reduces memory but speeds up inference)
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
        
        # Sum all stems except the ones we want to remove
        output = torch.zeros_like(audio_tensor)
        for stem_name, idx in self.stem_indices.items():
            if stem_name not in self.remove_stems:
                output += sources[0, idx]
        
        # Denormalize
        output = output * ref.std() + ref.mean()
        
        # Convert back to numpy (samples, channels)
        return output.squeeze(0).T.cpu().numpy()

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
                print(f"Processing error: {e}")

    def audio_callback(self, indata, outdata, frames, time, status):
        """Called by sounddevice for each audio block."""
        if status:
            print(f"Audio status: {status}")
        
        # Store raw audio for passthrough fallback
        self.passthrough_buffer.extend(indata.copy())
        
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
            # Only queue if we have room - prevents backing up
            if self.processing_queue.qsize() < self.processing_queue.maxsize // 2:
                try:
                    self.processing_queue.put_nowait(chunk)
                except queue.Full:
                    pass  # Skip this chunk if we're backed up
        
        # Try to get processed audio for output
        # Keep filling buffer until we have enough for smooth playback
        target_buffer_size = int(self.sample_rate * 10.0)  # Target 10 second buffer for safety
        while self.output_buffer.shape[0] < target_buffer_size:
            try:
                processed = self.output_queue.get_nowait()
                # Only use hop_samples (non-overlapping portion) to prevent overlap artifacts
                # This is the amount of new audio in each chunk
                hop_portion = processed[:self.hop_samples]
                self.output_buffer = np.vstack([self.output_buffer, hop_portion]) if self.output_buffer.shape[0] > 0 else hop_portion
                if self.buffer_filling and self.output_buffer.shape[0] >= target_buffer_size:
                    self.buffer_filling = False
                    self.use_passthrough = False
                    print(f"✓ Buffer ready ({self.output_buffer.shape[0] / self.sample_rate:.1f}s)")
            except queue.Empty:
                break
        
        # Check if we need to switch to passthrough mode
        # Use a higher threshold to prevent premature switching
        if self.output_buffer.shape[0] < int(self.sample_rate * 1.5) and not self.buffer_filling:
            # Buffer is getting low - switch to passthrough temporarily
            if not self.use_passthrough:
                self.use_passthrough = True
                print("⚠ Passthrough mode")
        
        # Output what we have
        if self.output_buffer.shape[0] >= frames and not self.use_passthrough:
            # Use processed audio
            outdata[:] = self.output_buffer[:frames]
            self.output_buffer = self.output_buffer[frames:]
            # If buffer recovered, switch back to processed mode
            if self.output_buffer.shape[0] >= target_buffer_size:
                self.use_passthrough = False
        elif self.use_passthrough and len(self.passthrough_buffer) >= frames:
            # Passthrough mode - output raw audio (better than silence)
            passthrough_data = np.array(list(self.passthrough_buffer)[:frames])
            outdata[:] = passthrough_data
            # Remove used samples from passthrough buffer
            for _ in range(frames):
                if self.passthrough_buffer:
                    self.passthrough_buffer.popleft()
        else:
            # Not enough processed audio - repeat last sample or pad with zeros
            if self.output_buffer.shape[0] > 0:
                # Repeat last samples to avoid complete silence
                last_samples = self.output_buffer[-1:]
                padding_needed = frames - self.output_buffer.shape[0]
                padding = np.tile(last_samples, (padding_needed, 1))
                outdata[:] = np.vstack([self.output_buffer, padding])
                self.output_buffer = np.zeros((0, 2), dtype=np.float32)
            else:
                # Complete silence - buffer underrun (shouldn't happen with passthrough)
                outdata[:] = 0

    def run(self):
        """Start the real-time processing."""
        self.running = True
        
        # Start processing thread
        process_thread = threading.Thread(target=self.processing_worker, daemon=True)
        process_thread.start()
        
        print("Starting audio stream... Press Ctrl+C to stop\n")
        
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
            print("\nStopping...")
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
        default=5.0,
        help="Chunk duration in seconds (default: 5.0)"
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
    stripper = VinylStripper(
        input_device=args.input,
        output_device=args.output,
        remove_stems=remove_stems,
        chunk_duration=args.chunk,
        model_name=args.model,
    )
    stripper.run()


if __name__ == "__main__":
    main()
