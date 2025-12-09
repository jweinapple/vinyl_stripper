# Hailo-8L NPU Setup Guide for Raspberry Pi 5

## Hardware Setup
- **Raspberry Pi 5** (8GB RAM)
- **Hailo-8L AI Kit** (M.2 HAT+ with Hailo AI module)
- **Behringer UCA202** (USB audio interface)
- **Audio-Technica Sound Burger** (3.5mm line out)

## Audio Flow
1. Sound Burger → 3.5mm-to-RCA → UCA202 RCA inputs
2. UCA202 → USB → Raspberry Pi
3. Pi processes audio with Demucs (accelerated by Hailo-8L)
4. Processed audio → USB → UCA202 headphone output
5. User hears processed audio (~4-5s latency)

---

## Answers to Your Questions

### 1. ✅ Will it work with UCA202 as both input and output?

**YES**, but you need to update the device detection code. The current code has USB audio detection but needs enhancement for UCA202 specifically.

**Current Status:**
- Code has `find_usb_audio_device()` function that looks for USB audio devices
- Behringer UCA202 is class-compliant USB audio (no drivers needed)
- Should work as both input and output device

**Required Changes:**
```python
# Add to vinyl_stripper.py - enhance USB detection
def find_uca202_device():
    """Find Behringer UCA202 USB audio interface."""
    devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        name = device['name'].lower()
        # UCA202 shows up as "USB Audio Device" or "UCA202" in ALSA
        if ('usb audio' in name or 'uca202' in name or 'behringer' in name):
            # Check if it has both input and output
            if device['max_input_channels'] >= 2 and device['max_output_channels'] >= 2:
                return i
    return None
```

**Testing:**
```bash
# On Pi, list devices to find UCA202
python3 vinyl_stripper.py --list-devices

# Should show something like:
# 1 USB Audio Device, ALSA (2 in, 2 out)  ← UCA202
```

---

### 2. ⚠️ Model Compatibility with Hailo-8L NPU

**CRITICAL ISSUE:** The current code uses PyTorch/Demucs models which are **NOT directly compatible** with Hailo-8L.

**Current Model:**
- `htdemucs_ft` - PyTorch model (.pth format)
- Runs on CPU/GPU/MPS, NOT on Hailo NPU

**Hailo-8L Requirements:**
- Models must be converted to Hailo's format (.hef files)
- Requires Hailo Dataflow Compiler (DFC)
- Demucs models are complex Transformer-based architectures that may not convert easily

**Options:**

#### Option A: Convert Demucs to Hailo (Complex)
1. Export Demucs to ONNX
2. Convert ONNX to Hailo format using Hailo DFC
3. May require model architecture modifications
4. **Difficulty:** High - Transformer models are challenging to convert

#### Option B: Use CPU with Optimizations (Recommended for now)
- Keep current PyTorch implementation
- Use CPU quantization (already implemented)
- Accept ~4-5s latency
- **Easier to implement, works immediately**

#### Option C: Hybrid Approach
- Use Hailo for simpler operations (if any)
- Keep Demucs on CPU
- **Not recommended** - adds complexity without clear benefit

**Recommendation:** Start with CPU (Option B), then explore Hailo conversion if needed.

---

### 3. ✅ Chunking/Buffering for ~4-5s Latency

**Current Settings:**
- Chunk duration: `0.25s` (very small)
- Target buffer: `10 seconds` (before playback starts)
- Overlap: `10%`

**For 4-5s latency, adjust:**

```python
# In vinyl_stripper.py __init__:
chunk_duration: float = 1.0,  # Increase to 1s (better for CPU processing)
overlap: float = 0.25,        # Increase overlap for smoother transitions

# In audio_callback:
target_buffer_size = int(self.sample_rate * 4.0)  # 4 second buffer
```

**Recommended Settings for Pi 5:**
- Chunk: `1.0-2.0 seconds` (larger chunks = less overhead)
- Buffer: `4-5 seconds` (matches your latency target)
- Overlap: `25%` (better quality)

---

### 4. 🔧 Changes Needed for Raspberry Pi OS + Hailo

**System Setup:**

1. **Install Hailo Runtime** (if using Hailo):
```bash
# Follow Hailo documentation for Pi 5 setup
# Install Hailo runtime libraries
sudo apt-get install hailo-runtime
```

2. **Install Audio Dependencies:**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libsndfile1 \
    alsa-utils
```

3. **USB Audio Permissions:**
```bash
# Add user to audio group
sudo usermod -a -G audio $USER
# Log out and back in for changes to take effect
```

4. **Update edge_setup.sh:**
```bash
# Already handles Pi setup, but verify:
- Installs portaudio19-dev (for sounddevice)
- Installs libsndfile1 (for audio file support)
- Creates systemd service
```

5. **Model Loading (if staying with CPU):**
```python
# Current code already handles CPU quantization:
if self.device == "cpu":
    # Quantize model for CPU performance
    self.model = torch.quantization.quantize_dynamic(...)
```

---

### 5. ✅ Auto-Start on Boot

**Current Setup:**
The `edge_setup.sh` script already creates a systemd service, but needs updates for your setup.

**Updated Service File:**

Create `/etc/systemd/system/vinyl-stripper.service`:

```ini
[Unit]
Description=Vinyl Stem Stripper
After=sound.target network.target
Wants=sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/vinyl_stripper
Environment="PATH=/home/pi/vinyl_stripper/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/pi/vinyl_stripper/venv/bin/python3 /home/pi/vinyl_stripper/vinyl_stripper.py --input 1 --output 1 --remove vocals
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Enable Service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable vinyl-stripper
sudo systemctl start vinyl-stripper

# Check status
sudo systemctl status vinyl-stripper

# View logs
sudo journalctl -u vinyl-stripper -f
```

**Mode Switching Script:**

Create `/home/pi/vinyl_stripper/switch_mode.sh`:

```bash
#!/bin/bash
# Switch vinyl stripper mode via SSH

MODE=$1
SERVICE="vinyl-stripper"

if [ -z "$MODE" ]; then
    echo "Usage: ./switch_mode.sh [vocals|drums|both|none]"
    exit 1
fi

# Stop service
sudo systemctl stop $SERVICE

# Update service file with new mode
sudo sed -i "s/--remove [a-z]*/--remove $MODE/" /etc/systemd/system/$SERVICE.service

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl start $SERVICE

echo "Mode switched to: $MODE"
sudo systemctl status $SERVICE
```

**Usage:**
```bash
# Via SSH
ssh pi@raspberrypi.local
cd ~/vinyl_stripper
./switch_mode.sh vocals    # Remove vocals
./switch_mode.sh drums     # Remove drums
./switch_mode.sh both      # Remove both
./switch_mode.sh none      # No removal (passthrough)
```

---

## Recommended Implementation Steps

### Phase 1: Get It Working (CPU)
1. ✅ Test UCA202 detection
2. ✅ Verify audio I/O works
3. ✅ Test with CPU processing
4. ✅ Measure actual latency
5. ✅ Set up auto-start

### Phase 2: Optimize (if needed)
1. Adjust chunk size for Pi 5 performance
2. Fine-tune buffer sizes
3. Test different models (htdemucs vs htdemucs_ft)

### Phase 3: Hailo Integration (optional)
1. Research Demucs → Hailo conversion
2. Test with simpler models first
3. Benchmark performance gains

---

## Quick Start Commands

```bash
# 1. Setup
cd ~/vinyl_stripper
./edge_setup.sh

# 2. Find UCA202 device
python3 vinyl_stripper.py --list-devices

# 3. Test (replace X with UCA202 device index)
python3 vinyl_stripper.py --input X --output X --remove vocals

# 4. Enable auto-start
sudo systemctl enable vinyl-stripper
sudo systemctl start vinyl-stripper
```

---

## Troubleshooting

**UCA202 not detected:**
```bash
# Check USB connection
lsusb | grep Behringer

# Check ALSA devices
arecord -l  # Input devices
aplay -l    # Output devices

# Test audio
arecord -D hw:1,0 -f cd test.wav  # Record
aplay -D hw:1,0 test.wav          # Playback
```

**High latency:**
- Increase chunk size to 1.0-2.0s
- Reduce buffer to 4-5s
- Use `htdemucs` instead of `htdemucs_ft` (faster)

**Service won't start:**
- Check logs: `sudo journalctl -u vinyl-stripper`
- Verify device indices: `python3 vinyl_stripper.py --list-devices`
- Check permissions: `sudo usermod -a -G audio pi`


