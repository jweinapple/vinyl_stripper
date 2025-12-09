# Vinyl Stem Stripper

Real-time vocal/drum removal for vinyl playback. Play a record, hear it without vocals (or drums, or both).

**Workflow**: Sound Burger → Raspberry Pi → Headphones

## Features

- 🎵 Real-time stem separation using Demucs
- 🎧 Remove vocals, drums, bass, or other stems
- 🔌 Auto-detects USB audio interfaces (UCA202, etc.)
- 🚀 Optimized for Raspberry Pi 5
- ⚡ Low latency (~4-5 seconds)
- 🎛️ Device nicknames: `ua` (Universal Audio), `sid` (SoundID), `usb` (USB audio)

## Quick Start (Raspberry Pi)

```bash
# 1. Setup (one time)
./edge_setup.sh

# 2. Connect hardware:
#    Sound Burger LINE OUT → USB audio interface (UCA202) RCA inputs
#    Headphones → USB audio interface headphone output

# 3. Run (auto-detects devices)
python3 vinyl_stripper.py --remove vocals
```

See [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md) for complete setup guide.

## Quick Start (macOS - Development)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. List audio devices
python3 vinyl_stripper.py --list-devices

# 4. Run with device nicknames
python3 vinyl_stripper.py --input ua --output sid --remove vocals
```

## Usage

### Basic Commands

```bash
# List available audio devices
python3 vinyl_stripper.py --list-devices

# Remove vocals (default)
python3 vinyl_stripper.py --input ua --output sid --remove vocals

# Remove drums
python3 vinyl_stripper.py --input ua --output sid --remove drums

# Remove both vocals and drums
python3 vinyl_stripper.py --input ua --output sid --remove vocals drums

# Use specific device indices
python3 vinyl_stripper.py --input 1 --output 1 --remove vocals

# Use different model
python3 vinyl_stripper.py --model htdemucs --input ua --output sid --remove vocals
```

### Device Nicknames

- `ua` or `apollo` - Universal Audio Thunderbolt interface
- `sid` or `soundid` - SoundID Reference output device
- `usb` - USB audio interface (UCA202, Behringer, etc.)

### Available Models

- `htdemucs` - Baseline Hybrid Transformer (fastest)
- `htdemucs_ft` - Fine-tuned version (default, best quality)
- `htdemucs_6s` - 6 sources (adds piano/guitar stems)

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--list-devices` | Show available audio devices | - |
| `-i`, `--input` | Input device index or nickname | Auto-detect |
| `-o`, `--output` | Output device index or nickname | Auto-detect |
| `--remove` | Stems to remove: `vocals`, `drums`, `bass`, `other`, `all` | `vocals` |
| `--model` | Demucs model: `htdemucs`, `htdemucs_ft`, `htdemucs_6s` | `htdemucs_ft` |
| `--chunk` | Chunk duration in seconds | `5.0` |

## Hardware Setup

### Raspberry Pi 5 Setup

**Required Hardware:**
- Raspberry Pi 5 (8GB recommended)
- USB audio interface (Behringer UCA202 recommended)
- USB-C power bank (20,000mAh+)
- MicroSD card (64GB+)

**Audio Flow:**
```
Sound Burger (LINE OUT)
    ↓ 3.5mm-to-RCA cable
USB Audio Interface (UCA202) RCA inputs
    ↓ USB
Raspberry Pi 5 (processing)
    ↓ USB
USB Audio Interface headphone output
    ↓
Headphones
```

See [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md) for detailed setup instructions.

### Hailo NPU Support

For Raspberry Pi 5 with Hailo-8L NPU acceleration, see [HAILO_SETUP.md](HAILO_SETUP.md).

## Expected Behavior

- **Latency:** ~4-5 seconds from input to output
- **First few seconds:** Silence while buffer fills (~10 seconds)
- **Quality:** Clean separation, optimized for real-time processing
- **CPU Usage:** 80-100% on Raspberry Pi 5

## Troubleshooting

**No sound output:**
- Run `--list-devices` to verify device detection
- Check audio interface connections
- Ensure turntable is set to LINE out (not phono)

**Choppy/glitchy audio:**
- Try increasing chunk size: `--chunk 10.0`
- Close other applications
- Ensure adequate power supply

**"Device not found" error:**
- Run `--list-devices` to see available devices
- Use device index numbers instead of nicknames
- Check USB audio interface is connected

**Model download fails:**
- Demucs downloads models automatically on first use (~500MB)
- Ensure internet connectivity
- Models are cached for future use

**Passthrough mode warning:**
- Processing can't keep up with real-time
- Audio will play unprocessed (better than silence)
- Try reducing chunk size or using `htdemucs` model

## Switching Modes (Raspberry Pi)

If running as a systemd service, use the mode switching script:

```bash
# Remove vocals
./switch_mode.sh vocals

# Remove drums
./switch_mode.sh drums

# Remove both
./switch_mode.sh both

# No removal (passthrough)
./switch_mode.sh none
```

## Architecture

```
┌─────────────┐
│ Sound Burger│  (Audio-Technica portable turntable)
│  LINE OUT   │
└──────┬──────┘
       │ 3.5mm-to-RCA cable
       ▼
┌──────────────────┐
│ USB Audio        │  (Behringer UCA202)
│ Interface        │  RCA inputs → USB → Pi
└──────┬───────────┘
       │ USB
       ▼
┌──────────────┐
│ Raspberry Pi │  (Real-time stem separation)
│     5        │  Demucs model processing
└──────┬───────┘
       │ USB
       ▼
┌──────────────────┐
│ USB Audio        │  Headphone output
│ Interface        │
└──────┬───────────┘
       │
       ▼
  ┌─────────┐
  │Headphones│  (Processed audio)
  └─────────┘
```

## Requirements

- Python 3.8+
- PyTorch
- Demucs
- sounddevice
- numpy

See `requirements.txt` for complete list.

## License

This project uses the Demucs library for stem separation. See individual library licenses.

## Contributing

This is a personal project optimized for Raspberry Pi deployment. For issues or improvements, please open an issue on GitHub.
