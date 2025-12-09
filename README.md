# Vinyl Stem Stripper

Real-time vocal/drum removal for vinyl playback. Play a record, hear it without vocals (or drums, or both).

**Workflow**: Sound Burger → Raspberry Pi → Headphones

## Quick Start (Raspberry Pi)

See [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md) for complete Raspberry Pi setup guide.

**Not sure what hardware to buy?** See [HARDWARE_COMPARISON.md](HARDWARE_COMPARISON.md) for detailed comparison and shopping list.

```bash
# 1. Setup (one time)
./edge_setup.sh

# 2. Connect hardware:
#    Sound Burger LINE OUT → ADC input (HiFiBerry ADC+ or USB dongle)
#    Headphones → innomaker HiFi DAC HAT (3.5mm or RCA outputs)

# 3. Run (auto-detects devices)
python3 vinyl_stripper.py --remove vocals
```

## Testing with Streaming Audio

Test the stem separation without vinyl hardware using streaming audio:

```bash
# List available audio devices
python3 test_streaming.py --list-devices

# Test with Universal Audio Thunderbolt interface (Apollo, etc.)
python3 test_streaming.py --input ua --remove vocals

# Test with microphone input
python3 test_streaming.py --input mic --remove vocals

# Test with system audio loopback (requires BlackHole on Mac)
python3 test_streaming.py --input loopback --remove vocals

# Test with specific devices
python3 test_streaming.py --input 1 --output 0 --remove vocals
```

**For Universal Audio Apollo/Thunderbolt:**
- Connect Sound Burger LINE OUT → Apollo inputs
- Run: `python3 test_streaming.py --input ua --remove vocals`
- Or use main script: `python3 vinyl_stripper.py --input <apollo_device_index> --output <apollo_device_index>`

**For macOS loopback (to capture system audio):**
1. Install BlackHole: https://github.com/ExistentialAudio/BlackHole
2. Set BlackHole as output device in System Preferences
3. Run: `python3 test_streaming.py --input loopback --remove vocals`
4. Play music from Spotify/YouTube/etc. - it will be processed!

## Driver Requirements

**For the app: NO drivers needed** - uses standard system audio APIs.

**For Universal Audio Apollo: YES** - install UAD software from https://www.uaudio.com/uad/downloads.html

See [DRIVER_REQUIREMENTS.md](DRIVER_REQUIREMENTS.md) for complete details.

## Setup (Mac - Development/Testing)

### 1. Install Universal Audio Drivers (if using Apollo)

Download and install UAD software from Universal Audio:
- https://www.uaudio.com/uad/downloads.html
- Includes Thunderbolt/USB drivers for Apollo interfaces
- Restart computer after installation

### 2. Install Python dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install torch torchaudio sounddevice numpy

# Install demucs
pip3 install demucs
```

### 2. Find your audio devices

```bash
python3 vinyl_stripper.py --list-devices
```

Look for your Apollo Twin in the list. Note the index numbers for input and output.

Example output:
```
  0 Built-in Microphone, Core Audio (2 in, 0 out)
  1 Apollo Twin USB, Core Audio (2 in, 2 out)    <-- use this
  2 Built-in Output, Core Audio (0 in, 2 out)
```

### 3. Connect your turntable

```
Sound Burger (line out) → 1/8" to 1/4" cable → Apollo Twin inputs
Apollo Twin outputs → Headphones or monitors
```

Make sure the Sound Burger is set to LINE out (not phono).

### 4. Run it

```bash
# Remove vocals (default)
python3 vinyl_stripper.py --input 1 --output 1 --remove vocals

# Remove drums
python3 vinyl_stripper.py --input 1 --output 1 --remove drums

# Remove both vocals and drums (just bass + other)
python3 vinyl_stripper.py --input 1 --output 1 --remove both
```

Replace `1` with your actual Apollo Twin device index.

## Options

| Flag | Description |
|------|-------------|
| `--list-devices` | Show available audio devices |
| `-i`, `--input` | Input device index |
| `-o`, `--output` | Output device index |
| `--remove` | What to remove: `vocals`, `drums`, `both`, or `none` |
| `--model` | Demucs model (default: `htdemucs`) |
| `--chunk` | Chunk duration in seconds (default: 3.0) |

## Expected Behavior

- **Latency:** ~3-4 seconds from input to output
- **First few seconds:** Silence while the buffer fills
- **Quality:** Should sound clean, similar to offline Demucs processing

## Troubleshooting

**No sound output:**
- Check that your Apollo Twin is set as the output device in the script AND in macOS Sound preferences
- Make sure the turntable is set to LINE out
- Check input levels in your Apollo Console app

**Choppy/glitchy audio:**
- Try increasing chunk size: `--chunk 5.0`
- Close other applications
- Make sure you're on power (not battery)

**Model download fails:**
- Demucs downloads models on first run (~1GB for htdemucs)
- Make sure you have internet connectivity

**"Device not found" error:**
- Run `--list-devices` and double-check the index numbers

## Product Workflow

```
┌─────────────┐
│ Sound Burger│  (Audio-Technica portable turntable)
│  LINE OUT   │
└──────┬──────┘
       │ 3.5mm cable
       ▼
┌──────────────────┐
│ ADC Board        │  (Input: HiFiBerry ADC+ or USB dongle)
│ (Audio Input)    │
└──────┬───────────┘
       │ GPIO/USB
       ▼
┌──────────────┐
│ Raspberry Pi │  (Real-time stem separation)
│     5        │
└──────┬───────┘
       │ GPIO
       ▼
┌──────────────────┐
│ innomaker DAC    │  (High-quality output: 112dB SNR)
│ HAT (PCM5122)    │
│ 3.5mm/RCA Output │
└──────┬───────────┘
       │
       ▼
  ┌─────────┐
  │Headphones│  (Listen to processed audio)
  └─────────┘
```

## Edge Computing Product

This is designed as a portable edge computing device:

- **Hardware**: Raspberry Pi 5 + USB Audio Interface
- **Power**: USB-C Power Bank (2-3 hours runtime)
- **Size**: Fits in bag with Sound Burger
- **Latency**: ~5-6 seconds
- **Quality**: Good (optimized for CPU)

See [EDGE_PRODUCT_ROADMAP.md](EDGE_PRODUCT_ROADMAP.md) for full product roadmap.
