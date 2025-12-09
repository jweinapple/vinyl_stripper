# Raspberry Pi Setup Guide
## Sound Burger → Raspberry Pi → Headphones (No USB Audio Interface)

Complete setup guide for portable vinyl stem stripping with Audio-Technica Sound Burger using GPIO-based audio input.

## Hardware Required

### Core Components
- **Raspberry Pi 5** (8GB recommended) - $75
- **innomaker HiFi DAC HAT** (PCM5122, GPIO audio output board) - $30
  - High-quality DAC with 112dB SNR
  - RCA outputs + 3.5mm headphone jack
  - Better quality than Pi's built-in audio
- **Audio Input Solution** (for Sound Burger):
  - Option A: HiFiBerry ADC+ (~$35-50) - GPIO-based ADC
  - Option B: Simple USB audio dongle (~$5-10) - Cheaper, simpler
  - Option C: Generic I2S ADC module (~$10-15)
- **USB-C Power Bank** (20,000mAh+, 27W+) - $40
- **MicroSD Card** (64GB+, Class 10) - $10
- **Audio-Technica Sound Burger** (with line out)

### Cables
- **3.5mm to 3.5mm cable** (Sound Burger line out → ADC input)
- **Headphones** (3.5mm for DAC HAT headphone jack, or use RCA outputs)
- **USB-C cable** (for power bank)

## Physical Setup

```
┌─────────────┐
│ Sound Burger│
│  (Vinyl)    │
└──────┬──────┘
       │ 3.5mm LINE OUT
       ▼
┌──────────────────┐
│ ADC Board        │  ← Input (HiFiBerry ADC+ or USB dongle)
│ (Audio Input)    │
└──────┬───────────┘
       │ GPIO/USB
       ▼
┌──────────────┐      ┌──────────┐
│ Raspberry Pi │      │Power Bank│
│     5        │◄─────┤          │
│              │ USB-C│          │
└──────┬───────┘      └──────────┘
       │ GPIO header
       ▼
┌──────────────────┐
│ innomaker DAC    │  ← Output (High-quality DAC HAT)
│ HAT (PCM5122)    │
│ 3.5mm/RCA Output │
└──────┬───────────┘
       │ 3.5mm or RCA
       ▼
  ┌─────────┐
  │Headphones│
  └─────────┘
```

## Software Setup

### 1. Install Raspberry Pi OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Flash Raspberry Pi OS (64-bit) to microSD card
3. Enable SSH and configure WiFi during imaging
4. Boot Raspberry Pi

### 2. Install Audio HAT Drivers

```bash
# SSH into Raspberry Pi
ssh pi@raspberrypi.local

# Update system
sudo apt update && sudo apt upgrade -y

# Configure I2S for DAC HAT
sudo nano /boot/config.txt
# Add this line for innomaker DAC HAT:
# dtoverlay=hifiberry-dac

# If using HiFiBerry ADC+ for input, also add:
# dtoverlay=hifiberry-dacplusadc

# Or for generic I2S:
# dtoverlay=i2s-mmap

# Save and reboot
sudo reboot
```

**Note**: The innomaker HiFi DAC HAT uses the same driver as HiFiBerry DAC, so `dtoverlay=hifiberry-dac` works.

### 3. Verify Audio Input

After reboot, check if ADC is detected:

```bash
# List audio devices
aplay -l
arecord -l

# Test recording
arecord -D hw:1,0 -f cd -d 5 test.wav
# Play it back
aplay -D hw:0,0 test.wav
```

### 4. Install Python Dependencies

```bash
# Install system packages
sudo apt install -y python3-pip python3-venv git portaudio19-dev libsndfile1

# Clone/copy project
cd ~
# (copy your vinyl_stripper files here)

# Run setup
cd vinyl_stripper
chmod +x edge_setup.sh
./edge_setup.sh
```

### 5. Configure Audio Devices

```bash
source venv/bin/activate
python3 vinyl_stripper.py --list-devices
```

Expected output (with innomaker DAC HAT + ADC):
```
Available audio devices:

  0 snd_rpi_hifiberry_dac, ALSA (0 in, 2 out)  ← Output (innomaker DAC HAT)
  1 snd_rpi_hifiberry_dacplusadc, ALSA (2 in, 2 out)  ← Input (ADC board)
  OR
  1 USB Audio Device, ALSA (2 in, 2 out)  ← Input (USB audio dongle)

Auto-detection (Sound Burger → Pi → Headphones):
  → Input (Sound Burger):  Device 1 - [ADC device name]
  → Output (Headphones):  Device 0 - snd_rpi_hifiberry_dac
```

### 6. Connect Hardware

1. **Mount innomaker HiFi DAC HAT** on Raspberry Pi GPIO header (for output)
2. **Mount ADC board** (HiFiBerry ADC+ or connect USB audio dongle) for input
3. **Connect Sound Burger**:
   - Sound Burger LINE OUT (3.5mm) → ADC input (3.5mm jack or USB dongle)
   - **Important**: Set Sound Burger to LINE mode (not PHONO)
4. **Connect Headphones**:
   - Option A: Plug into **innomaker DAC HAT 3.5mm headphone jack** (best quality)
   - Option B: Use RCA outputs from DAC HAT → RCA to 3.5mm adapter → headphones

### 7. Test Setup

```bash
# Test with auto-detection
python3 vinyl_stripper.py --remove vocals

# Or specify devices manually
python3 vinyl_stripper.py --input 1 --output 0 --remove vocals
```

**Expected behavior:**
- First 5-6 seconds: Silence (buffer filling)
- Then: Music plays with vocals removed
- Latency: ~5-6 seconds total

## Usage

### Basic Commands

```bash
# Remove vocals (default)
python3 vinyl_stripper.py --remove vocals

# Remove drums
python3 vinyl_stripper.py --remove drums

# Remove both vocals and drums
python3 vinyl_stripper.py --remove both

# Adjust chunk size for lower latency
python3 vinyl_stripper.py --remove vocals --chunk 3.0
```

### Auto-Start on Boot

Create systemd service:

```bash
sudo nano /etc/systemd/system/vinyl-stripper.service
```

Add:
```ini
[Unit]
Description=Vinyl Stem Stripper
After=sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/vinyl_stripper
ExecStart=/home/pi/vinyl_stripper/venv/bin/python3 /home/pi/vinyl_stripper/vinyl_stripper.py --remove vocals
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable vinyl-stripper
sudo systemctl start vinyl-stripper

# Check status
sudo systemctl status vinyl-stripper

# View logs
journalctl -u vinyl-stripper -f
```

## Troubleshooting

### HiFiBerry ADC+ Not Detected

1. **Check GPIO connection:**
   ```bash
   # Verify board is properly seated on GPIO header
   ```

2. **Check device tree overlay:**
   ```bash
   cat /boot/config.txt | grep dtoverlay
   # Should show: dtoverlay=hifiberry-dacplusadc
   ```

3. **Check I2S is enabled:**
   ```bash
   lsmod | grep snd_soc
   # Should show I2S modules loaded
   ```

4. **Test with arecord:**
   ```bash
   arecord -l
   # Should list HiFiBerry device
   ```

### No Sound Input

1. **Check Sound Burger settings:**
   - Must be set to **LINE OUT** (not PHONO)
   - Volume should be at reasonable level
   - Ensure 3.5mm cable is firmly connected to HiFiBerry ADC+ 3.5mm INPUT jack

2. **Verify HiFiBerry ADC+ connections:**
   - Board should be properly seated on GPIO header
   - 3.5mm input jack is on the HiFiBerry board itself (not the Pi)
   - Check that cable is plugged into INPUT jack (not output)

3. **Test input directly:**
   ```bash
   arecord -D hw:1,0 -f cd -d 5 test.wav
   # Play something on Sound Burger
   # Then play back:
   aplay test.wav
   ```

4. **Check input levels:**
   ```bash
   alsamixer
   # Select HiFiBerry device (F6)
   # Adjust input levels (may be labeled as "Capture" or "Input")
   ```

### No Sound Output

1. **Check headphone jack:**
   ```bash
   aplay -D hw:0,0 /usr/share/sounds/alsa/Front_Left.wav
   # Should hear sound in headphones
   ```

2. **Check volume:**
   ```bash
   alsamixer
   # Adjust headphone volume
   ```

3. **Verify device:**
   ```bash
   python3 vinyl_stripper.py --list-devices
   # Should show bcm2835 Headphones as output
   ```

### High Latency / Choppy Audio

1. **Reduce chunk size:**
   ```bash
   python3 vinyl_stripper.py --remove vocals --chunk 3.0
   ```

2. **Check CPU usage:**
   ```bash
   top
   # Should see Python process using ~80-100% CPU
   ```

3. **Check power supply:**
   - Ensure power bank provides 27W+ (5V/5A)
   - Low power can cause CPU throttling

## Alternative GPIO ADC Boards

### IQaudio DigiAMP+ with ADC
- Similar to HiFiBerry
- Uses different device tree overlay: `dtoverlay=iqaudio-dacplusadc`

### Generic I2S ADC Module
- Cheaper option (~$10-15)
- Requires manual I2S configuration
- May need custom device tree overlay

### Comparison

| Board | Cost | Quality | Ease of Setup |
|-------|------|---------|---------------|
| HiFiBerry ADC+ | $35-50 | Excellent | Easy |
| IQaudio DigiAMP+ | $40-55 | Excellent | Easy |
| Generic I2S ADC | $10-15 | Good | Moderate |

## Performance Optimization

### For Lower Latency

```bash
python3 vinyl_stripper.py --remove vocals --chunk 3.0
```

**Trade-off**: Smaller chunks = lower latency but more artifacts

### For Better Quality

```bash
python3 vinyl_stripper.py --remove vocals --chunk 6.0
```

**Trade-off**: Larger chunks = better quality but higher latency (~7 seconds)

## Power Management

### Battery Life

- **Expected**: 2-3 hours with 20,000mAh power bank
- **Optimization**: Use smaller chunks (3s) to reduce CPU load

### Low Power Mode

The code automatically uses optimized model (`htdemucs_ft`) on CPU, which is more power-efficient.

## Expected Performance

- **Latency**: 5-6 seconds (with 5s chunks)
- **CPU Usage**: 80-100% on Raspberry Pi 5
- **Memory**: ~2GB RAM usage
- **Quality**: Good (slightly lower than desktop `mdx_extra` but acceptable)
- **Battery**: 2-3 hours continuous use

## Cost Breakdown

### Option 1: High Quality (Recommended)
- Raspberry Pi 5 (8GB): $75
- innomaker HiFi DAC HAT: $30
- HiFiBerry ADC+: $35-50
- Power Bank: $40
- MicroSD Card: $10
- Cables: $5
- **Total**: ~$195-210

### Option 2: Budget
- Raspberry Pi 5 (8GB): $75
- innomaker HiFi DAC HAT: $30
- USB Audio Dongle (for input): $5-10
- Power Bank: $40
- MicroSD Card: $10
- Cables: $5
- **Total**: ~$165-170

**Note**: innomaker DAC HAT provides much better output quality (112dB SNR) than Pi's built-in audio!

## Next Steps

1. ✅ Test basic functionality
2. ⚠️ Add physical controls (buttons/knobs)
3. ⚠️ Add OLED display for status
4. ⚠️ Design 3D printed enclosure
5. ⚠️ Optimize for production

## Support

For issues or questions:
- Check logs: `journalctl -u vinyl-stripper -f`
- Test audio: `python3 vinyl_stripper.py --list-devices`
- Verify hardware: `arecord -l` and `aplay -l`
