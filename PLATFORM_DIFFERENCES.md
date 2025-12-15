# Handling Platform Differences: Mac vs Raspberry Pi

## Strategy: Platform Detection with Conditional Code

### Recommended Approach

Use **platform detection** to keep a single codebase that adapts automatically:

```python
# Detect platform
IS_RASPBERRY_PI = os.path.exists('/proc/device-tree/model') and \
                  'raspberry pi' in open('/proc/device-tree/model').read().lower()

# Platform-specific settings
if IS_RASPBERRY_PI:
    CHUNK_DURATION = 2.5  # Pi needs smaller chunks
    MIN_BUFFER = 15.0     # Pi needs larger buffer
    NUM_WORKERS = 4       # Pi 5 has 4 cores
else:
    CHUNK_DURATION = 3.0  # Mac can handle larger chunks
    MIN_BUFFER = 8.0      # Mac needs smaller buffer
    NUM_WORKERS = 1       # Mac uses single worker
```

### Implementation in test_spleeter_new.py

Add platform detection at the top:

```python
# Platform detection
IS_RASPBERRY_PI = os.path.exists('/proc/device-tree/model') and \
                  'raspberry pi' in open('/proc/device-tree/model').read().lower() \
                  if os.path.exists('/proc/device-tree/model') else False

# Platform-specific configuration
if IS_RASPBERRY_PI:
    DEFAULT_CHUNK_DURATION = 2.5
    DEFAULT_MIN_BUFFER = 15.0
    DEFAULT_LOW_THRESHOLD = 5.0
else:
    DEFAULT_CHUNK_DURATION = 3.0
    DEFAULT_MIN_BUFFER = 8.0
    DEFAULT_LOW_THRESHOLD = 2.0
```

### Device Detection Differences

**Mac:**
- Universal Audio: Device 5
- Mac Speakers: Device 2

**Pi:**
- USB Audio: Auto-detected via `find_usb_audio_device()`
- Output: HiFi DAC HAT or headphone jack

### Code Structure

Keep platform-specific code in same file with clear conditionals:

```python
class SpleeterProcessor:
    def __init__(self, ...):
        # Platform-specific chunk size
        self.chunk_duration = DEFAULT_CHUNK_DURATION
        
        # Platform-specific buffer
        self.min_buffer_samples = int(DEFAULT_MIN_BUFFER * self.sample_rate)
        
        # Platform-specific device detection
        if IS_RASPBERRY_PI:
            input_device = find_usb_audio_device()
            output_device = find_pi_output_device()
        else:
            input_device = find_universal_audio_device()
            output_device = find_mac_speakers()
```

## Debugging Workflow

### On Mac (Development)

1. **Test locally** with optimized settings
2. **Benchmark** performance
3. **Commit** working code
4. **Push** to git

```bash
# Test
python3 test_spleeter_new.py

# Benchmark
python3 benchmark_spleeter.py

# Commit
git add .
git commit -m "Optimize: ..."
git push
```

### On Pi (Testing)

1. **Pull** latest code
2. **Test** with Pi-specific settings (auto-detected)
3. **Debug** any issues
4. **Commit** Pi-specific fixes

```bash
# Pull latest
git pull origin main

# Test (platform detection handles differences)
python3 test_spleeter_new.py

# If issues, debug and commit fixes
git add .
git commit -m "Fix Pi: adjust buffer size for stability"
git push
```

## Alternative: Separate Files

If platform differences are too complex, use separate files:

- `test_spleeter_new.py` - Mac optimized
- `test_spleeter_pi.py` - Pi optimized (based on Mac version)
- Share common code via `spleeter_common.py`

## Configuration File Approach

Create `config.py`:

```python
# config.py
import os

IS_PI = os.path.exists('/proc/device-tree/model')

if IS_PI:
    CHUNK_DURATION = 2.5
    MIN_BUFFER = 15.0
    # ... Pi settings
else:
    CHUNK_DURATION = 3.0
    MIN_BUFFER = 8.0
    # ... Mac settings
```

Then import in your script:

```python
from config import CHUNK_DURATION, MIN_BUFFER
```

## Git Branch Strategy

### Recommended: Feature Branch

```bash
# On Mac: commit optimized code
git checkout main
git add test_spleeter_new.py
git commit -m "Add optimized Mac implementation"
git push

# Create Pi port branch
git checkout -b pi-port

# Make Pi-specific changes
# ... edit files ...

# Commit Pi changes
git commit -m "Port to Pi: adjust for slower hardware"
git push origin pi-port
```

### Testing Both Platforms

```bash
# Mac: test on main branch
git checkout main
python3 test_spleeter_new.py

# Pi: test on pi-port branch
git checkout pi-port
python3 test_spleeter_new.py
```

## Best Practices

1. **Use platform detection** - Single codebase is easier to maintain
2. **Clear conditionals** - Comment why settings differ
3. **Test on both** - Verify Mac and Pi work correctly
4. **Document differences** - Note platform-specific behavior
5. **Version control** - Use branches for Pi-specific work

## Common Differences to Handle

| Setting | Mac | Pi 5 |
|---------|-----|------|
| Chunk size | 3.0s | 2.0-2.5s |
| Buffer | 8s | 15-30s |
| Processing speed | 0.5x RT | 2-4x RT |
| Workers | 1 | 4 |
| Device detection | UA/Mac speakers | USB/HiFi DAC |

