# Git Workflow for Mac/Pi Development

## Recommended Approach

### 1. Commit Current Mac Optimizations
```bash
# Add new optimized files
git add test_spleeter_new.py benchmark_spleeter.py
git add .gitignore  # Update if needed

# Commit with descriptive message
git commit -m "Add optimized Spleeter implementation for Mac (0.5x real-time)"

# Push to remote
git push origin main
```

### 2. Create Feature Branch for Pi Port
```bash
# Create and switch to Pi-specific branch
git checkout -b pi-port

# Make Pi-specific changes
# ... edit files for Pi compatibility ...

# Commit Pi changes
git commit -m "Port optimizations to Raspberry Pi 5"

# Push branch
git push origin pi-port
```

### 3. Handling Platform Differences

**Option A: Platform Detection (Recommended)**
- Use existing `IS_RASPBERRY_PI` detection
- Keep same codebase, use conditionals for platform-specific code
- Example: `chunk_duration = 3.0 if not IS_RASPBERRY_PI else 2.5`

**Option B: Separate Scripts**
- Keep `test_spleeter_new.py` for Mac
- Create `test_spleeter_pi.py` for Pi (based on Mac version)
- Share common code via imports

**Option C: Configuration File**
- Create `config.py` with platform-specific settings
- Auto-detect platform and load appropriate config
- Easy to override for testing

### 4. Debugging Strategy

**On Mac (Development):**
```bash
# Use optimized script
python3 test_spleeter_new.py

# Benchmark performance
python3 benchmark_spleeter.py
```

**On Pi (Testing):**
```bash
# SSH to Pi
ssh pi@raspberrypi

# Pull latest code
cd vinyl_stripper
git pull origin pi-port  # or main

# Test with Pi-specific script
python3 test_spleeter_pi.py  # or use platform detection
```

**Sync Changes:**
```bash
# On Mac: commit and push
git add .
git commit -m "Fix: ..."
git push

# On Pi: pull changes
git pull
```

### 5. Branch Strategy

- `main`: Stable, tested code (Mac-optimized)
- `pi-port`: Pi-specific porting work
- `feature/*`: New features
- `fix/*`: Bug fixes

### 6. Testing Workflow

1. Develop/test on Mac (faster iteration)
2. Commit working Mac code
3. Switch to `pi-port` branch
4. Adapt for Pi (chunk sizes, buffers, etc.)
5. Test on Pi hardware
6. Merge back to main when stable

