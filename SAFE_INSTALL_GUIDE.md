# Safe Installation Guide for Raspberry Pi

## Overview

The `safe_setup.sh` script provides a safer, more reliable installation process with:
- ✅ Pre-flight checks (disk space, memory, network)
- ✅ Phased installation with checkpoints
- ✅ Error recovery and resume capability
- ✅ Progress monitoring
- ✅ Safer PyTorch installation
- ✅ Model download with retry logic

## Key Improvements Over Original Script

### 1. **Pre-flight Checks**
- Verifies 5GB+ disk space available
- Checks memory availability
- Tests network connectivity
- Validates Python installation

### 2. **Phased Installation**
Installation is split into 6 phases:
1. **System dependencies** - apt packages
2. **Virtual environment** - Python venv creation
3. **PyTorch** - Largest package (~500MB), installed separately
4. **Other dependencies** - sounddevice, numpy, demucs, etc.
5. **Model download** - Pre-downloads htdemucs_ft model
6. **Systemd service** - Creates auto-start service

### 3. **Checkpoint System**
- Saves progress after each phase
- Can resume from last successful phase if interrupted
- Prevents re-installing completed phases

### 4. **Better Error Handling**
- Timeout protection for PyTorch installation (30 min max)
- Retry logic for model downloads (3 attempts)
- Verbose logging for troubleshooting
- Graceful failure with helpful error messages

### 5. **Resource Management**
- Uses `--no-cache-dir` to save disk space
- Monitors memory during installation
- Lower priority installation (won't freeze system)

## Usage

### Basic Installation

```bash
# On Raspberry Pi
cd ~/vinyl_stripper
./safe_setup.sh
```

### Resuming After Interruption

If installation is interrupted (Ctrl+C, network failure, etc.):

```bash
# Script will detect checkpoint and ask to resume
./safe_setup.sh
# Answer 'y' to continue from last phase
```

### Starting Fresh

To start over from the beginning:

```bash
rm .setup_checkpoint
./safe_setup.sh
```

## Installation Phases Explained

### Phase 1: System Dependencies (~2-5 minutes)
- Updates package lists
- Installs: python3-pip, python3-venv, portaudio19-dev, libsndfile1
- **Low risk** - Standard apt packages

### Phase 2: Virtual Environment (~30 seconds)
- Creates Python virtual environment
- Upgrades pip
- **Low risk** - Fast operation

### Phase 3: PyTorch (~5-15 minutes) ⚠️ **HIGHEST RISK**
- Downloads and installs PyTorch (~500MB)
- CPU-only version for ARM
- **Why risky**: Large download, can timeout or exhaust resources
- **Safeguards**: 
  - 30-minute timeout
  - No cache to save disk space
  - Progress logging
  - Verification after installation

### Phase 4: Other Dependencies (~2-5 minutes)
- Installs: sounddevice, numpy, soundfile, demucs
- Installed one-by-one for better error tracking
- **Medium risk** - Smaller packages, but demucs can be large

### Phase 5: Model Download (~2-5 minutes)
- Pre-downloads htdemucs_ft model (~100-200MB)
- **Safeguards**: 3 retry attempts with delays
- **Note**: Can skip if network is unstable, model will download on first run

### Phase 6: Systemd Service (~10 seconds)
- Creates auto-start service file
- **Low risk** - Just file creation

## Monitoring Installation

### Check Progress

```bash
# View checkpoint status
cat .setup_checkpoint

# Monitor disk space during installation
watch -n 5 df -h

# Monitor memory
watch -n 5 free -h
```

### View Logs

```bash
# PyTorch installation log (if Phase 3 fails)
cat /tmp/pytorch_install.log

# System logs (if service fails)
journalctl -u vinyl-stripper -f
```

## Troubleshooting

### Installation Fails at Phase 3 (PyTorch)

**Symptoms**: Script hangs or times out during PyTorch installation

**Solutions**:
1. **Check network**: `ping pypi.org`
2. **Check disk space**: `df -h` (need 5GB+)
3. **Check memory**: `free -h` (close other apps)
4. **Try manual PyTorch install**:
   ```bash
   source venv/bin/activate
   pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchaudio
   ```
5. **Resume from Phase 4**: Edit `.setup_checkpoint` to `3` and rerun

### Installation Fails at Phase 5 (Model Download)

**Symptoms**: Model download fails after retries

**Solutions**:
1. **Skip for now**: Model will download automatically on first run
2. **Manual download**:
   ```bash
   source venv/bin/activate
   python3 -c "from demucs.pretrained import get_model; get_model('htdemucs_ft')"
   ```
3. **Use smaller model**: Edit script to use `htdemucs` instead of `htdemucs_ft`

### Out of Disk Space

**Symptoms**: Installation fails with "No space left on device"

**Solutions**:
1. **Clean pip cache**: `pip3 cache purge`
2. **Remove old checkpoints**: `rm .setup_checkpoint`
3. **Clean apt cache**: `sudo apt-get clean`
4. **Remove unused packages**: `sudo apt-get autoremove`
5. **Expand filesystem** (if using small SD card)

### System Becomes Unresponsive

**Symptoms**: Pi freezes during installation

**Solutions**:
1. **Wait**: PyTorch installation can take 10-15 minutes
2. **SSH from another terminal**: Check progress with `cat .setup_checkpoint`
3. **Reboot and resume**: Script will detect checkpoint and resume
4. **Install in tmux**: 
   ```bash
   tmux new -s install
   ./safe_setup.sh
   # Detach: Ctrl+B, then D
   # Reattach: tmux attach -t install
   ```

## Comparison: safe_setup.sh vs edge_setup.sh

| Feature | edge_setup.sh | safe_setup.sh |
|---------|---------------|---------------|
| Pre-flight checks | ❌ | ✅ |
| Phased installation | ❌ | ✅ |
| Checkpoint/resume | ❌ | ✅ |
| Error recovery | ❌ | ✅ |
| Progress monitoring | ❌ | ✅ |
| Timeout protection | ❌ | ✅ |
| Retry logic | ❌ | ✅ |
| Disk space checks | ❌ | ✅ |
| Memory monitoring | ❌ | ✅ |

## Expected Installation Time

On Raspberry Pi 5 with good network:
- **Phase 1**: 2-5 minutes
- **Phase 2**: 30 seconds
- **Phase 3**: 5-15 minutes ⚠️ (longest)
- **Phase 4**: 2-5 minutes
- **Phase 5**: 2-5 minutes
- **Phase 6**: 10 seconds

**Total**: ~15-30 minutes (mostly Phase 3)

## Disk Space Requirements

- System dependencies: ~100MB
- Virtual environment: ~50MB
- PyTorch: ~500MB
- Other packages: ~200MB
- Model: ~200MB
- **Total**: ~1GB (with 5GB recommended free space for safety)

## Best Practices

1. **Run in tmux/screen** for long-running installation:
   ```bash
   tmux new -s install
   ./safe_setup.sh
   ```

2. **Monitor resources** in another terminal:
   ```bash
   watch -n 5 'df -h; free -h'
   ```

3. **Check network stability** before starting:
   ```bash
   ping -c 10 pypi.org
   ```

4. **Ensure stable power** - Use good USB-C power supply (27W+)

5. **Don't interrupt Phase 3** - Let PyTorch installation complete

## Next Steps After Installation

1. **Test installation**:
   ```bash
   source venv/bin/activate
   python3 vinyl_stripper.py --list-devices
   ```

2. **Test audio processing**:
   ```bash
   python3 vinyl_stripper.py --input X --output Y --remove vocals
   ```

3. **Enable auto-start** (optional):
   ```bash
   sudo systemctl enable vinyl-stripper
   sudo systemctl start vinyl-stripper
   ```

4. **Check service status**:
   ```bash
   sudo systemctl status vinyl-stripper
   ```
