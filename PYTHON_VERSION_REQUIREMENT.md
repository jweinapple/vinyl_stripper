# Python Version Requirement for Spleeter on Raspberry Pi

## Problem

Spleeter requires TensorFlow, which has compatibility issues with Python 3.13:
- TensorFlow uses TensorFlow 1.x-style session API (`tf.compat.v1.Session`)
- Python 3.13 introduces changes that break TensorFlow's internal handling of NumPy arrays and Python objects
- This causes "Unsupported object type NoneType" errors during graph execution

## Solution: Use Python 3.11

**Python 3.11 is required for reliable Spleeter operation on Raspberry Pi.**

## Setup Instructions

### Option A: Automated Setup Script (Recommended)

Run the setup script on your Raspberry Pi:

```bash
cd ~/vinyl_stripper
chmod +x setup_python311_pi.sh
./setup_python311_pi.sh
```

The script offers two methods:
- **Method A (pyenv)**: Builds Python 3.11 from source (~1-2 hours) - works everywhere
- **Method B (apt)**: Attempts to install via apt backports - faster if available

**Note:** Building Python 3.11 from source takes 1-2 hours on a Raspberry Pi. The script will:
1. Install build dependencies
2. Install pyenv (if using method A)
3. Build/install Python 3.11.9
4. Create a new virtual environment (`venv311`)
5. Install all dependencies

### Option B: Manual pyenv Setup

If you prefer to set up pyenv manually:

If you prefer to do it manually:

```bash
# Install build dependencies
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash

# Add to ~/.bashrc
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Install Python 3.11
pyenv install 3.11.9

# Create virtual environment
cd ~/vinyl_stripper
pyenv local 3.11.9
python3.11 -m venv venv311
source venv311/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel

# Install in correct order to avoid conflicts
pip install "numpy>=1.19.0,<1.24.0"
pip install tensorflow==2.12.1
pip install tensorflow-estimator==2.12.0
pip install sounddevice
pip install spleeter==2.4.2
```

## Using the New Environment

After setup, always use the Python 3.11 environment:

```bash
cd ~/vinyl_stripper
source venv311/bin/activate  # Instead of venv/bin/activate
python3 test_spleeter_new.py
```

## Verification

Check your Python version:

```bash
python3 --version
# Should show: Python 3.11.9
```

## Troubleshooting

### If pyenv build fails:
- Ensure all build dependencies are installed
- Check available disk space (need ~500MB)
- Try: `pyenv install -v 3.11.9` for verbose output

### If TensorFlow still has issues:
- Ensure you're using `tensorflow==2.12.1` (required by Spleeter 2.4.2)
- Verify `tensorflow-estimator==2.12.0` is installed
- Check that numpy version is compatible (`>=1.19.0,<1.24.0`)
- Check that the TensorFlow compatibility patches in `test_spleeter_new.py` are active

### Dependency Conflicts:
If you see dependency conflicts:
- **Spleeter 2.4.2 requires TensorFlow 2.12.1** (not 2.13.0)
- Install packages in order: numpy → tensorflow → tensorflow-estimator → spleeter
- Use the setup script which handles this automatically

## Alternative: Deadsnakes PPA (Ubuntu only)

**Note:** Deadsnakes PPA is for Ubuntu. Raspberry Pi OS is Debian-based, so deadsnakes won't work directly.

If you're on Ubuntu (not Raspberry Pi OS), you can use deadsnakes:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

For Raspberry Pi OS, use **pyenv** (Option A in the setup script).

## Why Not Python 3.12?

Python 3.12 may work but Python 3.11 is the most tested and stable version with TensorFlow 2.13.0 and Spleeter.

