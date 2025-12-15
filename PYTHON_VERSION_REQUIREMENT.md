# Python Version Requirement for Spleeter on Raspberry Pi

## Problem

Spleeter requires TensorFlow, which has compatibility issues with Python 3.13:
- TensorFlow uses TensorFlow 1.x-style session API (`tf.compat.v1.Session`)
- Python 3.13 introduces changes that break TensorFlow's internal handling of NumPy arrays and Python objects
- This causes "Unsupported object type NoneType" errors during graph execution

## Solution: Use Python 3.11

**Python 3.11 is required for reliable Spleeter operation on Raspberry Pi.**

## Setup Instructions

### Option 1: Automated Setup (Recommended)

Run the setup script on your Raspberry Pi:

```bash
cd ~/vinyl_stripper
chmod +x setup_python311_pi.sh
./setup_python311_pi.sh
```

**Note:** Building Python 3.11 from source takes 1-2 hours on a Raspberry Pi. The script will:
1. Install build dependencies
2. Install pyenv
3. Build Python 3.11.9 from source
4. Create a new virtual environment (`venv311`)
5. Install all dependencies

### Option 2: Manual Setup

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
pip install --upgrade pip
pip install numpy sounddevice tensorflow==2.13.0 tensorflow-estimator spleeter
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
- Ensure you're using `tensorflow==2.13.0` (not 2.20.0)
- Verify `tensorflow-estimator` is installed
- Check that the TensorFlow compatibility patches in `test_spleeter_new.py` are active

## Why Not Python 3.12?

Python 3.12 may work but Python 3.11 is the most tested and stable version with TensorFlow 2.13.0 and Spleeter.

