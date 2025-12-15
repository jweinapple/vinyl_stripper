#!/bin/bash
# Setup script for Python 3.11 on Raspberry Pi
# This will take a while - Python builds can take 1-2 hours on Pi

set -e

echo "=========================================="
echo "Python 3.11 Setup for Raspberry Pi"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Install build dependencies"
echo "  2. Install pyenv (if not already installed)"
echo "  3. Build Python 3.11 from source (~1-2 hours)"
echo "  4. Create a new virtual environment"
echo "  5. Install Spleeter and dependencies"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Install build dependencies
echo ""
echo "Installing build dependencies..."
sudo apt update
sudo apt install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

# Setup pyenv
if [ ! -d "$HOME/.pyenv" ]; then
    echo ""
    echo "Installing pyenv..."
    curl https://pyenv.run | bash
    
    # Add to shell config
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    
    echo "" >> ~/.bashrc
    echo "# Pyenv" >> ~/.bashrc
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
else
    echo "Pyenv already installed"
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
fi

# Install Python 3.11
echo ""
echo "Installing Python 3.11..."
echo "This will take 1-2 hours - please be patient!"
pyenv install 3.11.9

# Set as local version
cd ~/vinyl_stripper
pyenv local 3.11.9

# Create new virtual environment
echo ""
echo "Creating virtual environment with Python 3.11..."
rm -rf venv311
python3.11 -m venv venv311
source venv311/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install numpy sounddevice tensorflow==2.13.0 tensorflow-estimator spleeter

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To use the new environment:"
echo "  cd ~/vinyl_stripper"
echo "  source venv311/bin/activate"
echo "  python3 test_spleeter_new.py"
echo ""

