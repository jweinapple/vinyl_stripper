#!/bin/bash
# Setup script for Python 3.11 on Raspberry Pi
# Raspberry Pi OS Bookworm ships with Python 3.13, which is incompatible with TensorFlow/Spleeter
# This script provides two options: pyenv (recommended) or apt backports (if available)

set -e

echo "=========================================="
echo "Python 3.11 Setup for Raspberry Pi"
echo "=========================================="
echo ""
echo "Raspberry Pi OS Bookworm ships with Python 3.13, which causes TensorFlow errors."
echo "Python 3.11 is required for Spleeter to work properly."
echo ""
echo "Installation methods:"
echo "  A) pyenv (recommended) - Builds from source, works everywhere (~1-2 hours)"
echo "  B) apt backports - Faster if available, but may not work on all systems"
echo ""
read -p "Choose method (A/B) [A]: " method
method=${method:-A}

if [[ ! "$method" =~ ^[AaBb]$ ]]; then
    echo "Invalid choice. Using pyenv (option A)."
    method="A"
fi

if [[ "$method" =~ ^[Bb]$ ]]; then
    echo ""
    echo "Attempting to install Python 3.11 via apt backports..."
    sudo apt update
    
    # Try to install from backports or testing
    if sudo apt install -y python3.11 python3.11-venv python3.11-dev 2>/dev/null; then
        echo "✓ Python 3.11 installed via apt"
        PYTHON_CMD="python3.11"
    else
        echo "Python 3.11 not available via apt. Falling back to pyenv..."
        method="A"
    fi
fi

if [[ "$method" =~ ^[Aa]$ ]]; then
    echo ""
    echo "Using pyenv method (will build from source)"
    echo "This will take 1-2 hours - please be patient!"
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
    echo "Building Python 3.11 from source..."
    echo "This will take 1-2 hours - please be patient!"
    pyenv install 3.11.9

    # Set as local version
    cd ~/vinyl_stripper
    pyenv local 3.11.9
    PYTHON_CMD="python3.11"
fi

# Create new virtual environment
echo ""
echo "Creating virtual environment with Python 3.11..."
cd ~/vinyl_stripper
rm -rf venv311
$PYTHON_CMD -m venv venv311
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

