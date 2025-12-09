#!/bin/bash
# Setup script for Raspberry Pi / Edge Device

set -e

echo "=========================================="
echo "Vinyl Stripper - Edge Device Setup"
echo "=========================================="
echo ""

# Detect platform
if [[ "$(uname -m)" == "aarch64" ]] || [[ "$(uname -m)" == "armv7l" ]]; then
    PLATFORM="ARM"
    echo "✓ Detected ARM platform (Raspberry Pi)"
elif [[ "$(uname)" == "Darwin" ]]; then
    PLATFORM="MAC"
    echo "✓ Detected macOS"
else
    PLATFORM="LINUX"
    echo "✓ Detected Linux"
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."
if [[ "$PLATFORM" == "ARM" ]] || [[ "$PLATFORM" == "LINUX" ]]; then
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-venv portaudio19-dev libsndfile1
    echo "✓ System dependencies installed"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip3 install --upgrade pip

# Install PyTorch (platform-specific)
echo ""
echo "Installing PyTorch..."
if [[ "$PLATFORM" == "ARM" ]]; then
    # ARM64 PyTorch (for Raspberry Pi)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    pip3 install torch torchaudio
fi

# Install other dependencies
echo ""
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Pre-download model
echo ""
echo "Pre-downloading model (this may take a few minutes)..."
python3 -c "from demucs.pretrained import get_model; get_model('htdemucs_ft'); print('✓ Model downloaded')"

# Create systemd service (Linux/ARM only)
if [[ "$PLATFORM" == "ARM" ]] || [[ "$PLATFORM" == "LINUX" ]]; then
    echo ""
    echo "Creating systemd service..."
    SERVICE_FILE="/etc/systemd/system/vinyl-stripper.service"
    sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Vinyl Stem Stripper
After=sound.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python3 $(pwd)/vinyl_stripper.py --input 1 --output 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    echo "✓ Systemd service created"
    echo "  Enable with: sudo systemctl enable vinyl-stripper"
    echo "  Start with: sudo systemctl start vinyl-stripper"
fi

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Connect USB audio interface"
echo "2. Run: python3 vinyl_stripper.py --list-devices"
echo "3. Test with: python3 vinyl_stripper.py --input X --output Y"
echo ""


