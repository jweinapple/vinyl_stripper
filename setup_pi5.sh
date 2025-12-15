#!/bin/bash
#
# Vinyl Stripper - Raspberry Pi 5 Setup Script
#
# This script sets up all dependencies for running vinyl stem separation
# on a Raspberry Pi 5, including optional Hailo AI Kit support.
#
# Usage:
#   chmod +x setup_pi5.sh
#   ./setup_pi5.sh
#   ./setup_pi5.sh --hailo  # Also setup Hailo AI Kit
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

SETUP_HAILO=false
SETUP_SPLEETER=true
SETUP_DEMUCS=false  # Skip by default on Pi (too slow)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hailo)
            SETUP_HAILO=true
            shift
            ;;
        --demucs)
            SETUP_DEMUCS=true
            shift
            ;;
        --no-spleeter)
            SETUP_SPLEETER=false
            shift
            ;;
        --full)
            SETUP_HAILO=true
            SETUP_DEMUCS=true
            SETUP_SPLEETER=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --hailo       Setup Hailo AI Kit support"
            echo "  --demucs      Also install Demucs (slow on Pi CPU)"
            echo "  --no-spleeter Skip Spleeter installation"
            echo "  --full        Install everything"
            echo "  --help        Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "  Vinyl Stripper - Raspberry Pi 5 Setup"
echo "=============================================="
echo ""

# Check if running on Pi
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    log_info "Detected: $MODEL"
else
    log_warn "Not running on Raspberry Pi. Script designed for Pi 5."
fi

# Check for Pi 5 specifically
if [[ "$MODEL" == *"Pi 5"* ]]; then
    log_success "Raspberry Pi 5 detected"
else
    log_warn "Not a Pi 5. Some features may not work."
fi

# ============================================
# PHASE 1: System Dependencies
# ============================================
echo ""
log_info "Phase 1: System dependencies"

sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libasound2-dev \
    git

log_success "System dependencies installed"

# ============================================
# PHASE 2: Python Virtual Environment
# ============================================
echo ""
log_info "Phase 2: Python virtual environment"

VENV_DIR="$HOME/vinyl_stripper/venv"
mkdir -p "$HOME/vinyl_stripper"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    log_success "Created virtual environment at $VENV_DIR"
else
    log_info "Virtual environment already exists"
fi

source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# ============================================
# PHASE 3: Core Python Dependencies
# ============================================
echo ""
log_info "Phase 3: Core Python dependencies"

pip install numpy scipy
pip install sounddevice soundfile
pip install psutil

log_success "Core dependencies installed"

# ============================================
# PHASE 4: Spleeter (Fast Backend)
# ============================================
if [ "$SETUP_SPLEETER" = true ]; then
    echo ""
    log_info "Phase 4: Installing Spleeter (fast backend)"

    # Spleeter uses TensorFlow which can be heavy
    # Try tensorflow-lite or tflite-runtime for Pi

    # First try tflite
    pip install tensorflow-cpu || {
        log_warn "TensorFlow CPU failed, trying alternatives..."

        # Try installing from piwheels (Pi-optimized)
        pip install tensorflow --extra-index-url https://www.piwheels.org/simple/ || {
            log_warn "TensorFlow installation complex on Pi"
            log_warn "Spleeter may not work without TensorFlow"
        }
    }

    pip install spleeter || {
        log_warn "Spleeter installation failed"
        log_warn "You can still use Demucs or ONNX backends"
    }

    # Pre-download Spleeter model
    log_info "Pre-downloading Spleeter model..."
    python3 -c "
try:
    from spleeter.separator import Separator
    sep = Separator('spleeter:4stems')
    print('Spleeter model ready')
except Exception as e:
    print(f'Spleeter setup: {e}')
" || true

    log_success "Spleeter setup complete"
fi

# ============================================
# PHASE 5: ONNX Runtime (Optimized Inference)
# ============================================
echo ""
log_info "Phase 5: Installing ONNX Runtime"

# ONNX Runtime has ARM64 builds
pip install onnxruntime

log_success "ONNX Runtime installed"

# ============================================
# PHASE 6: Demucs (Optional - Slow on Pi)
# ============================================
if [ "$SETUP_DEMUCS" = true ]; then
    echo ""
    log_info "Phase 6: Installing Demucs (optional, slow on CPU)"

    # PyTorch for ARM64
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install demucs

    log_success "Demucs installed (note: very slow on Pi CPU)"
fi

# ============================================
# PHASE 7: Hailo AI Kit (Optional)
# ============================================
if [ "$SETUP_HAILO" = true ]; then
    echo ""
    log_info "Phase 7: Setting up Hailo AI Kit"

    # Check if Hailo device exists
    if [ -e /dev/hailo0 ]; then
        log_success "Hailo device detected at /dev/hailo0"
    else
        log_warn "Hailo device not found. Make sure AI Kit is installed."
        log_info "Install Hailo driver with:"
        log_info "  sudo apt install hailo-all"
    fi

    # Install Hailo SDK
    sudo apt-get install -y hailo-all || {
        log_warn "Hailo packages not in apt. Adding Hailo repo..."

        # Add Hailo repository
        wget -qO- https://hailo.ai/developer-zone/software-downloads/repo-key | sudo apt-key add -
        echo "deb https://hailo.ai/developer-zone/software-downloads/debian bookworm main" | \
            sudo tee /etc/apt/sources.list.d/hailo.list

        sudo apt-get update
        sudo apt-get install -y hailo-all || log_warn "Hailo installation requires manual setup"
    }

    # Install Hailo Python package
    pip install hailo-platform || {
        log_warn "Hailo Python package not available via pip"
        log_info "See: https://hailo.ai/developer-zone/"
    }

    # Test Hailo
    hailortcli fw-control identify || log_warn "Hailo device not responding"

    log_success "Hailo setup complete"
    echo ""
    log_info "NOTE: To use Hailo with Demucs, you need to convert the model to HEF format."
    log_info "This requires the Hailo Dataflow Compiler and model conversion."
    log_info "See: https://hailo.ai/developer-zone/documentation/"
fi

# ============================================
# PHASE 8: Configure Audio
# ============================================
echo ""
log_info "Phase 8: Configuring audio"

# Add user to audio group
sudo usermod -a -G audio $USER

# Create ALSA config for USB audio
if [ ! -f "$HOME/.asoundrc" ]; then
    log_info "Creating ALSA configuration..."

    # Detect USB audio card
    USB_CARD=$(aplay -l 2>/dev/null | grep -i "usb\|codec" | head -1 | sed -n 's/card \([0-9]\).*/\1/p')

    if [ -n "$USB_CARD" ]; then
        cat > "$HOME/.asoundrc" << EOF
# USB Audio Configuration for Vinyl Stripper
pcm.!default {
    type hw
    card $USB_CARD
}

ctl.!default {
    type hw
    card $USB_CARD
}
EOF
        log_success "ALSA configured for USB audio (card $USB_CARD)"
    else
        log_warn "No USB audio device detected"
        log_info "Connect your USB audio interface and re-run this script"
    fi
fi

# ============================================
# PHASE 9: Swap Space (for memory-intensive processing)
# ============================================
echo ""
log_info "Phase 9: Configuring swap space"

# Check current swap
CURRENT_SWAP=$(free -m | awk '/^Swap:/ {print $2}')
log_info "Current swap: ${CURRENT_SWAP}MB"

if [ "$CURRENT_SWAP" -lt 2000 ]; then
    log_info "Increasing swap space to 2GB for ML processing..."

    # Disable existing swap
    sudo dphys-swapfile swapoff || true

    # Configure 2GB swap
    sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile

    # Enable swap
    sudo dphys-swapfile setup
    sudo dphys-swapfile swapon

    log_success "Swap configured to 2GB"
else
    log_success "Swap already adequate (${CURRENT_SWAP}MB)"
fi

# ============================================
# PHASE 10: Create Launcher Script
# ============================================
echo ""
log_info "Phase 10: Creating launcher script"

cat > "$HOME/vinyl_stripper/run.sh" << 'EOF'
#!/bin/bash
# Vinyl Stripper Launcher

cd "$(dirname "$0")"
source venv/bin/activate

# Default to Spleeter on Pi (faster)
BACKEND="spleeter"

# Check for arguments
if [[ "$1" == "--demucs" ]]; then
    BACKEND="demucs"
    shift
elif [[ "$1" == "--onnx" ]]; then
    BACKEND="onnx"
    shift
elif [[ "$1" == "--hailo" ]]; then
    BACKEND="hailo"
    shift
fi

echo "Starting Vinyl Stripper (backend: $BACKEND)"
python3 vinyl_stripper_pi.py --backend "$BACKEND" --pi-mode "$@"
EOF

chmod +x "$HOME/vinyl_stripper/run.sh"
log_success "Launcher script created at ~/vinyl_stripper/run.sh"

# ============================================
# PHASE 11: Copy Script Files
# ============================================
echo ""
log_info "Phase 11: Copying script files"

# Copy the Pi-optimized script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/vinyl_stripper_pi.py" ]; then
    cp "$SCRIPT_DIR/vinyl_stripper_pi.py" "$HOME/vinyl_stripper/"
    log_success "Copied vinyl_stripper_pi.py"
fi

if [ -f "$SCRIPT_DIR/vinyl_stripper.py" ]; then
    cp "$SCRIPT_DIR/vinyl_stripper.py" "$HOME/vinyl_stripper/"
    log_success "Copied vinyl_stripper.py"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Installed backends:"
[ "$SETUP_SPLEETER" = true ] && echo "  - Spleeter (fast, recommended for Pi)"
echo "  - ONNX Runtime (optimized inference)"
[ "$SETUP_DEMUCS" = true ] && echo "  - Demucs (high quality, slow on CPU)"
[ "$SETUP_HAILO" = true ] && echo "  - Hailo AI Kit (requires model conversion)"
echo ""
echo "Usage:"
echo "  cd ~/vinyl_stripper"
echo "  ./run.sh --remove vocals"
echo ""
echo "Or directly:"
echo "  source ~/vinyl_stripper/venv/bin/activate"
echo "  python3 vinyl_stripper_pi.py --backend spleeter --pi-mode --remove vocals"
echo ""
echo "List audio devices:"
echo "  python3 vinyl_stripper_pi.py --list-devices"
echo ""

if [ "$SETUP_HAILO" = true ]; then
    echo "Hailo AI Kit:"
    echo "  To use hardware acceleration, you need to convert Demucs to HEF format."
    echo "  This requires the Hailo Dataflow Compiler."
    echo "  See: https://hailo.ai/developer-zone/documentation/dataflow-compiler/"
    echo ""
fi

echo "For best results on Raspberry Pi 5:"
echo "  1. Use Spleeter backend (--backend spleeter)"
echo "  2. Connect USB audio interface (UFO202, etc.)"
echo "  3. Allow 30-60 seconds for buffer to fill"
echo "  4. Consider Hailo AI Kit for faster processing"
echo ""
log_success "Setup complete! Reboot recommended."
