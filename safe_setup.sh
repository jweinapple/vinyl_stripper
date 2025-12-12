#!/bin/bash
# Safe setup script for Raspberry Pi / Edge Device
# Phased installation with pre-flight checks and error recovery

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Checkpoint file to track progress
CHECKPOINT_FILE=".setup_checkpoint"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if checkpoint exists and read it
get_checkpoint() {
    if [ -f "$CHECKPOINT_FILE" ]; then
        cat "$CHECKPOINT_FILE"
    else
        echo "0"
    fi
}

# Save checkpoint
save_checkpoint() {
    echo "$1" > "$CHECKPOINT_FILE"
}

# Pre-flight checks
preflight_checks() {
    log "Running pre-flight checks..."
    
    # Check disk space (need at least 5GB free)
    AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 5 ]; then
        log_error "Insufficient disk space: ${AVAILABLE_SPACE}GB available (need 5GB+)"
        exit 1
    fi
    log_success "Disk space: ${AVAILABLE_SPACE}GB available"
    
    # Check memory (need at least 1GB free)
    if command -v free &> /dev/null; then
        FREE_MEM=$(free -m | awk '/^Mem:/{print $7}')
        if [ "$FREE_MEM" -lt 1024 ]; then
            log_warning "Low memory: ${FREE_MEM}MB free (recommend 1GB+)"
        else
            log_success "Memory: ${FREE_MEM}MB free"
        fi
    fi
    
    # Check network connectivity
    if ! ping -c 1 -W 2 pypi.org &> /dev/null; then
        log_error "No network connectivity to pypi.org"
        exit 1
    fi
    log_success "Network connectivity OK"
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found"
        exit 1
    fi
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    log_success "Python version: $PYTHON_VERSION"
    
    echo ""
}

# Detect platform
detect_platform() {
    if [[ "$(uname -m)" == "aarch64" ]] || [[ "$(uname -m)" == "armv7l" ]]; then
        PLATFORM="ARM"
        log "Detected ARM platform (Raspberry Pi)"
    elif [[ "$(uname)" == "Darwin" ]]; then
        PLATFORM="MAC"
        log "Detected macOS"
    else
        PLATFORM="LINUX"
        log "Detected Linux"
    fi
    echo "$PLATFORM"
}

# Phase 1: System dependencies
phase1_system_deps() {
    log "Phase 1: Installing system dependencies..."
    
    if [[ "$PLATFORM" == "ARM" ]] || [[ "$PLATFORM" == "LINUX" ]]; then
        log "Updating package lists..."
        sudo apt-get update -qq
        
        log "Installing system packages..."
        sudo apt-get install -y \
            python3-pip \
            python3-venv \
            portaudio19-dev \
            libsndfile1 \
            build-essential \
            > /dev/null 2>&1
        
        log_success "System dependencies installed"
    else
        log_success "Skipping system dependencies (not ARM/Linux)"
    fi
    
    save_checkpoint "1"
    echo ""
}

# Phase 2: Virtual environment
phase2_venv() {
    log "Phase 2: Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        log "Creating venv (this may take a minute)..."
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_success "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    log "Upgrading pip..."
    pip3 install --upgrade pip --quiet --no-cache-dir
    
    save_checkpoint "2"
    echo ""
}

# Phase 3: PyTorch installation (most critical)
phase3_pytorch() {
    log "Phase 3: Installing PyTorch (this is the largest package, ~500MB)..."
    log "This may take 5-10 minutes. Please be patient..."
    
    source venv/bin/activate
    
    if [[ "$PLATFORM" == "ARM" ]]; then
        # ARM64 PyTorch - install with verbose output and no cache
        log "Installing PyTorch CPU-only for ARM..."
        
        # Install PyTorch with timeout and retry
        if ! timeout 1800 pip3 install \
            --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cpu \
            torch torchaudio \
            2>&1 | tee /tmp/pytorch_install.log; then
            log_error "PyTorch installation failed or timed out"
            log "Check /tmp/pytorch_install.log for details"
            exit 1
        fi
        
        log_success "PyTorch installed"
    else
        log "Installing PyTorch for $PLATFORM..."
        pip3 install --no-cache-dir torch torchaudio
        log_success "PyTorch installed"
    fi
    
    # Verify PyTorch installation
    log "Verifying PyTorch installation..."
    if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        log_success "PyTorch verification successful"
    else
        log_error "PyTorch verification failed"
        exit 1
    fi
    
    save_checkpoint "3"
    echo ""
}

# Phase 4: Other Python dependencies
phase4_dependencies() {
    log "Phase 4: Installing other Python dependencies..."
    
    source venv/bin/activate
    
    # Install dependencies one by one for better error tracking
    log "Installing sounddevice..."
    pip3 install --no-cache-dir sounddevice || { log_error "sounddevice installation failed"; exit 1; }
    
    log "Installing numpy..."
    pip3 install --no-cache-dir numpy || { log_error "numpy installation failed"; exit 1; }
    
    log "Installing soundfile..."
    pip3 install --no-cache-dir soundfile || { log_error "soundfile installation failed"; exit 1; }
    
    log "Installing demucs..."
    pip3 install --no-cache-dir demucs || { log_error "demucs installation failed"; exit 1; }
    
    log_success "All dependencies installed"
    
    save_checkpoint "4"
    echo ""
}

# Phase 5: Model download (with retry)
phase5_model() {
    log "Phase 5: Pre-downloading model (this may take a few minutes)..."
    
    source venv/bin/activate
    
    # Try downloading model with retry logic
    MAX_RETRIES=3
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if python3 -c "
from demucs.pretrained import get_model
import sys
try:
    print('Downloading htdemucs_ft model...')
    model = get_model('htdemucs_ft')
    print('✓ Model downloaded successfully')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
            log_success "Model downloaded"
            save_checkpoint "5"
            echo ""
            return 0
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                log_warning "Model download failed, retrying ($RETRY_COUNT/$MAX_RETRIES)..."
                sleep 5
            else
                log_error "Model download failed after $MAX_RETRIES attempts"
                log_warning "You can download the model later when running the script"
                return 1
            fi
        fi
    done
}

# Phase 6: Systemd service
phase6_systemd() {
    log "Phase 6: Creating systemd service..."
    
    if [[ "$PLATFORM" == "ARM" ]] || [[ "$PLATFORM" == "LINUX" ]]; then
        SERVICE_FILE="/etc/systemd/system/vinyl-stripper.service"
        WORK_DIR=$(pwd)
        
        sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Vinyl Stem Stripper
After=sound.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
Environment="PATH=$WORK_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$WORK_DIR/venv/bin/python3 $WORK_DIR/vinyl_stripper.py --input 1 --output 1 --remove vocals
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
        
        log_success "Systemd service created"
        log "  Enable with: sudo systemctl enable vinyl-stripper"
        log "  Start with: sudo systemctl start vinyl-stripper"
    else
        log_success "Skipping systemd service (not ARM/Linux)"
    fi
    
    save_checkpoint "6"
    echo ""
}

# Main installation function
main() {
    echo "=========================================="
    echo "Vinyl Stripper - Safe Setup"
    echo "=========================================="
    echo ""
    
    # Detect platform
    PLATFORM=$(detect_platform)
    
    # Check if resuming from checkpoint
    CHECKPOINT=$(get_checkpoint)
    if [ "$CHECKPOINT" != "0" ]; then
        log_warning "Resuming from checkpoint: $CHECKPOINT"
        read -p "Continue from checkpoint? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$CHECKPOINT_FILE"
            CHECKPOINT="0"
        fi
    fi
    
    # Run pre-flight checks
    preflight_checks
    
    # Run phases based on checkpoint
    if [ "$CHECKPOINT" -lt "1" ]; then phase1_system_deps; fi
    if [ "$CHECKPOINT" -lt "2" ]; then phase2_venv; fi
    if [ "$CHECKPOINT" -lt "3" ]; then phase3_pytorch; fi
    if [ "$CHECKPOINT" -lt "4" ]; then phase4_dependencies; fi
    if [ "$CHECKPOINT" -lt "5" ]; then phase5_model; fi
    if [ "$CHECKPOINT" -lt "6" ]; then phase6_systemd; fi
    
    # Cleanup checkpoint file
    rm -f "$CHECKPOINT_FILE"
    
    echo ""
    echo "=========================================="
    log_success "Setup complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. List audio devices: python3 vinyl_stripper.py --list-devices"
    echo "3. Test with: python3 vinyl_stripper.py --input X --output Y --remove vocals"
    echo ""
    if [[ "$PLATFORM" == "ARM" ]] || [[ "$PLATFORM" == "LINUX" ]]; then
        echo "To enable auto-start:"
        echo "  sudo systemctl enable vinyl-stripper"
        echo "  sudo systemctl start vinyl-stripper"
        echo ""
    fi
}

# Run main function
main
