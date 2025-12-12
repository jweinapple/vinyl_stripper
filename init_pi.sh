#!/bin/bash
# Initialize Raspberry Pi - Complete Setup Automation
# This script transfers files and sets up the Pi remotely

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PI_USER="${PI_USER:-pi}"
PI_HOST="${PI_HOST:-raspberrypi.local}"
PI_DIR="${PI_DIR:-~/vinyl_stripper}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Step 1: Check SSH connectivity
check_ssh() {
    log "Step 1: Checking SSH connectivity to ${PI_USER}@${PI_HOST}..."
    
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${PI_USER}@${PI_HOST}" "echo 'SSH connection successful'" 2>/dev/null; then
        log_success "SSH connection working"
        return 0
    else
        log_error "Cannot connect to ${PI_USER}@${PI_HOST}"
        log_warning "Troubleshooting steps:"
        echo "  1. Check Pi is powered on and on the network"
        echo "  2. Try: ping ${PI_HOST}"
        echo "  3. Check router admin for Pi's IP address"
        echo "  4. Try: ssh ${PI_USER}@<PI_IP_ADDRESS>"
        echo "  5. Run: bash troubleshoot_ssh.sh"
        return 1
    fi
}

# Step 2: Check Pi prerequisites
check_pi_prerequisites() {
    log "Step 2: Checking Pi prerequisites..."
    
    ssh "${PI_USER}@${PI_HOST}" bash << 'EOF'
        # Check Python
        if ! command -v python3 &> /dev/null; then
            echo "ERROR: python3 not found"
            exit 1
        fi
        
        # Check disk space (need at least 5GB)
        AVAILABLE=$(df -BG ~ | tail -1 | awk '{print $4}' | sed 's/G//')
        if [ "$AVAILABLE" -lt 5 ]; then
            echo "WARNING: Low disk space: ${AVAILABLE}GB (need 5GB+)"
        fi
        
        # Check network
        if ! ping -c 1 -W 2 pypi.org &> /dev/null; then
            echo "ERROR: No internet connectivity"
            exit 1
        fi
        
        echo "OK"
EOF
    
    if [ $? -eq 0 ]; then
        log_success "Pi prerequisites met"
        return 0
    else
        log_error "Pi prerequisites check failed"
        return 1
    fi
}

# Step 3: Transfer files to Pi
transfer_files() {
    log "Step 3: Transferring project files to Pi..."
    log "  Excluding: venv/, __pycache__, .git, .DS_Store"
    
    # Create rsync exclude file
    EXCLUDE_FILE=$(mktemp)
    cat > "$EXCLUDE_FILE" << 'EOF'
venv/
__pycache__/
*.pyc
*.pyo
*.pth
*.th
.git/
.DS_Store
*.log
.cache/
*.swp
*.swo
*~
EOF
    
    # Transfer files using rsync
    if rsync -avz --progress \
        --exclude-from="$EXCLUDE_FILE" \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='.DS_Store' \
        "${LOCAL_DIR}/" \
        "${PI_USER}@${PI_HOST}:${PI_DIR}/" 2>&1 | grep -v "venv\|__pycache__"; then
        log_success "Files transferred successfully"
        rm -f "$EXCLUDE_FILE"
        return 0
    else
        log_error "File transfer failed"
        rm -f "$EXCLUDE_FILE"
        return 1
    fi
}

# Step 4: Make scripts executable
make_executable() {
    log "Step 4: Making scripts executable on Pi..."
    
    ssh "${PI_USER}@${PI_HOST}" bash << EOF
        cd ${PI_DIR}
        chmod +x safe_setup.sh edge_setup.sh switch_mode.sh 2>/dev/null || true
        chmod +x *.sh 2>/dev/null || true
EOF
    
    log_success "Scripts made executable"
}

# Step 5: Run setup script
run_setup() {
    log "Step 5: Running setup script on Pi..."
    log_warning "This will take 10-15 minutes. Do not interrupt!"
    echo ""
    
    # Run setup script with output streaming
    ssh -t "${PI_USER}@${PI_HOST}" bash << EOF
        cd ${PI_DIR}
        echo "Starting setup..."
        ./safe_setup.sh
EOF
    
    SETUP_EXIT=$?
    
    if [ $SETUP_EXIT -eq 0 ]; then
        log_success "Setup completed successfully"
        return 0
    else
        log_error "Setup script exited with code $SETUP_EXIT"
        log_warning "You may need to run setup manually:"
        echo "  ssh ${PI_USER}@${PI_HOST}"
        echo "  cd ${PI_DIR}"
        echo "  ./safe_setup.sh"
        return 1
    fi
}

# Step 6: Verify installation
verify_installation() {
    log "Step 6: Verifying installation..."
    
    ssh "${PI_USER}@${PI_HOST}" bash << EOF
        cd ${PI_DIR}
        
        # Check venv exists
        if [ ! -d "venv" ]; then
            echo "ERROR: venv not found"
            exit 1
        fi
        
        # Check Python packages
        source venv/bin/activate
        if ! python3 -c "import torch; import demucs" 2>/dev/null; then
            echo "ERROR: Required packages not installed"
            exit 1
        fi
        
        echo "OK"
EOF
    
    if [ $? -eq 0 ]; then
        log_success "Installation verified"
        return 0
    else
        log_error "Installation verification failed"
        return 1
    fi
}

# Step 7: Test audio device detection
test_audio() {
    log "Step 7: Testing audio device detection..."
    
    ssh "${PI_USER}@${PI_HOST}" bash << EOF
        cd ${PI_DIR}
        source venv/bin/activate
        python3 vinyl_stripper.py --list-devices 2>&1 | head -20
EOF
    
    log_success "Audio device detection test completed"
}

# Main initialization function
main() {
    echo "=========================================="
    echo "Raspberry Pi Initialization"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  Pi User: ${PI_USER}"
    echo "  Pi Host: ${PI_HOST}"
    echo "  Pi Directory: ${PI_DIR}"
    echo "  Local Directory: ${LOCAL_DIR}"
    echo ""
    
    # Check if user wants to proceed
    read -p "Continue with initialization? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warning "Initialization cancelled"
        exit 0
    fi
    
    echo ""
    
    # Run steps
    if ! check_ssh; then
        log_error "Cannot proceed without SSH connection"
        exit 1
    fi
    
    if ! check_pi_prerequisites; then
        log_error "Pi prerequisites not met"
        exit 1
    fi
    
    if ! transfer_files; then
        log_error "File transfer failed"
        exit 1
    fi
    
    make_executable
    
    if ! run_setup; then
        log_error "Setup failed"
        exit 1
    fi
    
    if ! verify_installation; then
        log_error "Verification failed"
        exit 1
    fi
    
    test_audio
    
    echo ""
    echo "=========================================="
    log_success "Initialization complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Connect audio hardware"
    echo "  2. SSH into Pi: ssh ${PI_USER}@${PI_HOST}"
    echo "  3. Test: cd ${PI_DIR} && source venv/bin/activate"
    echo "  4. Run: python3 vinyl_stripper.py --list-devices"
    echo "  5. Start: python3 vinyl_stripper.py --remove vocals"
    echo ""
    echo "To enable auto-start:"
    echo "  ssh ${PI_USER}@${PI_HOST}"
    echo "  sudo systemctl enable vinyl-stripper"
    echo "  sudo systemctl start vinyl-stripper"
    echo ""
}

# Run main function
main
