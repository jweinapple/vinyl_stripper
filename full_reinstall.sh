#!/bin/bash
# Full reinstall: Cleanup + Reinstall
# Run this on the Raspberry Pi

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Full Reinstall: Cleanup + Setup"
echo "=========================================="
echo ""

# ===== CLEANUP PHASE =====
log "=== CLEANUP PHASE ==="

# Stop service
log "Stopping vinyl-stripper service..."
if systemctl is-active --quiet vinyl-stripper 2>/dev/null; then
    sudo systemctl stop vinyl-stripper 2>/dev/null || true
    log_success "Service stopped"
fi

if systemctl is-enabled --quiet vinyl-stripper 2>/dev/null; then
    sudo systemctl disable vinyl-stripper 2>/dev/null || true
    log_success "Service disabled"
fi

# Remove service file
log "Removing systemd service file..."
if [ -f "/etc/systemd/system/vinyl-stripper.service" ]; then
    sudo rm -f /etc/systemd/system/vinyl-stripper.service
    sudo systemctl daemon-reload
    log_success "Service file removed"
fi

# Remove venv
log "Removing virtual environment..."
if [ -d "venv" ]; then
    rm -rf venv
    log_success "Virtual environment removed"
fi

# Remove checkpoint
log "Removing checkpoint file..."
rm -f .setup_checkpoint 2>/dev/null || true

# Remove cache
log "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove logs
log "Removing log files..."
find . -type f -name "*.log" -delete 2>/dev/null || true

# Optional: Remove model cache
log "Removing model cache (will be re-downloaded)..."
rm -rf ~/.cache/torch/hub/checkpoints 2>/dev/null || true

log_success "Cleanup complete!"
echo ""

# ===== SETUP PHASE =====
log "=== SETUP PHASE ==="
echo ""

# Make sure setup script is executable
chmod +x safe_setup.sh

# Run setup
log "Starting safe_setup.sh..."
echo ""
./safe_setup.sh

echo ""
echo "=========================================="
log_success "Full reinstall complete!"
echo "=========================================="
