#!/bin/bash
# Wrapper script for vinyl_stripper.py with enhanced logging and error handling

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    log_error "Virtual environment not found. Run safe_setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check memory before running
log "Checking system resources..."
if command -v free &> /dev/null; then
    FREE_MEM=$(free -m | awk '/^Mem:/{print $7}')
    SWAP_SIZE=$(free -m | awk '/^Swap:/{print $2}')
    log "Available memory: ${FREE_MEM}MB"
    log "Swap space: ${SWAP_SIZE}MB"
    
    if [ "$FREE_MEM" -lt 1024 ]; then
        log_warning "Low memory (${FREE_MEM}MB). Model loading may fail."
        log_warning "Consider using lighter model: --model htdemucs"
    fi
fi

# Create logs directory
mkdir -p logs

# Run with error handling
log "Starting vinyl_stripper.py..."
log "Logs will be saved to: logs/"

# Trap errors to ensure we log them
trap 'log_error "Script failed with exit code $?"' ERR

# Run the Python script with all arguments passed through
python3 vinyl_stripper.py "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log "Script completed successfully"
else
    log_error "Script exited with code $EXIT_CODE"
    log_error "Check logs/ directory for detailed error information"
    exit $EXIT_CODE
fi


