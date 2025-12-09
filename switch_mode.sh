#!/bin/bash
# Switch vinyl stripper mode via SSH
# Usage: ./switch_mode.sh [vocals|drums|both|none]

MODE=$1
SERVICE="vinyl-stripper"
SERVICE_FILE="/etc/systemd/system/${SERVICE}.service"

if [ -z "$MODE" ]; then
    echo "Usage: ./switch_mode.sh [vocals|drums|both|none]"
    echo ""
    echo "Modes:"
    echo "  vocals  - Remove vocals only"
    echo "  drums   - Remove drums only"
    echo "  both    - Remove vocals and drums"
    echo "  none    - No removal (passthrough)"
    exit 1
fi

# Validate mode
case "$MODE" in
    vocals|drums|both|none)
        ;;
    *)
        echo "Error: Invalid mode '$MODE'"
        echo "Valid modes: vocals, drums, both, none"
        exit 1
        ;;
esac

echo "Switching mode to: $MODE"

# Stop service
echo "Stopping service..."
sudo systemctl stop $SERVICE 2>/dev/null || true

# Update service file with new mode
if [ -f "$SERVICE_FILE" ]; then
    # Find the ExecStart line and update --remove parameter
    sudo sed -i "s/--remove [a-z]*/--remove $MODE/g" $SERVICE_FILE
    
    # If mode is "none", remove --remove parameter entirely
    if [ "$MODE" == "none" ]; then
        sudo sed -i "s/--remove none//g" $SERVICE_FILE
    fi
    
    echo "✓ Service file updated"
else
    echo "Warning: Service file not found at $SERVICE_FILE"
    echo "Run edge_setup.sh first to create the service"
    exit 1
fi

# Reload systemd
echo "Reloading systemd..."
sudo systemctl daemon-reload

# Start service
echo "Starting service..."
sudo systemctl start $SERVICE

# Wait a moment and check status
sleep 2
echo ""
echo "Service status:"
sudo systemctl status $SERVICE --no-pager -l

echo ""
echo "✓ Mode switched to: $MODE"
echo "View logs with: sudo journalctl -u $SERVICE -f"


