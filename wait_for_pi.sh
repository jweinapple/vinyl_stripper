#!/bin/bash
# Script to wait for Raspberry Pi to come online and connect via SSH

echo "Waiting for Raspberry Pi to come online..."
echo "This may take 2-3 minutes after power-on"
echo ""

MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo -n "Attempt $ATTEMPT/$MAX_ATTEMPTS: "
    
    # Try hostname first
    if ping -c 1 -W 1 pi.local &>/dev/null; then
        echo "✓ Pi found at pi.local"
        echo ""
        echo "Testing SSH connection..."
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no pi@pi.local "echo 'SSH working!' && hostname -I" 2>/dev/null; then
            echo ""
            echo "✓✓✓ Pi is online and SSH is working! ✓✓✓"
            echo ""
            echo "You can now connect with:"
            echo "  ssh pi@pi.local"
            exit 0
        else
            echo "  Pi is online but SSH not ready yet..."
        fi
    # Try direct IP
    elif ping -c 1 -W 1 192.168.1.153 &>/dev/null; then
        echo "✓ Pi found at 192.168.1.153"
        echo ""
        echo "Testing SSH connection..."
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no pi@192.168.1.153 "echo 'SSH working!' && hostname -I" 2>/dev/null; then
            echo ""
            echo "✓✓✓ Pi is online and SSH is working! ✓✓✓"
            echo ""
            echo "You can now connect with:"
            echo "  ssh pi@192.168.1.153"
            exit 0
        else
            echo "  Pi is online but SSH not ready yet..."
        fi
    else
        echo "Not found yet..."
    fi
    
    sleep 10
done

echo ""
echo "Pi did not come online after $MAX_ATTEMPTS attempts."
echo ""
echo "Troubleshooting:"
echo "1. Check if Pi's power LED is on"
echo "2. Check router admin page for connected devices"
echo "3. Verify WiFi credentials in network-config"
echo "4. Try connecting monitor/keyboard to Pi directly"
echo ""
exit 1




