#!/bin/bash
# Workaround script to enable SSH on Raspberry Pi
# This ensures SSH is enabled even if the 'ssh' file is missing from boot partition
# Run this on the Pi via physical access or after first SSH connection

echo "Enabling SSH workaround..."

# Enable SSH service
sudo systemctl enable ssh
sudo systemctl start ssh

# Ensure SSH is enabled in cloud-init for future boots
if [ -f /boot/user-data ]; then
    # Check if enable_ssh is already set
    if ! grep -q "enable_ssh: true" /boot/user-data; then
        echo "Adding enable_ssh to user-data..."
        sudo sed -i '/^#cloud-config/a enable_ssh: true' /boot/user-data
    fi
fi

# Create the ssh file in boot partition for next boot
sudo touch /boot/ssh
echo "✓ SSH file created in /boot/ssh"

# Verify SSH is running
if systemctl is-active --quiet ssh; then
    echo "✓ SSH service is running"
else
    echo "⚠ SSH service is not running, starting it..."
    sudo systemctl start ssh
fi

echo ""
echo "SSH should now be enabled. Try connecting:"
echo "  ssh pi@pi.local"
echo "  or"
echo "  ssh pi@<PI_IP_ADDRESS>"




