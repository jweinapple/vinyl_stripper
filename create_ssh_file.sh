#!/bin/bash
# Script to create SSH file on microSD boot partition
# Run this when microSD card is mounted on your Mac

BOOT_PATH="/Volumes/bootfs"

echo "=========================================="
echo "Creating SSH File on MicroSD Boot Partition"
echo "=========================================="
echo ""

# Check if bootfs is mounted
if [ ! -d "$BOOT_PATH" ]; then
    echo "❌ Error: Boot partition not found at $BOOT_PATH"
    echo ""
    echo "Please:"
    echo "1. Insert microSD card into your computer"
    echo "2. Wait for it to mount (should appear as 'bootfs')"
    echo "3. Run this script again"
    echo ""
    echo "To check mounted volumes:"
    echo "  ls /Volumes/"
    exit 1
fi

echo "✓ Found boot partition at: $BOOT_PATH"
echo ""

# Create SSH file
if [ -f "$BOOT_PATH/ssh" ]; then
    echo "✓ SSH file already exists"
else
    touch "$BOOT_PATH/ssh"
    if [ -f "$BOOT_PATH/ssh" ]; then
        echo "✓ SSH file created successfully"
    else
        echo "❌ Failed to create SSH file"
        exit 1
    fi
fi

# Verify user-data has enable_ssh
echo ""
echo "Checking user-data configuration..."
if grep -q "enable_ssh: true" "$BOOT_PATH/user-data"; then
    echo "✓ enable_ssh: true is set in user-data"
else
    echo "⚠ enable_ssh is not set in user-data"
    echo "  Adding it now..."
    # Add enable_ssh after #cloud-config line
    if grep -q "^#cloud-config" "$BOOT_PATH/user-data"; then
        # Use sed to add enable_ssh after #cloud-config
        # macOS sed requires different syntax
        sed -i '' '/^#cloud-config/a\
enable_ssh: true
' "$BOOT_PATH/user-data"
        echo "✓ Added enable_ssh: true to user-data"
    else
        echo "⚠ Could not find #cloud-config line, manual edit may be needed"
    fi
fi

# Show summary
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Files modified:"
ls -lh "$BOOT_PATH/ssh" "$BOOT_PATH/user-data" 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Next steps:"
echo "1. Eject the microSD card safely"
echo "2. Insert it into your Raspberry Pi"
echo "3. Power on the Pi"
echo "4. Wait 1-2 minutes for boot"
echo "5. Try SSH: ssh pi@pi.local"
echo ""



