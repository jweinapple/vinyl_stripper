#!/bin/bash
# SSH Troubleshooting Script for Raspberry Pi

echo "=== Raspberry Pi SSH Troubleshooting ==="
echo ""

# Check network connectivity
echo "1. Checking network connectivity..."
MY_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
echo "   Your IP: $MY_IP"
echo ""

# Try common Pi hostnames
echo "2. Testing common Pi hostnames..."
for hostname in pi.local raspberrypi.local raspberrypi pi; do
    if ping -c 1 -W 1 $hostname &>/dev/null; then
        echo "   ✓ $hostname is reachable"
        echo "   Try: ssh pi@$hostname"
    else
        echo "   ✗ $hostname not found"
    fi
done
echo ""

# Check ARP table for potential Pi devices
echo "3. Checking ARP table for devices..."
arp -a | grep "192.168.1." | grep -v "192.168.1.1" | grep -v "192.168.1.151" | while read line; do
    IP=$(echo $line | awk '{print $2}' | tr -d '()')
    MAC=$(echo $line | awk '{print $4}')
    # Raspberry Pi MAC addresses typically start with: b8:27:eb, dc:a6:32, or e4:5f:01
    if [[ $MAC =~ ^(b8:27:eb|dc:a6:32|e4:5f:01) ]]; then
        echo "   ✓ Potential Pi found: $IP (MAC: $MAC)"
        echo "   Try: ssh pi@$IP"
    fi
done
echo ""

# Test SSH on common IPs
echo "4. Testing SSH on common IP addresses..."
NETWORK=$(echo $MY_IP | cut -d. -f1-3)
for i in {100..110}; do
    IP="$NETWORK.$i"
    if timeout 1 bash -c "echo > /dev/tcp/$IP/22" 2>/dev/null; then
        echo "   ✓ SSH port open on $IP"
        echo "   Try: ssh pi@$IP"
    fi
done
echo ""

# Common solutions
echo "5. Common Solutions:"
echo ""
echo "   A. Find Pi IP address:"
echo "      - Check your router's admin page (usually http://192.168.1.1)"
echo "      - Look for 'raspberrypi' or 'pi' in connected devices"
echo "      - Or connect monitor/keyboard to Pi and run: hostname -I"
echo ""
echo "   B. Enable SSH on Pi (if not enabled):"
echo "      - Boot Pi with monitor/keyboard"
echo "      - Run: sudo systemctl enable ssh"
echo "      - Run: sudo systemctl start ssh"
echo "      - Or: sudo raspi-config → Interface Options → SSH → Enable"
echo ""
echo "   C. Enable mDNS/Bonjour on Pi:"
echo "      - Install: sudo apt install avahi-daemon"
echo "      - Enable: sudo systemctl enable avahi-daemon"
echo "      - Start: sudo systemctl start avahi-daemon"
echo ""
echo "   D. Check Pi is on same network:"
echo "      - Both devices must be on same WiFi/LAN"
echo "      - Check Pi's IP: hostname -I (on Pi)"
echo ""
echo "   E. Try direct IP connection:"
echo "      ssh pi@<PI_IP_ADDRESS>"
echo "      Default password is usually 'raspberry'"
echo ""








