# Enable SSH Using Raspberry Pi Imager

## Method 1: Edit Custom Image (Recommended - No Re-imaging)

1. **Open Raspberry Pi Imager**
   - Download from: https://www.raspberrypi.com/software/
   - Install and open the application

2. **Click "Choose OS"**
   - Select "Use custom image"
   - Navigate to your SD card's boot partition (or the image file if you have one)

3. **Click the Gear Icon** (⚙️) or "Edit custom image"
   - This opens the advanced options

4. **Enable SSH:**
   - Check "Enable SSH"
   - Choose authentication method:
     - **Use password authentication** (recommended)
     - Set username: `pi`
     - Set password: `raspberry` (or your preferred password)
   - OR
     - **Use public key authentication** (more secure)

5. **Configure WiFi (if needed):**
   - Check "Configure wireless LAN"
   - SSID: `byj` (or your network name)
   - Password: Your WiFi password
   - Wireless LAN country: `US`

6. **Set Hostname:**
   - Hostname: `pi` (or leave default)

7. **Save Settings**
   - Click "Save" or "Done"

8. **Write to SD Card**
   - Select your SD card
   - Click "Write" (this will apply the SSH settings without re-imaging)

## Method 2: Re-image with SSH Enabled

If Method 1 doesn't work, you can re-image:

1. **Open Raspberry Pi Imager**

2. **Choose OS:**
   - Select "Raspberry Pi OS (64-bit)" or your preferred version

3. **Click Gear Icon** for advanced options

4. **Enable SSH:**
   - Check "Enable SSH"
   - Set username: `pi`
   - Set password: `raspberry`

5. **Configure WiFi:**
   - Check "Configure wireless LAN"
   - SSID: `byj`
   - Password: Your WiFi password

6. **Set Hostname:** `pi`

7. **Choose Storage:**
   - Select your microSD card

8. **Write:**
   - Click "Write" to create new image with SSH enabled

9. **After Writing:**
   - The SSH file will be automatically created
   - Insert SD card into Pi and boot

## Verification

After using either method:

1. **Eject SD card safely**

2. **Insert into Pi and power on**

3. **Wait 1-2 minutes for boot**

4. **Test SSH:**
   ```bash
   ssh pi@pi.local
   # Password: raspberry
   ```

## Troubleshooting

- If SSH still doesn't work, check:
  - Pi is powered on and LEDs are active
  - Pi is connected to network (check router admin)
  - Try IP address instead: `ssh pi@192.168.1.XXX`
  - Check if SSH service is running: `sudo systemctl status ssh`

## Notes

- Raspberry Pi Imager automatically creates the `ssh` file in the boot partition
- It also configures `user-data` for cloud-init
- This is the most reliable method for enabling SSH



