# Shell Script Review and Recommendations

## Core Scripts (KEEP - Essential)

### ✅ safe_setup.sh
**Purpose:** Main setup script for Raspberry Pi/Edge Device  
**Status:** ESSENTIAL - Core functionality  
**Usage:** Primary installation script with phased installation and error recovery

### ✅ run_vinyl_stripper.sh
**Purpose:** Wrapper script for vinyl_stripper.py with enhanced logging  
**Status:** USEFUL - Provides better error handling and logging  
**Usage:** Run vinyl_stripper.py with better diagnostics

### ✅ switch_mode.sh
**Purpose:** Switch vinyl stripper mode via SSH (vocals/drums/both/none)  
**Status:** USEFUL - Runtime control utility  
**Usage:** Change removal mode without manual service file editing

## Setup & Installation Scripts

### ✅ edge_setup.sh
**Purpose:** Alternative setup script for edge devices  
**Status:** KEEP - Alternative setup method  
**Note:** Simpler than safe_setup.sh, may be useful for quick setup

### ✅ init_pi.sh
**Purpose:** Complete Pi setup automation (SSH check, file transfer, setup)  
**Status:** KEEP - Useful automation tool  
**Usage:** Automated remote Pi setup from Mac

## Reinstall/Cleanup Scripts (REDUNDANCY ISSUE)

### ⚠️ full_reinstall.sh
**Purpose:** Full reinstall (cleanup + setup)  
**Status:** KEEP - But note redundancy  
**Location:** Run on Pi

### ⚠️ run_on_pi.sh
**Purpose:** Full reinstall (cleanup + setup) - **DUPLICATE**  
**Status:** **CONSIDER REMOVING** - Identical to full_reinstall.sh  
**Issue:** Does exactly the same thing as full_reinstall.sh

### ⚠️ cleanup_before_reinstall.sh
**Purpose:** Cleanup before reinstall  
**Status:** **CONSIDER REMOVING** - Redundant  
**Issue:** Cleanup is already part of full_reinstall.sh

**Recommendation:** Keep `full_reinstall.sh`, remove `run_on_pi.sh` and `cleanup_before_reinstall.sh`

## SSH Setup & Troubleshooting Scripts

### ✅ create_ssh_file.sh
**Purpose:** Create SSH file on microSD boot partition  
**Status:** KEEP - Useful for initial Pi setup  
**Usage:** When setting up Pi from Mac with microSD card

### ✅ enable_ssh_workaround.sh
**Purpose:** Workaround to enable SSH on Pi  
**Status:** KEEP - Useful troubleshooting tool  
**Usage:** Run on Pi if SSH wasn't enabled during initial setup

### ✅ troubleshoot_ssh.sh
**Purpose:** SSH troubleshooting script  
**Status:** KEEP - Useful troubleshooting tool  
**Usage:** Diagnose SSH connection issues

### ✅ wait_for_pi.sh
**Purpose:** Wait for Raspberry Pi to come online  
**Status:** KEEP - Useful utility  
**Usage:** Wait for Pi to boot and become available

## Summary

### Scripts to KEEP (10):
1. safe_setup.sh - Core setup
2. run_vinyl_stripper.sh - Wrapper with logging
3. switch_mode.sh - Runtime mode switching
4. edge_setup.sh - Alternative setup
5. init_pi.sh - Automated Pi setup
6. full_reinstall.sh - Full reinstall
7. create_ssh_file.sh - SSH setup
8. enable_ssh_workaround.sh - SSH troubleshooting
9. troubleshoot_ssh.sh - SSH diagnostics
10. wait_for_pi.sh - Pi availability check

### Scripts to CONSIDER REMOVING (2):
1. **run_on_pi.sh** - Duplicate of full_reinstall.sh
2. **cleanup_before_reinstall.sh** - Redundant (cleanup is in full_reinstall.sh)

## Recommendations

1. **Remove duplicates:** Delete `run_on_pi.sh` and `cleanup_before_reinstall.sh` since `full_reinstall.sh` covers their functionality
2. **Documentation:** Consider adding a README explaining when to use each script
3. **Consolidation:** The SSH-related scripts (create_ssh_file.sh, enable_ssh_workaround.sh, troubleshoot_ssh.sh) could potentially be consolidated, but they serve different use cases so keeping them separate is fine
