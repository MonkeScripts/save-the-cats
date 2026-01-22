# ESP32/FireBeetle Not Found on Linux (BRLTTY Conflict) 
## Issue Description
When attempting to upload firmware to an ESP32 (FireBeetle) via PlatformIO or Arduino IDE on Linux, the device fails to connect. The system may auto-detect /dev/ttyS0 (which is incorrect), and the **actual port (/dev/ttyUSB0) disappears immediately after the device is plugged in**.

## Root Cause Analysis
The BRLTTY (Braille TTY) daemon, intended for Braille display support, incorrectly identifies the CH341/CP210x serial-to-USB chip as a Braille device.
When the ESP32 board is connected, BRLTTY claims the USB interface, causing the serial driver to disconnect.
## Diagnostic Steps
As seen in dmesg, the kernel assigns the port, but BRLTTY immediately "claims" the interface and disconnects the driver:
```
usb 3-5: usbfs: interface 0 claimed by ch341 while 'brltty' sets config #1 ch341-uart ttyUSB0: ch341-uart converter now disconnected from ttyUSB0
```

Solution
1. Remove the Conflicting Service
The most effective fix (if you do not require Braille display support) is to remove the brltty package:

```bash
sudo apt remove brltty
```

2. Manual Service Override (Alternative)
If you prefer not to uninstall the package, disable the service and its udev rules:
```bash

sudo systemctl stop brltty-udev.service
sudo systemctl mask brltty-udev.service
```
3. Grant User Permissions
Ensure your user has the rights to access serial ports:

```

sudo usermod -a -G dialout $USER
Note: You must log out and log back in for group changes to take effect.
```
## Verification
To verify the fix, run dmesg -w and plug in the board. You should see the following output without a subsequent "disconnected" message:
