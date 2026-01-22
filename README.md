# save-the-cats
A gamified AR gym experience to motivate fitness enthusiasts to achieve their workout goals by saving virtual cats.
<img width="686" height="386" alt="image" src="https://github.com/user-attachments/assets/ffda538e-302c-4de9-8c08-4c2f97c4b2ab" />


Project uses ESP32/FireBeetle microcontroller boards for tracking user movements and interactions within the gym environment.

## Setting up PIO Environment
1. Install PlatformIO IDE extension in VSCode.
2. Open this project folder through PIO Home in VSCode.

### Linux Specific Setup
Extra steps for linux devices:
1. Add your user to the dialout group to access serial ports:
   ```bash
   sudo usermod -a -G dialout $USER
   ```
2. Log out and log back in for the group changes to take effect.
3. Add udev rules for ESP32/FireBeetle boards. Adapted from [PlatformIO documentation](https://docs.platformio.org/en/latest/core/installation/udev-rules.html):
    ```bash
    curl -fsSL https://raw.githubusercontent.com/platformio/platformio-core/develop/platformio/assets/system/99-platformio-udev.rules | sudo tee /etc/udev/rules.d/99-platformio-udev.rules
    ```
    Then reload udev rules:
    ```bash
    sudo service udev restart

    # or

    sudo udevadm control --reload-rules
    sudo udevadm trigger
    ```
4. Check whether the device is recognized:
    ```bash
    ls -l /dev/ttyUSB*
    ```
     > **Note:** If it is not recognized, see the troubleshooting section below.
    >

5. Add the specific device path to `platformio.ini` if necessary:
    ```
    upload_port = /dev/ttyUSB0
    monitor_port = /dev/ttyUSB0
    ``` 
    If you have multiple serial devices, ensure you specify the correct one.
    > **WIP:** Automate detection of the correct serial port in future updates with udev rules.
    >



# Issues and fixes
## Issues with Serial Port Recognition due to BRLTTY Conflict
When attempting to upload firmware to an ESP32 (FireBeetle) via PlatformIO or Arduino IDE on Linux, the device fails to connect. The **actual port (/dev/ttyUSB0) disappears immediately after the device is plugged in**.

### Root Cause Analysis
The BRLTTY (Braille TTY) daemon, intended for Braille display support, incorrectly identifies the CH341/CP210x serial-to-USB chip as a Braille device.
When the ESP32 board is connected, BRLTTY claims the USB interface, causing the serial driver to disconnect. This can be seen in `dmesg`, the kernel assigns the port, but BRLTTY immediately "claims" the interface and disconnects the driver:
```
usb 3-5: usbfs: interface 0 claimed by ch341 while 'brltty' sets config #1 ch341-uart ttyUSB0: ch341-uart converter now disconnected from ttyUSB0
```

### Solution
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

### Verification
To verify the fix, run dmesg -w and plug in the board. You should see the device being assigned to `/dev/ttyUSB*` without any disconnection messages from brltty.
