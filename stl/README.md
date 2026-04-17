# STL Files for 3D Printing

This folder contains the STL files for the custom 3D-printed enclosures used in the wearable sensor nodes of the project.

## Files Included

* **Biceps 2.0.stl**
  Enclosure designed for the bicep-mounted sensor node.

* **Thigh & Chest 2.0.stl**
  Enclosure designed for the thigh-mounted and chest-mounted sensor nodes.

* **Lid_2.0.stl**
  Top cover / lid used to secure and protect the electronics inside the enclosure.

## How to Use

1. Download the required `.stl` file.
2. Open the file using a **3D slicer software** (e.g. Cura, PrusaSlicer, Bambu Studio).
3. Configure print settings such as:

   * Material type (PLA recommended)
   * Layer height
   * Infill percentage
   * Supports (if required)
4. Slice the model and export the G-code file.
5. Send the G-code to the 3D printer and begin printing.

## Recommended Material

* **PLA** – Easy to print and suitable for prototyping
* **PETG** – Better durability and heat resistance

## Notes

* Ensure printer bed dimensions are sufficient for the model size.
* Print orientation may affect surface finish and strength.
* Minor post-processing (sanding / fitting adjustments) may be required depending on printer tolerance.

## Purpose

These enclosures are designed to house the FireBeetle ESP32, IMU sensor, OLED display, and battery into a compact wearable node for exercise motion tracking.
