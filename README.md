# Haps Vision 

**Haps Vision** is a haptic navigation wearable designed for the visually impaired. By moving navigation instructions from the ears to the wrist, we provide a silent, discrete, and safer way to navigate loud urban environments without losing situational awareness.

---

## Features

- **Haptic Directional Guidance:** Three vibration motors provide intuitive "Left," "Right," and "Forward" signals.
- **Smart Calibration:** Uses the phone's magnetometer to ensure the user is facing the correct cardinal direction before the first step.
- **Event-Driven Navigation:** The ESP32 confirms physical turn completion via the MPU-6050 gyroscope before advancing the route.
- **Accessibility-First Web App:** A clean, high-contrast mobile web interface compatible with screen readers and Bluefy (iOS Web Bluetooth).

---

## Tech Stack

### Software
- **Languages:** JavaScript, C++, HTML, CSS
- **APIs:** Google Maps (Directions, Places Autocomplete), Web Bluetooth API, Device Orientation API
- **Frameworks/Libraries:** ESP-DL (Deep Learning), ESP-IDF

### Hardware
- **Microcontroller:** ESP32-S3
- **Sensors:** MPU-6050 (6-Axis Accelerometer & Gyroscope)
- **Actuators:** Haptic Vibration Motors

---

## Project Structure

- `/web-app`: The frontend interface built with JavaScript and Google Maps API.
- `/firmware`: C++ source code for the ESP32, including the BLE GATT server and MPU-6050 integration.
- `/models`: Training data and quantized ESP-DL model files for gesture recognition.

---

## Installation & Setup

### Web App
1. Clone the repository.
2. Enable Google Maps API key.
3. Host the folder using an HTTPS-enabled server (GitHub Pages, Vercel, or Netlify).
4. Access the site via the **Bluefy** browser on iOS.

### Firmware
1. Open `/firmware` in the Arduino IDE or VS Code (PlatformIO).
2. Install the `ESP32` board manager and the `ESP-DL` library.
3. Flash the code to your ESP32-S3.
4. Open the Serial Monitor to verify the BLE service is advertising.

---
*Developed for the 2026 StarkHacks Hackathon.*
