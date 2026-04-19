import serial
import csv
import time
import os
from datetime import datetime
from serial import SerialException

PORT = "COM7"
BAUD = 115200

LABEL_NAME = "idle"   # change to: idle, tap, flick, turn
BASE_DIR = "data"

run_dir = os.path.join(BASE_DIR, LABEL_NAME)
os.makedirs(run_dir, exist_ok=True)

OUTFILE = os.path.join(
    run_dir,
    f"{LABEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

with open(OUTFILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_ms", "label", "ax", "ay", "az", "gx", "gy", "gz"])
    f.flush()

    print(f"Logging to {OUTFILE} from {PORT} at {BAUD} baud...")

    ser = None

    while True:
        try:
            if ser is None or not ser.is_open:
                print(f"Connecting to {PORT}...")
                ser = serial.Serial(PORT, BAUD, timeout=1)
                time.sleep(1.0)
                print("Connected.")

            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            if line.startswith("CSV,"):
                row = line[4:].split(",")
                if len(row) == 8:
                    writer.writerow(row)
                    f.flush()
                    print("logged:", row)
            else:
                print("ESP:", line)

        except KeyboardInterrupt:
            print("\nStopped logging.")
            if ser and ser.is_open:
                ser.close()
            break

        except SerialException as e:
            print(f"Serial lost: {e}")
            if ser:
                try:
                    ser.close()
                except Exception:
                    pass
            ser = None
            time.sleep(1.0)