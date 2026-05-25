import time
import numpy as np  # pyright: ignore[reportMissingImports]
from app.audio.capture import AudioCapture


def main():
    devices = AudioCapture.list_devices()
    print("Available input devices:")
    for idx, name in devices:
        print(f"  [{idx}] {name}")

    if not devices:
        print("No input devices found.")
        return

    choice = input("\nDevice index (default 0): ").strip()
    dev_idx = int(choice) if choice else 0

    cap = AudioCapture(device_index=dev_idx)
    cap.start()
    print(f"\nRecording @ {cap.rate}Hz, chunk={cap.chunk}")
    print("Press Ctrl+C to stop.\n")

    recent_rms = []

    try:
        while cap.running:
            frame = cap.read(timeout=0.1)
            if frame is None:
                continue

            rms = float(np.sqrt(np.mean(frame ** 2)))
            recent_rms.append(rms)
            if len(recent_rms) > 40:
                recent_rms = recent_rms[-40:]

            avg_rms = np.mean(recent_rms) if recent_rms else 0

            bar_len = min(int(avg_rms * 200), 78)
            bar = "#" * bar_len
            print(f"\r[{time.time() % 100:05.2f}] RMS: {avg_rms:.4f} | {bar:<78}", end="", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
