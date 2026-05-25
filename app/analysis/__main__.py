import time
from threading import Thread

from app.analysis.mouth import MouthDetector
from app.analysis.emotion import EmotionClassifier
from app.audio.capture import AudioCapture
from app.display.renderer import Renderer
from app.display.faces import get_face_grid, MOUTH_LEVELS

MOUTH_INTERVAL = 0.05
EMOTION_INTERVAL = 1.0


def main():
    devices = AudioCapture.list_devices()
    mic_idx = devices[0][0] if devices else 0

    renderer = Renderer(pixel_size=16)
    cap = AudioCapture(device_index=mic_idx)
    cap.start()

    mouth = MouthDetector()
    emotion = EmotionClassifier()

    last_mouth = 0.0
    last_emotion = 0.0
    current_emotion = "neutral"
    current_mouth = 0
    t_start = time.perf_counter()
    stop = False

    print("=" * 50)
    print("  Bastard TV Head — Mouth + Emotion Analyzer")
    print("=" * 50)

    def analyze_loop():
        nonlocal last_mouth, last_emotion, current_emotion, current_mouth
        while not stop:
            raw = cap.read(timeout=0.01)
            if raw is None:
                current_mouth = mouth.viseme
            else:
                mouth.update(raw)
                emotion.feed(raw)
                current_mouth = mouth.viseme

            now = time.perf_counter()

            if now - last_mouth >= MOUTH_INTERVAL:
                renderer.push(get_face_grid(current_emotion, current_mouth))
                last_mouth = now

            if now - last_emotion >= EMOTION_INTERVAL and emotion.ready:
                current_emotion, score = emotion.predict()
                last_emotion = now
                mouth_label = MOUTH_LEVELS.get(current_mouth, "?")
                t = now - t_start
                print(
                    f"[{t:6.1f}s] emo={current_emotion:10s} ({score:.2f})"
                    f" | mouth={mouth_label:6s} (rms={mouth.rms:.4f})",
                    flush=True,
                )

    worker = Thread(target=analyze_loop, daemon=True)
    worker.start()

    try:
        while renderer.tick():
            pass
    except KeyboardInterrupt:
        pass
    finally:
        stop = True
        worker.join(timeout=1.0)
        cap.stop()
        renderer.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
