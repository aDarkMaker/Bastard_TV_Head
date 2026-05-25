import argparse
import time
from threading import Thread

from app.audio.capture import AudioCapture
from app.display.faces import get_face_grid
from app.display.renderer import Renderer
from app.core.orchestrator import Orchestrator


def parse_args():
    p = argparse.ArgumentParser(description="Bastard TV Head")
    p.add_argument("--mode", choices=["live", "demo"], default="live")
    p.add_argument("--mic-device", type=int, default=None)
    p.add_argument("--pixel-size", type=int, default=16)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--profile", action="store_true", default=False)
    return p.parse_args()


def run_live(args):
    cap = AudioCapture(device_index=args.mic_device)
    renderer = Renderer(pixel_size=args.pixel_size, width=args.width, height=args.height)
    orch = Orchestrator(cap, renderer)
    orch.start()
    renderer.push(get_face_grid("neutral", 0))

    print("Bastard TV Head — Live mode")
    print("[ESC] quit | [F] toggle FPS")

    t_last_profile = time.perf_counter()
    stop = False

    def audio_loop():
        nonlocal stop, t_last_profile
        while not stop:
            orch.tick()
            if args.profile and time.perf_counter() - t_last_profile > 2.0:
                stats = orch.profile_stats
                if stats:
                    print(
                        f"[profile] cap={stats['capture_ms']:.2f}ms"
                        f" mouth={stats['mouth_ms']:.2f}ms"
                        f" emo={stats['emotion_ms']:.2f}ms"
                        f" render={stats['render_ms']:.2f}ms"
                    )
                t_last_profile = time.perf_counter()

    worker = Thread(target=audio_loop, daemon=True)
    worker.start()

    try:
        while renderer.tick():
            pass
    except KeyboardInterrupt:
        pass
    finally:
        stop = True
        worker.join(timeout=1.0)
        orch.stop()
        renderer.close()


def run_demo(args):
    from app.display.faces import get_face_grid, EMOTIONS

    renderer = Renderer(pixel_size=args.pixel_size, width=args.width, height=args.height)
    emotions = list(EMOTIONS)
    emotion_idx = 0
    mouth_idx = 0
    last_switch = time.perf_counter()

    print("Bastard TV Head — Demo mode")
    print("Auto-cycles emotions | [ESC] quit")

    try:
        while renderer.tick():
            now = time.perf_counter()
            if now - last_switch >= 2.0:
                emotion = emotions[emotion_idx % len(emotions)]
                renderer.push(get_face_grid(emotion, mouth_idx))
                emotion_idx += 1
                mouth_idx = (mouth_idx + 1) % 3
                last_switch = now
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()


def main():
    args = parse_args()
    if args.mode == "demo":
        run_demo(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
