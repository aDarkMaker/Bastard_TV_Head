import time

from app.display.faces import get_face_grid, EMOTIONS
from app.display.renderer import Renderer


def main():
    renderer = Renderer(pixel_size=16)
    emotion_idx = 0
    mouth_level = 0
    last_switch = time.perf_counter()

    print("=" * 50)
    print("  Bastard TV Head — Display Demo")
    print("  Auto-cycles emotions + mouth | [ESC] quit | [F] FPS")
    print("=" * 50)

    try:
        while renderer.tick():
            now = time.perf_counter()
            if now - last_switch >= 1.5:
                emotion = EMOTIONS[emotion_idx % len(EMOTIONS)]
                renderer.push(get_face_grid(emotion, mouth_level))
                print(
                    f"\remotion: {emotion:10s} | mouth: {mouth_level}",
                    end="",
                    flush=True,
                )
                emotion_idx += 1
                mouth_level = (mouth_level + 1) % 3
                last_switch = now
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()
        print()


if __name__ == "__main__":
    main()
