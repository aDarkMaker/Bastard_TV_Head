import time
from collections import deque

from app.analysis.mouth import MouthDetector
from app.analysis.emotion import EmotionClassifier
from app.display.faces import get_face_grid


class Orchestrator:
    def __init__(self, audio_capture, renderer, mouth_interval=0.05, emotion_interval=1.0):
        self.capture = audio_capture
        self.renderer = renderer
        self.mouth_interval = mouth_interval
        self.emotion_interval = emotion_interval

        self.mouth_det = MouthDetector()
        self.emotion_clf = EmotionClassifier()

        self._emotion_label = "neutral"
        self._mouth_level = 0

        self._last_mouth = 0.0
        self._last_emotion = 0.0
        self._t_start = 0.0

        self.profile_times = deque(maxlen=120)
        self._profile_enabled = True

    @property
    def emotion(self):
        return self._emotion_label

    @property
    def mouth_level(self):
        return self._mouth_level

    def start(self):
        self.capture.start()
        self._t_start = time.perf_counter()
        self._last_mouth = self._t_start
        self._last_emotion = self._t_start

    def stop(self):
        self.capture.stop()

    def tick(self):
        t0 = time.perf_counter()

        raw = self.capture.read(timeout=0.01)
        t_capture = time.perf_counter() - t0

        if raw is None:
            self._mouth_level = self.mouth_det.viseme
        else:
            self.mouth_det.update(raw)
            self.emotion_clf.feed(raw)
            self._mouth_level = self.mouth_det.viseme

        t_analysis = time.perf_counter() - t_capture - t0

        now = time.perf_counter()
        if now - self._last_emotion >= self.emotion_interval and self.emotion_clf.ready:
            self._emotion_label, _ = self.emotion_clf.predict()
            self._last_emotion = now

        t_emotion = time.perf_counter() - now

        now = time.perf_counter()
        if now - self._last_mouth >= self.mouth_interval:
            grid = get_face_grid(self._emotion_label, self._mouth_level)
            self.renderer.push(grid)
            self._last_mouth = now

        t_render = time.perf_counter() - now

        if self._profile_enabled:
            self.profile_times.append((t_capture, t_analysis, t_emotion, t_render))

    @property
    def profile_stats(self):
        if not self.profile_times:
            return {}
        n = len(self.profile_times)
        t_cap = sum(p[0] for p in self.profile_times) / n * 1000
        t_ana = sum(p[1] for p in self.profile_times) / n * 1000
        t_emo = sum(p[2] for p in self.profile_times) / n * 1000
        t_ren = sum(p[3] for p in self.profile_times) / n * 1000
        return {"capture_ms": t_cap, "mouth_ms": t_ana, "emotion_ms": t_emo, "render_ms": t_ren}
