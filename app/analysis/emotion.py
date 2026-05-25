from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]
from collections import deque

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "fearful"]
_EMOTION_IDX = {e: i for i, e in enumerate(EMOTIONS)}


class EmotionClassifier:
    def __init__(self, rate=16000, window_samples=None):
        self.rate = rate
        self.window_samples = window_samples or rate * 2
        self._buffer = deque(maxlen=self.window_samples)
        self._model: Any = None
        self._history = deque(maxlen=10)

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            from speechbrain.inference import EncoderClassifier  # pyright: ignore[reportMissingImports]

            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir="/tmp/speechbrain_emotion",
            )
        except Exception:
            self._model = False

    def feed(self, audio_frame):
        self._buffer.extend(audio_frame.astype(np.float64).tolist())

    @property
    def ready(self):
        return len(self._buffer) >= self.rate

    def predict(self):
        if not self.ready:
            return "neutral", 0.0
        self._lazy_load()
        signal = np.array(list(self._buffer)[-self.window_samples:], dtype=np.float64)

        if self._model and self._model is not False:
            label, score = self._predict_model(signal)
        else:
            label, score = self._predict_fallback(signal)

        self._history.append(label)
        return self._smooth(), score

    def _predict_model(self, signal):
        try:
            import torch  # pyright: ignore[reportMissingImports]
            waveform = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
            out_prob, _, _, _ = self._model.classify_batch(waveform)
            idx = out_prob.argmax(dim=1).item()
            score = out_prob.max().item()
            label_map = {
                0: "neutral",
                1: "happy",
                2: "sad",
                3: "angry",
            }
            return label_map.get(idx, "neutral"), score
        except Exception:
            return self._predict_fallback(signal)

    def _predict_fallback(self, signal):
        rms = float(np.sqrt(np.mean(signal**2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(signal)))) / 2)
        if rms < 0.005:
            return "neutral", 0.9
        if rms > 0.08 and zcr > 0.2:
            return "angry", 0.7
        if rms > 0.04 and zcr > 0.15:
            return "happy", 0.7
        if zcr < 0.05:
            return "sad", 0.6
        return "neutral", 0.5

    def _smooth(self):
        if not self._history:
            return "neutral"
        from collections import Counter
        return Counter(self._history).most_common(1)[0][0]
