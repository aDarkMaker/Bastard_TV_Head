import numpy as np  # pyright: ignore[reportMissingImports]


class MouthDetector:
    def __init__(self, rate=16000):
        self.rate = rate
        self._envelope = 0.0
        self.attack = 0.75
        self.release = 0.92
        self.t_closed = 0.0025
        self.t_half = 0.01

    def update(self, audio_frame):
        rms = float(np.sqrt(np.mean(np.square(audio_frame))))
        peak = float(np.max(np.abs(audio_frame)))
        instant = max(rms, peak * 0.55)

        coeff = self.attack if instant > self._envelope else self.release
        self._envelope = coeff * self._envelope + (1 - coeff) * instant

    @property
    def viseme(self):
        e = self._envelope
        if e < self.t_closed:
            return 0
        if e < self.t_half:
            return 1
        return 2

    @property
    def rms(self):
        return self._envelope
