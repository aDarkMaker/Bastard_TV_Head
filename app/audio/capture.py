import queue

import numpy as np  # pyright: ignore[reportMissingImports]

try:
    import pyaudio  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
except ImportError:
    pyaudio = None  # type: ignore[assignment]

DEFAULT_RATE = 16000
DEFAULT_CHUNK = 512
DEFAULT_CHANNELS = 1


class AudioCapture:
    def __init__(self, rate=DEFAULT_RATE, chunk=DEFAULT_CHUNK, device_index=None):
        self.rate = rate
        self.chunk = chunk
        self.device_index = device_index
        self.buffer = queue.Queue(maxsize=256)
        self.running = False
        self._stream = None
        self._p = None

    def start(self):
        if pyaudio is None:
            raise RuntimeError("pyaudio not installed")
        self.running = True
        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(
            format=pyaudio.paInt16,
            channels=DEFAULT_CHANNELS,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def stop(self):
        self.running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._p:
            self._p.terminate()

    def read(self, timeout=0.05):
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def _callback(self, in_data, frame_count, time_info, status):
        if self.running and pyaudio is not None:
            data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            if not self.buffer.full():
                self.buffer.put(data)
            return (None, pyaudio.paContinue)
        return (None, 0)

    @staticmethod
    def list_devices():
        if pyaudio is None:
            return []
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if int(info["maxInputChannels"]) > 0:
                devices.append((i, info["name"]))
        p.terminate()
        return devices
