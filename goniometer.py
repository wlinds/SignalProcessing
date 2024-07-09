import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import librosa
import sounddevice as sd

# https://python-sounddevice.readthedocs.io/en/0.3.12/api.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html

class Goniometer:
    def __init__(self, file_path, chunk_size=1024):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.data, self.sr = librosa.load(file_path, sr=None, mono=False)
        self.L, self.R = self.data[0, :], self.data[1, :]
        self.index = 0
        self.plot_event = threading.Event()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlabel('Left Channel')
        self.ax.set_ylabel('Right Channel')
        self.ax.set_title('Real-time Goniometer (Lissajous Curve)')
        self.ax.grid(True)

        self.ani = FuncAnimation(self.fig, self.update, init_func=self.init, blit=True, interval=self.chunk_size / self.sr * 1000)

        self.audio_thread = threading.Thread(target=self.play_audio)
        self.audio_thread.start()

    # Synchronize visualization with audio
    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        start = self.index
        end = start + frames
        chunk = self.L[start:end], self.R[start:end]
        if len(chunk[0]) < frames:
            outdata[:len(chunk[0])] = np.stack(chunk, axis=-1)
            outdata[len(chunk[0]):] = 0
            raise sd.CallbackStop
        outdata[:] = np.stack(chunk, axis=-1)
        self.index += frames
        self.plot_event.set()

    def play_audio(self):
        stream = sd.OutputStream(channels=2, callback=self.audio_callback, samplerate=self.sr, blocksize=self.chunk_size)
        with stream:
            while self.index < len(self.L):
                self.plot_event.wait()
                self.plot_event.clear()

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def update(self, frame):
        start = self.index
        end = start + self.chunk_size
        x = self.L[start:end] / np.max(np.abs(self.L))
        y = self.R[start:end] / np.max(np.abs(self.R))
        self.line.set_data(x, y)
        return self.line,

    def show(self):
        plt.show()
        self.audio_thread.join()


if __name__ == "__main__":
    file_path = 'data/wav-loops/its_0_drum_full_loop.wav'
    goniometer = Goniometer(file_path)
    goniometer.show()