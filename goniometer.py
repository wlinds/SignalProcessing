import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import librosa
import sounddevice as sd

# https://python-sounddevice.readthedocs.io/en/0.3.12/api.html

file_path = 'data/wav-loops/its_0_drum_full_loop.wav'

# Synchronize visualization with audio
def audio_callback(outdata, frames, time, status):
    global index, L, R, chunk_size, plot_event # TODO Should make class
    if status:
        print(status)
    chunk = L[index:index+frames], R[index:index+frames]
    if len(chunk[0]) < frames:
        outdata[:len(chunk[0])] = np.stack(chunk, axis=-1)
        outdata[len(chunk[0]):] = 0
        raise sd.CallbackStop
    outdata[:] = np.stack(chunk, axis=-1)
    index += frames
    plot_event.set()

def play_audio(data, sr):
    stream = sd.OutputStream(channels=2, callback=audio_callback, samplerate=sr, blocksize=chunk_size)
    with stream:
        while index < len(L):
            plot_event.wait()
            plot_event.clear()

data, sr = librosa.load(file_path, sr=None, mono=False)
L, R = data[0, :], data[1, :]
chunk_size = 1024
index = 0
plot_event = threading.Event()

# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel('Left Channel')
ax.set_ylabel('Right Channel')
ax.set_title('Goniometer (Lissajous Curve)')
ax.grid(True)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    global index, chunk_size
    start = index
    end = start + chunk_size
    x = L[start:end] / np.max(np.abs(L))
    y = R[start:end] / np.max(np.abs(R))
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=chunk_size / sr * 1000)

audio_thread = threading.Thread(target=play_audio, args=(data, sr))
audio_thread.start()

plt.show()
audio_thread.join()