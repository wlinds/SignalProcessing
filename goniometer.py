import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import librosa
import sounddevice as sd

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

# https://python-sounddevice.readthedocs.io/en/0.3.12/api.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html

class Goniometer:
    def __init__(self, file_path, chunk_size=1024):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.data, self.sr = librosa.load(file_path, sr=None, mono=False)
        self.L, self.R = self.data[0, :], self.data[1, :]
        self.L = self.L / np.max(np.abs(self.L))
        self.R = self.R / np.max(np.abs(self.R))
        self.index = 0
        self.plot_event = threading.Event()
        self.is_paused = False

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)

        self.line, = self.ax.plot([], [], lw=0.5, color='#800ced')
        # self.ax.set_title(f'{file_path}', color='#bdbdbd')
        self.ax.grid(True, color='#181818')
        self.fig.patch.set_facecolor('#000000')
        self.ax.set_facecolor('#000000')

        self.ani = FuncAnimation(self.fig, self.update, init_func=self.init, blit=True, interval=self.chunk_size / self.sr * 1000)

        self.audio_thread = threading.Thread(target=self.play_audio)
        self.audio_thread.start()

    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        
        if self.is_paused:
            outdata[:] = np.zeros((frames, 2))
            return
        
        start = self.index
        end = start + frames
        chunk = self.L[start:end], self.R[start:end]

        if len(chunk[0]) < frames:
            outdata[:len(chunk[0])] = np.stack(chunk, axis=-1)
            outdata[len(chunk[0]):] = 0
            self.index = 0
            self.is_paused = True
            return
            # raise sd.CallbackStop

        outdata[:] = np.stack(chunk, axis=-1)
        self.index += frames
        self.plot_event.set()

    def play_audio(self):
        stream = sd.OutputStream(channels=2, callback=self.audio_callback, samplerate=self.sr, blocksize=self.chunk_size)
        with stream:
            while self.index < len(self.L):
                self.plot_event.wait()
                self.plot_event.clear()

    def play_frame_chunk(self):
        start = self.index
        end = start + self.chunk_size * 4 # Adjusted for audibility
        chunk = np.stack((self.L[start:end], self.R[start:end]), axis=-1)
        sd.play(chunk, samplerate=self.sr)

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def update(self, frame):
        start = self.index
        end = start + self.chunk_size
        x = self.L[start:end]
        y = self.R[start:end]
        
        if len(x) > 0 and len(y) > 0:
            rotated_x = (x - y) / np.sqrt(2)
            rotated_y = (x + y) / np.sqrt(2)
            self.line.set_data(rotated_x, rotated_y)
        return self.line,

    def show(self):
        plt.show()
        self.audio_thread.join()

    def toggle_play_pause(self):
        if self.index == 0 and self.is_paused:
            self.index = 0
        self.is_paused = not self.is_paused

    def previous_frame(self):
        self.index = max(self.index - self.chunk_size, 0)
        self.plot_event.set()
        self.play_frame_chunk()

    def next_frame(self):
        self.index = min(self.index + self.chunk_size, len(self.L))
        self.plot_event.set()
        self.play_frame_chunk()

    def get_current_frame(self):
        return self.index // self.chunk_size

    def get_total_frames(self):
        return len(self.L) // self.chunk_size

    def get_current_time(self):
        return self.index / self.sr

    def get_total_time(self):
        return len(self.L) / self.sr


class GUI(QMainWindow):
    def __init__(self, goniometer):
        super().__init__()
        self.goniometer = goniometer

        self.canvas = FigureCanvas(self.goniometer.fig)
        self.play_button = QPushButton("⏯️")
        self.play_button.clicked.connect(self.goniometer.toggle_play_pause)

        self.frame_label = QLabel()
        self.time_label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.play_button)

        info_layout = QHBoxLayout()
        info_layout.addWidget(self.frame_label)
        info_layout.addWidget(self.time_label)
        layout.addLayout(info_layout)

        self.frame_label.setFont(QFont("Courier", 10))
        self.time_label.setFont(QFont("Courier", 10))

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.update_info()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_info)
        self.timer.start(100)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.goniometer.toggle_play_pause()
        elif event.key() == Qt.Key_Left:
            self.goniometer.previous_frame()
        elif event.key() == Qt.Key_Right:
            self.goniometer.next_frame()
        else:
            super().keyPressEvent(event)
    
    def update_info(self):
        current_frame = self.goniometer.get_current_frame()
        total_frames = self.goniometer.get_total_frames()
        current_time = self.goniometer.get_current_time()
        total_time = self.goniometer.get_total_time()

        self.frame_label.setText(f"Frame: {current_frame}/{total_frames}")
        self.time_label.setText(f"Time: {current_time:.2f}/{total_time:.2f} s")


if __name__ == "__main__":
    file_path = 'data/renders/example_3.wav'
    goniometer = Goniometer(file_path)
    
    app = QApplication(sys.argv)
    gui = GUI(goniometer)
    gui.show()

    gui_thread = threading.Thread(target=app.exec_)
    gui_thread.start()

    sys.exit(app.exec_())