import sys
import numpy as np
import sounddevice as sd
import librosa
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

class LiveSpectrogram(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.device_list = sd.query_devices()

        self.sr = 44100
        self.buffer_size = 2048 
        self.update_interval = 50  # ms
        self.audio_buffer = np.zeros(self.buffer_size)

        self.spec_buffer_size = 100
        self.spec_buffer = np.zeros((self.buffer_size // 2 + 1, self.spec_buffer_size))
        
        self.mode = 'Scrolling'
        
        self.initUI()
        self.start_stream()

    def initUI(self):
        self.setWindowTitle('Spectrogram')

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Plot
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.plot_item = self.plot_widget.getPlotItem()
        self.img_item = pg.ImageItem()
        self.plot_item.addItem(self.img_item)
        
        # Controls
        controls_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(controls_layout)
        
        device_label = QtWidgets.QLabel("Input Device:")
        controls_layout.addWidget(device_label)

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems([d['name'] for d in self.device_list])
        controls_layout.addWidget(self.device_combo)
        self.device_combo.currentIndexChanged.connect(self.change_device)

        mode_label = QtWidgets.QLabel("Mode:")
        controls_layout.addWidget(mode_label)
        
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(['Scrolling', 'Instant'])
        controls_layout.addWidget(self.mode_combo)
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        
        scale_label = QtWidgets.QLabel("Scale:")
        controls_layout.addWidget(scale_label)
        
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(['Hz', 'Mel'])
        controls_layout.addWidget(self.scale_combo)
        self.scale_combo.currentIndexChanged.connect(self.change_scale)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.update_interval)
        
        self.show()
    
    def start_stream(self):
        device_index = self.device_combo.currentIndex()
        self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=self.sr, device=device_index)
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        self.audio_buffer[:-frames] = self.audio_buffer[frames:]
        self.audio_buffer[-frames:] = indata[:, 0]

    def update_plot(self):
        if self.scale_combo.currentText() == 'Hz':
            S = librosa.stft(self.audio_buffer, n_fft=self.buffer_size, hop_length=self.buffer_size // 4)
            S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            freqs = np.fft.fftfreq(self.buffer_size, 1/self.sr)[:self.buffer_size//2 + 1]
            freqs = freqs[:len(freqs)//2]  # Use only positive frequencies
            max_freq = self.sr / 2  # half the sampling rate
        else:
            S = librosa.feature.melspectrogram(y=self.audio_buffer, sr=self.sr, n_fft=self.buffer_size, hop_length=self.buffer_size // 4, n_mels=self.buffer_size // 2 + 1)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            freqs = librosa.mel_frequencies(n_mels=self.buffer_size // 2 + 1, fmin=0, fmax=self.sr//2)
            max_freq = self.sr / 2

        if self.mode == 'Scrolling':
            self.spec_buffer[:, :-1] = self.spec_buffer[:, 1:]  # Shift existing data
            self.spec_buffer[:, -1] = np.mean(S_dB, axis=1)

            img = (self.spec_buffer - np.min(self.spec_buffer)) / (np.max(self.spec_buffer) - np.min(self.spec_buffer)) * 255
        else:
            # Instant mode
            img = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min()) * 255

        img = img.astype(np.uint8)

        img = np.flipud(img.T)

        colormap = pg.colormap.get('viridis')
        lut = colormap.getLookupTable()

        self.img_item.setImage(img, autoLevels=False, lut=lut)


    def change_device(self, index):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.start_stream()
    
    def change_scale(self, index):
        pass

    def change_mode(self, index):
        self.mode = self.mode_combo.currentText()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    spectrogram = LiveSpectrogram()
    sys.exit(app.exec_())
