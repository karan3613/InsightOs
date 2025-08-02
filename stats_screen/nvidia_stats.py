import sys
import GPUtil
import psutil
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QGridLayout
from PyQt5.QtCore import QTimer
from pynvml import *

# Initialize NVML safely
try:
    nvmlInit()
    nvml_available = True
except NVMLError as err:
    print(f"Failed to initialize NVML: {err}")
    nvml_available = False


class GPUStatsScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎮 Real-time GPU & System Stats")
        self.setGeometry(100, 100, 520, 420)

        # Layout
        layout = QVBoxLayout()
        self.grid = QGridLayout()
        layout.addLayout(self.grid)
        self.setLayout(layout)

        # Labels
        self.labels = {}
        stat_names = [
            "GPU Load", "Temperature", "Memory Used", "Memory Total",
            "Power Draw", "Fan Speed", "Encoder Utilization",
            "Decoder Utilization", "Core Clock", "Memory Clock",
            "Throttle Reasons", "CPU Usage", "RAM Usage"
        ]
        for i, name in enumerate(stat_names):
            label_name = QLabel(f"{name}:")
            label_value = QLabel("Loading...")
            self.grid.addWidget(label_name, i, 0)
            self.grid.addWidget(label_value, i, 1)
            self.labels[name] = label_value

        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)

    def update_stats(self):
        gpus = GPUtil.getGPUs()
        if not gpus or not nvml_available:
            for label in self.labels.values():
                label.setText("GPU not detected.")
            return

        gpu = gpus[0]
        handle = nvmlDeviceGetHandleByIndex(gpu.id)

        try:
            power = nvmlDeviceGetPowerUsage(handle) / 1000  # W
        except NVMLError:
            power = 0

        try:
            fan_speed = nvmlDeviceGetFanSpeed(handle)
        except NVMLError:
            fan_speed = 0

        try:
            enc_util, _ = nvmlDeviceGetEncoderUtilization(handle)
        except NVMLError:
            enc_util = 0

        try:
            dec_util, _ = nvmlDeviceGetDecoderUtilization(handle)
        except NVMLError:
            dec_util = 0

        try:
            clocks = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)
        except NVMLError:
            clocks = 0

        try:
            mem_clocks = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)
        except NVMLError:
            mem_clocks = 0

        try:
            throttle_reasons = nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        except NVMLError:
            throttle_reasons = 0

        cpu_usage = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_usage = f"{ram.used // (1024 ** 2)} MB / {ram.total // (1024 ** 2)} MB"

        # Update Labels
        self.labels["GPU Load"].setText(f"{gpu.load * 100:.2f}%")
        self.labels["Temperature"].setText(f"{gpu.temperature}°C")
        self.labels["Memory Used"].setText(f"{gpu.memoryUsed} MB")
        self.labels["Memory Total"].setText(f"{gpu.memoryTotal} MB")
        self.labels["Power Draw"].setText(f"{power:.1f} W")
        self.labels["Fan Speed"].setText(f"{fan_speed}%")
        self.labels["Encoder Utilization"].setText(f"{enc_util}%")
        self.labels["Decoder Utilization"].setText(f"{dec_util}%")
        self.labels["Core Clock"].setText(f"{clocks} MHz")
        self.labels["Memory Clock"].setText(f"{mem_clocks} MHz")
        self.labels["Throttle Reasons"].setText(f"0x{throttle_reasons:X}")
        self.labels["CPU Usage"].setText(f"{cpu_usage}%")
        self.labels["RAM Usage"].setText(ram_usage)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    screen = GPUStatsScreen()
    screen.show()
    sys.exit(app.exec_())
