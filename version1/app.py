import sys
import psutil
import GPUtil
import platform
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTabWidget, QTextEdit
)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



# System Information Function
def get_system_info():
    uname = platform.uname()
    info = f"""
    System: {uname.system}
    Node Name: {uname.node}
    Release: {uname.release}
    Version: {uname.version}
    Machine: {uname.machine}
    Processor: {uname.processor}
    """
    gpus = GPUtil.getGPUs()
    if gpus:
        for gpu in gpus:
            info += f"\nGPU: {gpu.name} | Total Memory: {gpu.memoryTotal}MB"
    else:
        info += "\nGPU: Not detected"
    return info


# Matplotlib Canvas for live plotting
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(facecolor='#1e1e2f')
        self.cpu_ax = self.fig.add_subplot(311)
        self.ram_ax = self.fig.add_subplot(312)
        self.gpu_ax = self.fig.add_subplot(313)

        super().__init__(self.fig)
        self.setStyleSheet("background-color: #1e1e2f;")


# Main Application Window
class SystemMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Monitor — Clean Aesthetic")
        self.setGeometry(200, 200, 900, 800)
        self.setStyleSheet("background-color: #1e1e2f; color: #c7c7d9;")

        layout = QVBoxLayout()

        self.tabs = QTabWidget()

        # Tab for Top Processes
        self.process_tab = QWidget()
        process_layout = QVBoxLayout()

        self.process_text = QTextEdit()
        self.process_text.setReadOnly(True)
        self.process_text.setStyleSheet("background-color: #1e1e2f; color: #c7c7d9; font-size: 13px;")
        process_layout.addWidget(self.process_text)

        self.process_tab.setLayout(process_layout)
        self.tabs.addTab(self.process_tab, "Top Processes")
        self.tabs.setStyleSheet("background-color: #2e2e3e; color: #c7c7d9;")

        # Tab for Graphs
        self.graph_tab = QWidget()
        graph_layout = QVBoxLayout()
        self.canvas = MplCanvas(self)
        graph_layout.addWidget(self.canvas)
        self.graph_tab.setLayout(graph_layout)
        self.tabs.addTab(self.graph_tab, "Usage Graphs")

        # Gaming Mode Tab
        self.gaming_tab = QWidget()
        gaming_layout = QVBoxLayout()

        self.gaming_text = QTextEdit()
        self.gaming_text.setReadOnly(True)
        self.gaming_text.setStyleSheet("background-color: #1e1e2f; color: #c7c7d9; font-size: 13px;")
        gaming_layout.addWidget(self.gaming_text)

        boost_btn = QPushButton("Boost Performance (Kill Background Tasks)")
        boost_btn.setStyleSheet("background-color: #ff3e3e; color: #fff; padding: 6px; border-radius: 5px;")
        boost_btn.clicked.connect(self.boost_performance)
        gaming_layout.addWidget(boost_btn)

        self.gaming_tab.setLayout(gaming_layout)
        self.tabs.addTab(self.gaming_tab, "Gaming Mode")

        # Tab for GPU Insights
        self.gpu_tab = QWidget()
        gpu_layout = QVBoxLayout()

        self.gpu_text = QTextEdit()
        self.gpu_text.setReadOnly(True)
        self.gpu_text.setStyleSheet("background-color: #1e1e2f; color: #c7c7d9; font-size: 13px;")
        gpu_layout.addWidget(self.gpu_text)

        ask_bot_btn = QPushButton("Ask the Bot")
        ask_bot_btn.setStyleSheet("background-color: #007acc; color: #fff; padding: 6px; border-radius: 5px;")
        gpu_layout.addWidget(ask_bot_btn)

        self.gpu_tab.setLayout(gpu_layout)
        self.tabs.addTab(self.gpu_tab, "GPU Insights")

        # Tab for Specs
        self.spec_tab = QWidget()
        spec_layout = QVBoxLayout()
        self.spec_text = QTextEdit()
        self.spec_text.setReadOnly(True)
        self.spec_text.setStyleSheet("background-color: #1e1e2f; color: #c7c7d9; font-size: 14px;")
        spec_layout.addWidget(self.spec_text)
        self.spec_tab.setLayout(spec_layout)
        self.tabs.addTab(self.spec_tab, "System Specs")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

        # Data for live plotting
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.time_counter = list(range(30))

        # Timer for updating all metrics
        self.timer = QTimer()
        self.timer.setInterval(1000)  # update every 1 second
        self.timer.timeout.connect(self.update_all)
        self.timer.start()

        # Manually initialize all tabs with data
        self.show_specs()
        self.show_top_processes()
        self.show_gaming_stats()
        self.show_gpu_stats()

    def update_all(self):
        # Update all data every second
        self.update_metrics()  # This updates the graphs

        # Update other tabs based on current tab to save resources
        current_tab = self.tabs.currentIndex()

        # Always update these since they're lightweight
        self.show_top_processes()
        self.show_specs()

        # Update other tabs when they're selected
        if current_tab == 2:  # Gaming Mode tab
            self.show_gaming_stats()
        elif current_tab == 3:  # GPU Insights tab
            self.show_gpu_stats()

    def show_specs(self):
        self.spec_text.setText(get_system_info())

    def show_gaming_stats(self):
        gpus = GPUtil.getGPUs()
        if not gpus:
            self.gaming_text.setText("No GPU detected.")
            return

        gpu = gpus[0]  # Assume first GPU for now
        fps_estimate = int((1 - gpu.load) * 144)  # Naive FPS estimate for demo

        text = (
            f"🎮 Gaming Performance Stats\n"
            f"{'-' * 40}\n"
            f"GPU: {gpu.name}\n"
            f"Temperature: {gpu.temperature}°C\n"
            f"Memory Used: {gpu.memoryUsed:.0f}MB / {gpu.memoryTotal:.0f}MB\n"
            f"GPU Load: {gpu.load * 100:.2f}%\n"
            f"Estimated FPS: {fps_estimate} FPS (approx)\n"
            f"{'-' * 40}\n\n"
            f"Top Resource-Heavy Processes:\n"
        )

        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
            try:
                mem_mb = proc.info['memory_info'].rss / 1024 / 1024
                processes.append((
                    proc.info['pid'],
                    proc.info['name'],
                    mem_mb,
                    proc.info['cpu_percent']
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        processes.sort(key=lambda x: (x[2], x[3]), reverse=True)
        for proc in processes[:5]:
            text += f"PID: {proc[0]} | {proc[1]} | {proc[2]:.1f} MB RAM | {proc[3]:.1f}% CPU\n"

        self.gaming_text.setText(text)

    def boost_performance(self):
        killed = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            try:
                if proc.info['cpu_percent'] < 5 and proc.info['memory_info'].rss / 1024 / 1024 < 50:
                    if proc.info['name'] not in ["explorer.exe", "python.exe", "your_game.exe"]:
                        psutil.Process(proc.info['pid']).terminate()
                        killed.append(proc.info['name'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        self.gaming_text.append("\n[BOOST] Terminated background processes:\n" + "\n".join(
            killed) if killed else "\n[BOOST] No background tasks needed termination.")

    def show_gpu_stats(self):
        gpus = GPUtil.getGPUs()
        if not gpus:
            self.gpu_text.setText("No GPUs detected.")
            return

        stats_text = ""
        for gpu in gpus:
            stats_text += (
                f"Name: {gpu.name}\n"
                f"ID: {gpu.id}\n"
                f"Load: {gpu.load * 100:.2f}%\n"
                f"Memory Free: {gpu.memoryFree:.0f}MB\n"
                f"Memory Used: {gpu.memoryUsed:.0f}MB\n"
                f"Memory Total: {gpu.memoryTotal:.0f}MB\n"
                f"Temperature: {gpu}°C\n"
                f"UUID: {gpu.uuid}\n"
                f"{'-' * 40}\n"
            )

        self.gpu_text.setText(stats_text)

    # def ask_gemini(self):
    #     gpus = GPUtil.getGPUs()
    #     if not gpus:
    #         self.gpu_text.append("\nNo GPUs detected for Gemini analysis.")
    #         return
    #
    #     # Prepare GPU data string
    #     gpu_data = ""
    #     for gpu in gpus:
    #         gpu_data += (
    #             f"Name: {gpu.name}\n"
    #             f"ID: {gpu.id}\n"
    #             f"Load: {gpu.load * 100:.2f}%\n"
    #             f"Memory Free: {gpu.memoryFree:.0f}MB\n"
    #             f"Memory Used: {gpu.memoryUsed:.0f}MB\n"
    #             f"Memory Total: {gpu.memoryTotal:.0f}MB\n"
    #             f"Temperature: {gpu.temperature}°C\n"
    #             f"{'-' * 40}\n"
    #         )
    #
    #     # Call Gemini with the current data
    #     suggestion = get_performance_suggestions(gpu_data)
    #     self.gpu_text.append("\n[Gemini Bot Suggestion]\n" + suggestion)

    def show_top_processes(self):
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
            try:
                mem_mb = proc.info['memory_info'].rss / 1024 / 1024  # in MB
                processes.append((
                    proc.info['pid'],
                    proc.info['name'],
                    mem_mb,
                    proc.info['cpu_percent']
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by CPU and RAM usage descending
        processes.sort(key=lambda x: (x[3], x[2]), reverse=True)

        display_text = f"{'PID':<8}{'Name':<30}{'Memory (MB)':<15}{'CPU (%)':<10}\n"
        display_text += "-" * 65 + "\n"
        for proc in processes[:10]:
            display_text += f"{proc[0]:<8}{proc[1]:<30}{proc[2]:<15.2f}{proc[3]:<10.2f}\n"

        self.process_text.setText(display_text)

    def update_metrics(self):
        self.cpu_usage.append(psutil.cpu_percent())
        self.cpu_usage = self.cpu_usage[-30:]

        self.ram_usage.append(psutil.virtual_memory().percent)
        self.ram_usage = self.ram_usage[-30:]

        gpus = GPUtil.getGPUs()
        if gpus:
            self.gpu_usage.append(gpus[0].load * 100)
        else:
            self.gpu_usage.append(0)
        self.gpu_usage = self.gpu_usage[-30:]

        # Update Plots
        self.canvas.cpu_ax.clear()
        self.canvas.ram_ax.clear()
        self.canvas.gpu_ax.clear()

        self.canvas.cpu_ax.plot(self.time_counter[-len(self.cpu_usage):], self.cpu_usage, color="#7dd3fc")
        self.canvas.cpu_ax.set_title("CPU Usage (%)", color="#c7c7d9")
        self.canvas.cpu_ax.set_ylim(0, 100)
        self.canvas.cpu_ax.set_facecolor('#2e2e3e')

        self.canvas.ram_ax.plot(self.time_counter[-len(self.ram_usage):], self.ram_usage, color="#fca5a5")
        self.canvas.ram_ax.set_title("RAM Usage (%)", color="#c7c7d9")
        self.canvas.ram_ax.set_ylim(0, 100)
        self.canvas.ram_ax.set_facecolor('#2e2e3e')

        self.canvas.gpu_ax.plot(self.time_counter[-len(self.gpu_usage):], self.gpu_usage, color="#86efac")
        self.canvas.gpu_ax.set_title("GPU Usage (%)", color="#c7c7d9")
        self.canvas.gpu_ax.set_ylim(0, 100)
        self.canvas.gpu_ax.set_facecolor('#2e2e3e')
        self.canvas.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    monitor = SystemMonitor()
    monitor.show()
    sys.exit(app.exec_())