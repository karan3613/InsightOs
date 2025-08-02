import sys
import time

import psutil
import GPUtil
import platform
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTabWidget, QTextEdit, QMessageBox, QTableWidget,
    QHBoxLayout, QLineEdit, QTableWidgetItem
)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from stats_screen.MlInsightsScreen import TrainingSession
from helper.gemini_helper import WorkerThread





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

        # User prompt text area
        gaming_layout.addWidget(QLabel("Enter your GPU-related question:"))
        self.user_question_input = QTextEdit()
        self.user_question_input.setPlaceholderText("Example: Should I enable DLSS for better FPS?")
        self.user_question_input.setFixedHeight(100)
        gaming_layout.addWidget(self.user_question_input)

        # Button to ask AI
        self.ask_button = QPushButton("Ask GPU Expert AI")
        gaming_layout.addWidget(self.ask_button)

        # AI response text area (read-only)
        gaming_layout.addWidget(QLabel("Response from GPU Expert AI:"))
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setStyleSheet("background-color: #f4f4f4;")
        gaming_layout.addWidget(self.response_text)

        boost_btn = QPushButton("Boost Performance (Kill Background Tasks)")
        boost_btn.setStyleSheet("background-color: #ff3e3e; color: #fff; padding: 6px; border-radius: 5px;")
        boost_btn.clicked.connect(self.boost_performance)
        gaming_layout.addWidget(boost_btn)

        self.gaming_tab.setLayout(gaming_layout)
        self.tabs.addTab(self.gaming_tab, "Gaming Mode")


        #Tab for ML Insights

        self.ml_tab = QWidget()
        self.active_session = None
        self.sessions = []

        main_layout = QVBoxLayout()

        # Input model name and control buttons
        input_layout = QHBoxLayout()
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("Enter model name...")
        input_layout.addWidget(QLabel("Model Name:"))
        input_layout.addWidget(self.model_name_input)

        self.start_button = QPushButton("Start Session")
        self.start_button.clicked.connect(self.start_session)
        input_layout.addWidget(self.start_button)

        self.epoch_button = QPushButton("Record Epoch")
        self.epoch_button.setEnabled(False)
        self.epoch_button.clicked.connect(self.record_epoch)
        input_layout.addWidget(self.epoch_button)

        self.stop_button = QPushButton("Stop Session")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_session)
        input_layout.addWidget(self.stop_button)

        main_layout.addLayout(input_layout)

        # Active session info
        self.active_session_label = QLabel("No active training session.")
        main_layout.addWidget(self.active_session_label)

        # Session log
        self.session_log = QTextEdit()
        self.session_log.setReadOnly(True)
        main_layout.addWidget(QLabel("Recorded Training Sessions:"))
        main_layout.addWidget(self.session_log)

        # Process table
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(3)
        self.process_table.setHorizontalHeaderLabels(["PID", "Name", "GPU Memory (MB)"])
        main_layout.addWidget(QLabel("Current GPU Processes:"))
        main_layout.addWidget(self.process_table)

        self.ml_tab.setLayout(main_layout)
        self.tabs.addTab(self.ml_tab, "Ml Mode")


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
        self.update_gpu_info()

    def start_session(self):
        model_name = self.model_name_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Input Error", "Please enter a model name before starting session.")
            return
        if self.active_session:
            QMessageBox.warning(self, "Session Active", "A training session is already active.")
            return

        self.active_session = TrainingSession(model_name)
        self.epoch_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)
        self.model_name_input.setEnabled(False)
        self.active_session_label.setText(f"Active Session: {model_name}")

    def record_epoch(self):
        if not self.active_session:
            return
        self.active_session.epoch_count += 1

    # System Information Function
    def get_system_info(self):
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

    def stop_session(self):
        if not self.active_session:
            return

        self.active_session.end_session()
        self.sessions.append(self.active_session)

        # Log session details
        session = self.active_session
        duration = session.duration()
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        log_text = (
            f"Model: {session.model_name}\n"
            f"Epochs: {session.epoch_count}\n"
            f"Duration: {duration_str}\n"
            f"Peak GPU Load: {session.peak_gpu_load:.1f}%\n"
            f"Peak Memory Used: {session.peak_mem_used:.0f} MB\n"
            f"{'-'*40}\n"
        )
        self.session_log.append(log_text)

        # Reset UI state
        self.active_session = None
        self.epoch_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.model_name_input.setEnabled(True)
        self.active_session_label.setText("No active training session.")

    def update_gpu_info(self):
        gpus = GPUtil.getGPUs()
        if not gpus:
            self.active_session_label.setText("No GPU detected.")
            self.process_table.setRowCount(0)
            return

        gpu = gpus[0]  # assuming single GPU, can be extended

        load = gpu.load * 100  # convert to %
        memory_used = gpu.memoryUsed  # MB

        # Update active session peaks
        if self.active_session:
            if load > self.active_session.peak_gpu_load:
                self.active_session.peak_gpu_load = load
            if memory_used > self.active_session.peak_mem_used:
                self.active_session.peak_mem_used = memory_used

            # Update label with live info
            self.active_session_label.setText(
                f"Active Session: {self.active_session.model_name} | "
                f"Epochs: {self.active_session.epoch_count} | "
                f"GPU Load: {load:.1f}% | Memory Used: {memory_used:.0f} MB"
            )
        else:
            self.active_session_label.setText(f"GPU Load: {load:.1f}%, Memory Used: {memory_used:.0f} MB")

        # Update process table
        self.update_process_table(gpu)

    def update_process_table(self, gpu):
        procs = getattr(gpu, "processes", None)
        if procs is None:
            # fallback: no processes info available from GPUtil
            self.process_table.setRowCount(0)
            return

        self.process_table.setRowCount(len(procs))
        for i, proc in enumerate(procs):
            pid = proc['pid']
            mem_used = proc['used_memory']  # MB
            try:
                p = psutil.Process(pid)
                name = p.name()
            except Exception:
                name = "N/A"

            self.process_table.setItem(i, 0, QTableWidgetItem(str(pid)))
            self.process_table.setItem(i, 1, QTableWidgetItem(name))
            self.process_table.setItem(i, 2, QTableWidgetItem(f"{mem_used:.0f}"))

    def on_ask_button_clicked(self):
        gpus = GPUtil.getGPUs()
        if not gpus:
            self.gaming_text.setText("No GPU detected.")
            return

        gpu = gpus[0]  # Assume first GPU for now
        fps_estimate = int((1 - gpu.load) * 144)  # Naive FPS estimate for demo

        gpu_condition = (
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
        user_question = self.user_question_input.toPlainText().strip()

        if "No GPU detected" in gpu_condition:
            QMessageBox.warning(self, "GPU Error", "No GPU detected on your system.")
            return

        if not user_question:
            QMessageBox.warning(self, "Input Error", "Please enter your question.")
            return

        self.ask_button.setEnabled(False)
        self.response_text.setText("Waiting for response from GPU Expert AI...")

        self.thread = WorkerThread(gpu_condition, user_question)
        self.thread.finished.connect(self.display_response)
        self.thread.start()

    def display_response(self, answer):
        self.response_text.setText(answer)
        self.ask_button.setEnabled(True)

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
            self.update_gpu_info()

    def show_specs(self):
        self.spec_text.setText(self.get_system_info())

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