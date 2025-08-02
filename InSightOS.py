import sys
import time

import psutil
import GPUtil
import platform

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTabWidget, QTextEdit, QMessageBox, QTableWidget,
    QHBoxLayout, QLineEdit, QTableWidgetItem, QFrame
)
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from stats_screen.MlInsightsScreen import TrainingSession
from helper.gemini_helper import WorkerThread
from helper.process_tracker import ProcessStatsTracker


class BarPlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3.5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super(BarPlotCanvas, self).__init__(fig)
        self.setStyleSheet("background-color: #121212;")
        # Optional: initial empty plot
        self.ax.set_facecolor('#121212')

    def plot(self, labels, values, title, xlabel, ylabel):
        orange = '#FFA500'
        black = '#121212'

        self.ax.clear()
        self.ax.bar(labels, values, color=orange)
        self.ax.set_facecolor(black)
        self.ax.set_title(title, color=orange)
        self.ax.set_xlabel(xlabel, color=orange)
        self.ax.set_ylabel(ylabel, color=orange)
        self.ax.tick_params(axis='x', colors=orange, rotation=45)
        self.ax.tick_params(axis='y', colors=orange)

        for i, value in enumerate(values):
            self.ax.text(i, value, f'{value:.0f}', ha='center', va='bottom', color=orange, fontsize=8)

        self.figure.tight_layout()
        self.draw()


# Enhanced Matplotlib Canvas
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(facecolor='#0a0a0a', edgecolor='#ff6600')
        self.fig.patch.set_facecolor('#0a0a0a')

        # Create subplots with custom spacing
        self.cpu_ax = self.fig.add_subplot(311)
        self.ram_ax = self.fig.add_subplot(312)
        self.gpu_ax = self.fig.add_subplot(313)

        # Style the axes
        for ax in [self.cpu_ax, self.ram_ax, self.gpu_ax]:
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='#ff6600', labelsize=8)
            ax.spines['bottom'].set_color('#ff6600')
            ax.spines['top'].set_color('#ff6600')
            ax.spines['right'].set_color('#ff6600')
            ax.spines['left'].set_color('#ff6600')
            ax.grid(True, alpha=0.2, color='#ff6600')

        super().__init__(self.fig)
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0a0a;
                border: 2px solid #ff6600;
                border-radius: 8px;
            }
        """)


# Main Application Window
class SystemMonitor(QWidget):
    def __init__(self):
        super().__init__()

        #Intializing Process Tracker
        self.process_tracker = ProcessStatsTracker()
        self.setWindowTitle("◉ InsightOS")
        self.setGeometry(100, 100, 1200, 900)

        # Set application-wide style
        self.setStyleSheet(self.get_main_stylesheet())

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = self.create_header()
        main_layout.addWidget(header)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(self.get_tab_stylesheet())

        # Create all tabs
        self.create_process_tab()
        self.create_graph_tab()
        self.create_gaming_tab()
        self.create_ml_tab()
        self.create_spec_tab()
        self.create_usage_tab()

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        # Initialize data structures
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.time_counter = list(range(30))

        # ML session variables
        self.active_session = None
        self.sessions = []

        # Timer for updates
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_all)
        self.timer.start()

        # Initialize tabs
        self.show_specs()
        self.show_top_processes()
        self.show_gaming_stats()
        self.update_gpu_info()

    def get_main_stylesheet(self):
        return """
        QWidget {
            background-color: #0a0a0a;
            color: #ffffff;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            font-size: 11px;
        }

        QMainWindow {
            background-color: #0a0a0a;
        }
        """

    def get_tab_stylesheet(self):
        return """
        QTabWidget::pane {
            border: 2px solid #ff6600;
            border-radius: 8px;
            background-color: #1a1a1a;
            margin-top: -2px;
        }

        QTabWidget::tab-bar {
            alignment: center;
        }

        QTabBar::tab {
            background-color: #2a2a2a;
            color: #cccccc;
            padding: 12px 20px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: bold;
            font-size: 12px;
        }

        QTabBar::tab:selected {
            background-color: #ff6600;
            color: #000000;
        }

        QTabBar::tab:hover:!selected {
            background-color: #ff6600;
            color: #000000;
            opacity: 0.7;
        }
        """

    def create_header(self):
        header = QFrame()
        header.setFixedHeight(80)
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6600, stop:0.5 #ff8533, stop:1 #ff6600);
                border: none;
                border-radius: 0px;
            }
            QLabel {
                color: #000000;
                font-size: 24px;
                font-weight: bold;
                background: transparent;
            }
        """)

        layout = QHBoxLayout()
        title = QLabel("◉ InsightOS")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        header.setLayout(layout)

        return header

    def create_process_tab(self):
        self.process_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("🔥 TOP PROCESSES")
        title.setStyleSheet("""
            QLabel {
                color: #ff6600;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #1a1a1a;
                border: 2px solid #ff6600;
                border-radius: 8px;
            }
        """)
        layout.addWidget(title)

        self.process_text = QTextEdit()
        self.process_text.setReadOnly(True)
        self.process_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                border: 2px solid #ff6600;
                border-radius: 8px;
                padding: 10px;
                selection-background-color: #ff6600;
                selection-color: #000000;
            }
        """)
        layout.addWidget(self.process_text)

        self.process_tab.setLayout(layout)
        self.tabs.addTab(self.process_tab, "⚡ Processes")

    def create_graph_tab(self):
        self.graph_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("📊 REAL-TIME PERFORMANCE GRAPHS")
        title.setStyleSheet("""
            QLabel {
                color: #ff6600;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #1a1a1a;
                border: 2px solid #ff6600;
                border-radius: 8px;
            }
        """)
        layout.addWidget(title)

        self.canvas = MplCanvas(self)
        layout.addWidget(self.canvas)

        self.graph_tab.setLayout(layout)
        self.tabs.addTab(self.graph_tab, "📈 Graphs")



    def create_gaming_tab(self):
        self.gaming_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("🎮 GAMING PERFORMANCE CENTER")
        title.setStyleSheet("""
            QLabel {
                color: #ff6600;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #1a1a1a;
                border: 2px solid #ff6600;
                border-radius: 8px;
            }
        """)
        layout.addWidget(title)

        # Stats display
        self.gaming_stats_text = QTextEdit()
        self.gaming_stats_text.setReadOnly(True)
        self.gaming_stats_text.setStyleSheet("""
                    QTextEdit {
                        background-color: #1a1a1a;
                        color: #ffffff;
                        font-family: 'Consolas', 'Monaco', monospace;
                        font-size: 12px;
                        border: 2px solid #ff6600;
                        border-radius: 8px;
                        padding: 0px;
                        selection-background-color: #ff6600;
                        selection-color: #000000;
                    }
                """)
        layout.addWidget(self.gaming_stats_text)

        # Stats display
        self.gaming_text = QTextEdit()
        self.gaming_text.setReadOnly(True)
        self.gaming_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                border: 2px solid #ff6600;
                border-radius: 8px;
                padding: 15px;
                height: 20px ;
                selection-background-color: #ff6600;
                selection-color: #000000;
            }
        """)
        layout.addWidget(self.gaming_text)

        # AI Question section
        ai_section = QFrame()
        ai_section.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 2px solid #ff6600;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        ai_layout = QVBoxLayout()

        ai_title = QLabel("🤖 GPU EXPERT AI")
        ai_title.setStyleSheet("""
            QLabel {
                color: #ff6600;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        ai_layout.addWidget(ai_title)

        self.user_question_input = QTextEdit()
        self.user_question_input.setPlaceholderText("Ask about GPU performance, settings, optimization...")
        self.user_question_input.setFixedHeight(80)
        self.user_question_input.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #ff6600;
                border-radius: 5px;
                padding: 8px;
                font-size: 11px;
            }
            QTextEdit:focus {
                border: 2px solid #ff6600;
            }
        """)
        ai_layout.addWidget(self.user_question_input)

        self.ask_button = QPushButton("🚀 ASK AI EXPERT")
        self.ask_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff6600, stop:1 #e55a00);
                color: #000000;
                font-weight: bold;
                font-size: 12px;
                padding: 12px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff8533, stop:1 #ff6600);
            }
            QPushButton:pressed {
                background: #cc5500;
            }
        """)
        self.ask_button.clicked.connect(self.on_ask_button_clicked)
        ai_layout.addWidget(self.ask_button)

        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setPlaceholderText("AI responses will appear here...")
        self.response_text.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #ff6600;
                border-radius: 5px;
                padding: 10px;
                font-size: 11px;
            }
        """)
        ai_layout.addWidget(self.response_text)

        ai_section.setLayout(ai_layout)
        layout.addWidget(ai_section)

        # Boost button
        boost_btn = QPushButton("⚡ PERFORMANCE BOOST")
        boost_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff3333, stop:1 #cc0000);
                color: #ffffff;
                font-weight: bold;
                font-size: 14px;
                padding: 15px;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff6666, stop:1 #ff3333);
            }
            QPushButton:pressed {
                background: #990000;
            }
        """)
        boost_btn.clicked.connect(self.boost_performance)
        layout.addWidget(boost_btn)

        self.gaming_tab.setLayout(layout)
        self.tabs.addTab(self.gaming_tab, "🎮 Gaming")

    def create_ml_tab(self):
        self.ml_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("🧠 MACHINE LEARNING INSIGHTS")
        title.setStyleSheet("""
            QLabel {
                color: #ff6600;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #1a1a1a;
                border: 2px solid #ff6600;
                border-radius: 8px;
            }
        """)
        layout.addWidget(title)

        # Control panel
        control_panel = QFrame()
        control_panel.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 2px solid #ff6600;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        control_layout = QHBoxLayout()

        # Model name input
        model_label = QLabel("Model:")
        model_label.setStyleSheet("color: #ff6600; font-weight: bold;")
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("Enter model name...")
        self.model_name_input.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #ff6600;
                border-radius: 5px;
                padding: 8px;
                font-size: 11px;
                min-width: 200px;
            }
            QLineEdit:focus {
                border: 2px solid #ff6600;
            }
        """)

        # Buttons
        button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff6600, stop:1 #e55a00);
                color: #000000;
                font-weight: bold;
                font-size: 11px;
                padding: 10px 15px;
                border: none;
                border-radius: 5px;
                min-width: 100px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff8533, stop:1 #ff6600);
            }
            QPushButton:disabled {
                background: #666666;
                color: #999999;
            }
        """

        self.start_button = QPushButton("Start Session")
        self.start_button.setStyleSheet(button_style)
        self.start_button.clicked.connect(self.start_session)

        self.epoch_button = QPushButton("Record Epoch")
        self.epoch_button.setEnabled(False)
        self.epoch_button.setStyleSheet(button_style)
        self.epoch_button.clicked.connect(self.record_epoch)

        self.stop_button = QPushButton("Stop Session")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet(button_style)
        self.stop_button.clicked.connect(self.stop_session)

        control_layout.addWidget(model_label)
        control_layout.addWidget(self.model_name_input)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.epoch_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch()

        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel)

        # Session info
        self.active_session_label = QLabel("No active training session.")
        self.active_session_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 12px;
                padding: 10px;
                background-color: #2a2a2a;
                border: 1px solid #ff6600;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.active_session_label)

        # Session log
        log_label = QLabel("📝 Training Session Log:")
        log_label.setStyleSheet("color: #ff6600; font-weight: bold; font-size: 14px;")
        layout.addWidget(log_label)

        self.session_log = QTextEdit()
        self.session_log.setReadOnly(True)
        self.session_log.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                border: 2px solid #ff6600;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.session_log)

        # Process table
        proc_label = QLabel("⚙️ GPU Processes:")
        proc_label.setStyleSheet("color: #ff6600; font-weight: bold; font-size: 14px;")
        layout.addWidget(proc_label)

        self.process_table = QTableWidget()
        self.process_table.setColumnCount(3)
        self.process_table.setHorizontalHeaderLabels(["PID", "Name", "GPU Memory (MB)"])
        self.process_table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 2px solid #ff6600;
                border-radius: 8px;
                gridline-color: #ff6600;
                selection-background-color: #ff6600;
                selection-color: #000000;
            }
            QHeaderView::section {
                background-color: #ff6600;
                color: #000000;
                font-weight: bold;
                padding: 8px;
                border: none;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333333;
            }
        """)
        layout.addWidget(self.process_table)

        self.ml_tab.setLayout(layout)
        self.tabs.addTab(self.ml_tab, "🧠 ML Mode")

    def create_usage_tab(self):
        self.usage_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title Daily Usage
        title_daily = QLabel("🎮 Daily Usage")
        title_daily.setStyleSheet("""
                       QLabel {
                           color: #ff6600;
                           font-size: 18px;
                           font-weight: bold;
                           padding: 10px;
                           background-color: #1a1a1a;
                           border: 2px solid #ff6600;
                           border-radius: 8px;
                       }
                   """)
        layout.addWidget(title_daily)

        self.current_canvas = BarPlotCanvas()
        layout.addWidget(self.current_canvas)

        # Title Overall Usage
        title_overall = QLabel("🎮 Overall Usage")
        title_overall.setStyleSheet("""
                              QLabel {
                                  color: #ff6600;
                                  font-size: 18px;
                                  font-weight: bold;
                                  padding: 10px;
                                  background-color: #1a1a1a;
                                  border: 2px solid #ff6600;
                                  border-radius: 8px;
                              }
                          """)
        layout.addWidget(title_overall)

        self.overall_canvas = BarPlotCanvas()
        layout.addWidget(self.overall_canvas)

        # Title Last 10 days
        title_history = QLabel("🎮 Last 10 days")
        title_history.setStyleSheet("""
                              QLabel {
                                  color: #ff6600;
                                  font-size: 18px;
                                  font-weight: bold;
                                  padding: 10px;
                                  background-color: #1a1a1a;
                                  border: 2px solid #ff6600;
                                  border-radius: 8px;
                              }
                          """)
        layout.addWidget(title_history)

        self.history_canvas = BarPlotCanvas()
        layout.addWidget(self.history_canvas)

        self.usage_tab.setLayout(layout)
        self.tabs.addTab(self.usage_tab, "📈 Usage")

    def create_spec_tab(self):
        self.spec_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("💻 SYSTEM SPECIFICATIONS")
        title.setStyleSheet("""
            QLabel {
                color: #ff6600;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #1a1a1a;
                border: 2px solid #ff6600;
                border-radius: 8px;
            }
        """)
        layout.addWidget(title)

        self.spec_text = QTextEdit()
        self.spec_text.setReadOnly(True)
        self.spec_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                border: 2px solid #ff6600;
                border-radius: 8px;
                padding: 15px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(self.spec_text)

        self.spec_tab.setLayout(layout)
        self.tabs.addTab(self.spec_tab, "💻 Specs")

    # ML Session Methods
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
        self.active_session_label.setText(f"🔥 Active Session: {model_name}")

    def record_epoch(self):
        if not self.active_session:
            return
        self.active_session.epoch_count += 1

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
            f"🎯 Model: {session.model_name}\n"
            f"📊 Epochs: {session.epoch_count}\n"
            f"⏱️ Duration: {duration_str}\n"
            f"🔥 Peak GPU Load: {session.peak_gpu_load:.1f}%\n"
            f"💾 Peak Memory Used: {session.peak_mem_used:.0f} MB\n"
            f"{'=' * 50}\n"
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
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                self.active_session_label.setText("❌ No GPU detected.")
                self.process_table.setRowCount(0)
                return

            gpu = gpus[0]
            load = gpu.load * 100
            memory_used = gpu.memoryUsed

            # Update active session peaks
            if self.active_session:
                if load > self.active_session.peak_gpu_load:
                    self.active_session.peak_gpu_load = load
                if memory_used > self.active_session.peak_mem_used:
                    self.active_session.peak_mem_used = memory_used

                self.active_session_label.setText(
                    f"🔥 Active: {self.active_session.model_name} | "
                    f"📊 Epochs: {self.active_session.epoch_count} | "
                    f"⚡ GPU: {load:.1f}% | 💾 Memory: {memory_used:.0f} MB"
                )
            else:
                self.active_session_label.setText(f"⚡ GPU Load: {load:.1f}% | 💾 Memory: {memory_used:.0f} MB")

            self.update_process_table(gpu)
        except Exception as e:
            self.active_session_label.setText(f"❌ GPU Error: {str(e)}")

    def update_process_table(self, gpu):
        try:
            if hasattr(gpu, 'processes') and gpu.processes:
                procs = gpu.processes
                self.process_table.setRowCount(len(procs))
                for i, proc in enumerate(procs):
                    pid = proc.get('pid', 'N/A')
                    mem_used = proc.get('used_memory', 0)
                    try:
                        p = psutil.Process(int(pid)) if pid != 'N/A' else None
                        name = p.name() if p else "N/A"
                    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                        name = "N/A"

                    self.process_table.setItem(i, 0, QTableWidgetItem(str(pid)))
                    self.process_table.setItem(i, 1, QTableWidgetItem(name))
                    self.process_table.setItem(i, 2, QTableWidgetItem(f"{mem_used:.0f}"))
            else:
                self.process_table.setRowCount(0)
        except Exception as e:
            print(f"Error updating process table: {e}")
            self.process_table.setRowCount(0)

    def update_all_charts(self):
        current_data = self.process_tracker.fetch_current_day_stats()
        if current_data:
            labels, values = zip(*current_data)
            self.current_canvas.plot(labels, values, 'Current Day Usage', 'Processes', 'Time in Top 5 (s)')
        else:
            self.current_canvas.ax.clear()
            self.current_canvas.ax.text(0.5, 0.5, "No current day data available.", ha='center', va='center',
                                        color='#FFA500')
            self.current_canvas.draw()

        overall_data = self.process_tracker.fetch_overall_stats()
        if overall_data:
            labels, values = zip(*overall_data)
            self.overall_canvas.plot(labels, values, 'Overall Usage', 'Processes', 'Time in Top 5 (s)')
        else:
            self.overall_canvas.ax.clear()
            self.overall_canvas.ax.text(0.5, 0.5,   "No overall stats data available.", ha='center', va='center',
                                        color='#FFA500')
            self.overall_canvas.draw()

        history_data = self.process_tracker.fetch_daily_history_stats()
        if history_data:
            labels, values = zip(*history_data)
            self.history_canvas.plot(labels, values, 'Last 10 Days Usage', 'Date', 'Usage Time (s)')
        else:
            self.history_canvas.ax.clear()
            self.history_canvas.ax.text(0.5, 0.5, "No history summary data available.", ha='center', va='center',
                                        color='#FFA500')
            self.history_canvas.draw()

    def show_gaming_stats(self):
        gpus = GPUtil.getGPUs()
        if not gpus:
            self.gaming_text.setText("No GPU detected.")
            return

        gpu = gpus[0]  # Assume first GPU for now
        fps_estimate = int((1 - gpu.load) * 144)  #  FPS estimate for demo

        text = f""" 🔥 GPU : {gpu.name}
    💾 Total Memory: {gpu.memoryTotal:.0f} MB
    🌡️  Temperature: {gpu.temperature}°C
    ⚡ Driver Version: {getattr(gpu, 'driver', 'N/A')}
    🔌 GPU Load: {gpu.load * 100:.1f}%
    fps : {fps_estimate}fps 
    💿 Memory Used: {gpu.memoryUsed:.0f} MB / {gpu.memoryTotal:.0f} MB
    """

        self.gaming_stats_text.setText(text)

    def show_top_processes(self):
            try:
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

                # Sort by combined CPU and memory usage
                processes.sort(key=lambda x: (x[3] + x[2] / 100), reverse=True)

                display_text = f"""
    🔥 TOP SYSTEM PROCESSES
    {'═' * 70}

    {'PID':<10}{'PROCESS NAME':<25}{'RAM (MB)':<12}{'CPU (%)':<10}{'STATUS':<8}
    {'-' * 70}
    """

                for i, proc in enumerate(processes[:15], 1):
                    # Determine status based on resource usage
                    if proc[3] > 50 or proc[2] > 500:
                        status = "🔴 HIGH"
                    elif proc[3] > 20 or proc[2] > 200:
                        status = "🟡 MED"
                    else:
                        status = "🟢 LOW"

                    rank_icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."

                    display_text += f"{rank_icon} {proc[0]:<8}{proc[1]:<25}{proc[2]:<12.1f}{proc[3]:<10.1f}{status}\n"

                # Add system summary
                cpu_count = psutil.cpu_count()
                memory = psutil.virtual_memory()

                display_text += f"""
    {'-' * 70}
    📊 SYSTEM SUMMARY:
    • Total Processes: {len(processes)}
    • CPU Cores: {cpu_count} 
    • Total RAM: {memory.total / (1024 ** 3):.1f} GB
    • Available RAM: {memory.available / (1024 ** 3):.1f} GB
    • System Load: {psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 'N/A'}
    """

                self.process_text.setText(display_text)

            except Exception as e:
                self.process_text.setText(f"❌ Error retrieving process information: {str(e)}")

    def show_specs(self):
        self.spec_text.setText(self.get_system_info())

    # Gaming Mode Methods
    def on_ask_button_clicked(self):
        user_question = self.user_question_input.toPlainText().strip()
        if not user_question:
            QMessageBox.warning(self, "Input Error", "Please enter your question.")
            return
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
        self.ask_button.setEnabled(False)
        self.response_text.setText("Waiting for response from GPU Expert AI...")

        self.thread = WorkerThread(gpu_condition, user_question)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()


    def on_thread_finished(self, answer):
        self.response_text.setText(answer)
        self.ask_button.setEnabled(True)

    def boost_performance(self):
        try:
            killed = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    if proc.info['cpu_percent'] < 5 and proc.info['memory_info'].rss / 1024 / 1024 < 50:
                        if proc.info['name'] not in ["explorer.exe", "python.exe", "SystemMonitor.exe"]:
                            psutil.Process(proc.info['pid']).terminate()
                            killed.append(proc.info['name'])
                            if len(killed) >= 5:  # Limit to 5 processes
                                break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            boost_text = f"\n🚀 PERFORMANCE BOOST ACTIVATED!\n"
            boost_text += f"🔥 Terminated {len(killed)} background processes\n"
            if killed:
                boost_text += f"📋 Processes: {', '.join(killed[:3])}"
                if len(killed) > 3:
                    boost_text += f" and {len(killed) - 3} more..."
            else:
                boost_text += "✅ No background processes needed termination"

            self.gaming_text.append(boost_text)
        except Exception as e:
            self.gaming_text.append(f"\n❌ Boost failed: {str(e)}")

    def update_all(self):
        """Update all data every second"""
        try:
            # Always update metrics for graphs
            self.update_metrics()
            self.process_tracker.log_top_processes()
            self.update_all_charts()

            # Get current tab to optimize updates
            current_tab = self.tabs.currentIndex()

            # Update tabs based on which one is currently visible
            if current_tab == 0:  # Processes tab
                self.show_top_processes()
            elif current_tab == 2:  # Gaming tab
                self.show_gaming_stats()
            elif current_tab == 3:  # ML tab
                self.update_gpu_info()
            elif current_tab == 4:  # Specs tab
                # Update specs less frequently (every 5 seconds)
                if not hasattr(self, '_spec_counter'):
                    self._spec_counter = 0
                self._spec_counter += 1
                if self._spec_counter >= 5:
                    self.show_specs()
                    self._spec_counter = 0

        except Exception as e:
            print(f"Error in update_all: {e}")

    def update_metrics(self):
        try:
            # Get current metrics
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent

            gpus = GPUtil.getGPUs()
            gpu_percent = gpus[0].load * 100 if gpus else 0

            # Update data lists
            self.cpu_usage.append(cpu_percent)
            self.cpu_usage = self.cpu_usage[-30:]  # Keep last 30 data points

            self.ram_usage.append(ram_percent)
            self.ram_usage = self.ram_usage[-30:]

            self.gpu_usage.append(gpu_percent)
            self.gpu_usage = self.gpu_usage[-30:]

            # Clear and update plots
            self.canvas.cpu_ax.clear()
            self.canvas.ram_ax.clear()
            self.canvas.gpu_ax.clear()

            # Time axis
            time_points = list(range(len(self.cpu_usage)))

            # CPU Plot
            self.canvas.cpu_ax.plot(time_points, self.cpu_usage,
                                    color="#ff6600", linewidth=2, marker='o', markersize=3)
            self.canvas.cpu_ax.fill_between(time_points, self.cpu_usage, alpha=0.3, color="#ff6600")
            self.canvas.cpu_ax.set_title("🧠 CPU Usage (%)", color="#ff6600", fontsize=12, fontweight='bold')
            self.canvas.cpu_ax.set_ylim(0, 100)
            self.canvas.cpu_ax.set_facecolor('#1a1a1a')

            # RAM Plot
            self.canvas.ram_ax.plot(time_points, self.ram_usage,
                                    color="#00ff88", linewidth=2, marker='s', markersize=3)
            self.canvas.ram_ax.fill_between(time_points, self.ram_usage, alpha=0.3, color="#00ff88")
            self.canvas.ram_ax.set_title("💾 RAM Usage (%)", color="#00ff88", fontsize=12, fontweight='bold')
            self.canvas.ram_ax.set_ylim(0, 100)
            self.canvas.ram_ax.set_facecolor('#1a1a1a')

            # GPU Plot
            self.canvas.gpu_ax.plot(time_points, self.gpu_usage,
                                    color="#ff0066", linewidth=2, marker='^', markersize=3)
            self.canvas.gpu_ax.fill_between(time_points, self.gpu_usage, alpha=0.3, color="#ff0066")
            self.canvas.gpu_ax.set_title("🎮 GPU Usage (%)", color="#ff0066", fontsize=12, fontweight='bold')
            self.canvas.gpu_ax.set_ylim(0, 100)
            self.canvas.gpu_ax.set_xlabel("Time (seconds)", color="#ff6600")
            self.canvas.gpu_ax.set_facecolor('#1a1a1a')

            # Style all axes
            for ax in [self.canvas.cpu_ax, self.canvas.ram_ax, self.canvas.gpu_ax]:
                ax.tick_params(colors='#ff6600', labelsize=8)
                ax.spines['bottom'].set_color('#ff6600')
                ax.spines['top'].set_color('#ff6600')
                ax.spines['right'].set_color('#ff6600')
                ax.spines['left'].set_color('#ff6600')
                ax.grid(True, alpha=0.2, color='#ff6600')

            self.canvas.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"Error updating metrics: {e}")

    def get_system_info(self):
            try:
                uname = platform.uname()
                info = f"""
    🖥️  SYSTEM OVERVIEW
    {'═' * 50}

    💻 System: {uname.system}
    🌐 Node Name: {uname.node}
    📦 Release: {uname.release}
    🔧 Version: {uname.version}
    ⚙️  Machine: {uname.machine}
    🧠 Processor: {uname.processor}

    {'═' * 50}
    🎮 GPU SPECIFICATIONS
    {'═' * 50}
    """
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        for i, gpu in enumerate(gpus):
                            info += f"""
    🔥 GPU {i + 1}: {gpu.name}
    💾 Total Memory: {gpu.memoryTotal:.0f} MB
    🌡️  Temperature: {gpu.temperature}°C
    ⚡ Driver Version: {getattr(gpu, 'driver', 'N/A')}
    🔌 GPU Load: {gpu.load * 100:.1f}%
    💿 Memory Used: {gpu.memoryUsed:.0f} MB / {gpu.memoryTotal:.0f} MB
    """
                    else:
                        info += "\n❌ No GPU detected or drivers not installed"
                except Exception as e:
                    info += f"\n❌ GPU Error: {str(e)}"

                # Add CPU info
                info += f"""

    {'═' * 50}
    🧠 CPU SPECIFICATIONS  
    {'═' * 50}
    🔢 CPU Cores: {psutil.cpu_count(logical=False)} Physical, {psutil.cpu_count(logical=True)} Logical
    ⚡ CPU Frequency: {psutil.cpu_freq().current:.0f} MHz (Max: {psutil.cpu_freq().max:.0f} MHz)
    """

                # Add Memory info
                memory = psutil.virtual_memory()
                info += f"""

    {'═' * 50}
    💾 MEMORY SPECIFICATIONS
    {'═' * 50}
    🎯 Total RAM: {memory.total / (1024 ** 3):.2f} GB
    ✅ Available RAM: {memory.available / (1024 ** 3):.2f} GB
    📊 Memory Usage: {memory.percent}%
    """

                # Add Disk info
                disk = psutil.disk_usage('/')
                info += f"""

    {'═' * 50}
    💿 STORAGE SPECIFICATIONS
    {'═' * 50}
    📦 Total Disk Space: {disk.total / (1024 ** 3):.2f} GB
    ✅ Free Disk Space: {disk.free / (1024 ** 3):.2f} GB
    📊 Disk Usage: {(disk.used / disk.total) * 100:.1f}%
    """

                return info
            except Exception as e:
                return f"❌ Error retrieving system information: {str(e)}"

    def reset_boost_button(self, button):
        button.setEnabled(True)
        button.setText("⚡ PERFORMANCE BOOST")


if __name__ == "__main__":
    try:

        app = QApplication(sys.argv)
        # Set application style
        app.setStyle('Fusion')
        monitor = SystemMonitor()
        monitor.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application failed to start: {e}")
        sys.exit(1)