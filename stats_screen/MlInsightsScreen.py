import sys
import time
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QTextEdit, QTableWidget, QTableWidgetItem, QLineEdit, QHBoxLayout, QMessageBox
)
from PyQt5.QtCore import QTimer

import GPUtil
import psutil


class TrainingSession:
    def __init__(self, model_name):
        self.model_name = model_name
        self.start_time = time.time()
        self.end_time = None
        self.epoch_count = 0
        self.peak_gpu_load = 0
        self.peak_mem_used = 0

    def end_session(self):
        self.end_time = time.time()

    def duration(self):
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class GPUMonitorApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ML Training GPU Monitor - Epoch Recorder")
        self.setGeometry(100, 100, 900, 700)

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

        self.setLayout(main_layout)

        # Timer to update GPU info and process list
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gpu_info)
        self.timer.start(1000)  # every second

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
