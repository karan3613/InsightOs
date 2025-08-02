import sys
import GPUtil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QTextEdit, QPushButton, QMessageBox, QHBoxLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

import google.generativeai as genai

# Configure your Gemini API key here
API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

def get_gpu_condition_string():
    """
    Use GPUtil to fetch GPU stats and return a formatted string describing
    the current GPU condition.
    """
    gpus = GPUtil.getGPUs()
    if not gpus:
        return "No GPU detected."

    gpu = gpus[0]  # Assuming single GPU, extend if multiple GPUs needed
    condition = (
        f"GPU Name: {gpu.name}\n"
        f"Temperature: {gpu.temperature}°C\n"
        f"GPU Load: {gpu.load * 100:.1f}%\n"
        f"Memory Usage: {gpu.memoryUsed} MB / {gpu.memoryTotal} MB\n"
        f"Memory Utilization: {gpu.memoryUtil * 100:.1f}%"
    )
    return condition

def query_gemini_gpu_bot(gpu_condition: str, user_question: str) -> str:
    """
    Query Gemini AI with GPU condition context and user question using google-generativeai library.
    """
    system_prompt = (
        "You are a highly knowledgeable GPU expert specializing in graphics cards, "
        "their specifications, features like V-Sync, DLSS, and game performance optimization. "
        "A user will provide you with the current GPU condition and then ask a question. "
        "Use the GPU condition to give tailored advice or explanations. "
        "Answer strictly with detailed technical explanations and practical advice "
        "on GPU settings based on the given scenario, such as low FPS, GPU temperature, "
        "type of game, and hardware capabilities. Avoid unrelated topics."
    )

    user_input = f"GPU condition:\n{gpu_condition}\n\nUser question: {user_question}"

    full_prompt = f"{system_prompt}\n\n{user_input}"

    response = model.generate_content(full_prompt)
    return response.text


class WorkerThread(QThread):
    finished = pyqtSignal(str)

    def __init__(self, gpu_condition, user_question):
        super().__init__()
        self.gpu_condition = gpu_condition
        self.user_question = user_question

    def run(self):
        answer = query_gemini_gpu_bot(self.gpu_condition, self.user_question)
        self.finished.emit(answer)


class GPUInsightApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.start_gpu_update_timer()

    def init_ui(self):
        self.setWindowTitle("GPU Insight & Expert AI Bot")
        self.resize(700, 700)

        self.layout = QVBoxLayout()

        # GPU stats label (multiline)
        self.gpu_stats_label = QLabel("GPU stats loading...")
        self.gpu_stats_label.setAlignment(Qt.AlignLeft)
        self.gpu_stats_label.setStyleSheet("font-family: monospace; font-size: 14px;")
        self.layout.addWidget(QLabel("Current GPU Condition:"))
        self.layout.addWidget(self.gpu_stats_label)

        # User prompt text area
        self.layout.addWidget(QLabel("Enter your GPU-related question:"))
        self.user_question_input = QTextEdit()
        self.user_question_input.setPlaceholderText("Example: Should I enable DLSS for better FPS?")
        self.user_question_input.setFixedHeight(100)
        self.layout.addWidget(self.user_question_input)

        # Button to ask AI
        self.ask_button = QPushButton("Ask GPU Expert AI")
        self.layout.addWidget(self.ask_button)

        # AI response text area (read-only)
        self.layout.addWidget(QLabel("Response from GPU Expert AI:"))
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setStyleSheet("background-color: #f4f4f4;")
        self.layout.addWidget(self.response_text)

        self.setLayout(self.layout)

        self.ask_button.clicked.connect(self.on_ask_button_clicked)

    def start_gpu_update_timer(self):
        # Update GPU stats every 3 seconds
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gpu_stats)
        self.timer.start(3000)
        self.update_gpu_stats()

    def update_gpu_stats(self):
        condition = get_gpu_condition_string()
        self.gpu_stats_label.setText(condition)

    def on_ask_button_clicked(self):
        gpu_condition = self.gpu_stats_label.text()
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


def main():
    app = QApplication(sys.argv)
    window = GPUInsightApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
