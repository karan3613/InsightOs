import traceback

from PyQt5.QtCore import Qt, QThread, pyqtSignal
import google.generativeai as genai

def query_gemini_gpu_bot(gpu_condition: str, user_question: str) -> str:
    """
    Query Gemini AI with GPU condition context and user question using google-generativeai library.
    """
    # Replace with your actual Gemini API Key
    API_KEY = "YOUR_API_KEY_HERE"
    genai.configure(api_key=API_KEY)
    # Initialize the Gemini model
    model = genai.GenerativeModel("gemini-2.0-flash")

    system_prompt = (
        "You are a highly knowledgeable GPU expert specializing in graphics cards, "
        "their specifications, features like V-Sync, DLSS, and game performance optimization. "
        "A user will provide you with the current GPU condition and then ask a question. "
        "Use the GPU condition to give tailored advice or explanations. "
        "Answer strictly with detailed technical explanations and practical advice "
        "on GPU settings based on the given scenario, such as low FPS, GPU temperature, "
        "type of game, and hardware capabilities. Avoid unrelated topics."
    )

    user_input = f"GPU condition: {gpu_condition}\nUser question: {user_question}"

    full_prompt = f"{system_prompt}\n\n{user_input}"
    response = model.generate_content(full_prompt)
    print(response.text)
    return response.text


class WorkerThread(QThread):
    finished = pyqtSignal(str)

    def __init__(self, gpu_condition, user_question):
        super().__init__()

        self.gpu_condition = gpu_condition
        self.user_question = user_question

    def run(self):
        try:
            answer = query_gemini_gpu_bot(self.gpu_condition, self.user_question)
        except Exception as e:
            answer = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
        self.finished.emit(answer)
