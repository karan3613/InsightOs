import google.generativeai as genai

def get_performance_suggestions(gpu_data):
    try:
        # Set your Gemini API key
        genai.configure(api_key="YOUR_GEMINI_API_KEY")

        model = genai.GenerativeModel('gemini-pro')

        prompt = (
            "I'm monitoring GPU usage for ML training and gaming. Here is the current GPU data:\n\n"
            f"{gpu_data}\n\n"
            "Based on this data, suggest actionable tips to optimize GPU performance, improve memory management, and ensure smooth system operation. Focus on VRAM handling, temperature management, and compute task prioritization."
        )

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"Error fetching suggestion: {e}"
