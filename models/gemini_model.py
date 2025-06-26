import os
import re
import ast
from typing import List, Dict
from dotenv import load_dotenv
from models.base_model import BaseModel

import google.generativeai as genai

load_dotenv()

class GeminiModel(BaseModel):
    def __init__(self, api_key: str = None, model_name: str = "models/gemini-1.5-pro-latest"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def classify(self, text: str, labels: List[str]) -> Dict[str, any]:
        prompt = (
            f"Classify the following testimonial into any of these categories: {', '.join(labels)}.\n\n"
            f"Testimonial:\n\"{text}\"\n\n"
            f"Return a valid Python dictionary with categories as keys and confidence scores (0 to 1) as values. "
            f"Only include categories that apply. Then provide a brief explanation of your choices."
        )

        try:
            response = self.model.generate_content(prompt)
            content = response.text.strip()

            match = re.search(r'(\{.*?\})', content, re.DOTALL)
            if match:
                dict_str = match.group(1)
                explanation = content.replace(dict_str, "").strip()
                output_dict = ast.literal_eval(dict_str)
                return {
                    "labels": {k: float(v) for k, v in output_dict.items()},
                    "explanation": explanation
                }

        except Exception as e:
            print("⚠️ Gemini classification error:", str(e))
            return {
                "labels": {},
                "explanation": "Parsing failed or API call failed"
            }
