import requests
from models.base_model import BaseModel
from typing import List, Dict
import re
import ast 

class OllamaModel(BaseModel):
    def __init__(self, model_name="mistral"):
        self.api_url = "http://localhost:11434/api/generate"
        self.model_name = model_name

    def classify(self, text, labels):
        prompt = (
            f"Classify the following testimonial into any of these categories: {', '.join(labels)}.\n\n"
            f"Testimonial:\n\"{text}\"\n\n"
            f"Return a valid Python dictionary with categories as keys and a confidence score (0 to 1) as values. "
            f"Only include categories that apply. Then provide a brief explanation of your choices."
        )

        response = requests.post(self.api_url, json={
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        })

        raw_output = response.json()["response"]

        try:
            match = re.search(r'(\{.*?\})', raw_output, re.DOTALL)
            if match:
                dict_str = match.group(1)
                explanation = raw_output.replace(dict_str, '').strip()
                output_dict = ast.literal_eval(dict_str)
                return {
                    "labels": {k: float(v) for k, v in output_dict.items()},
                    "explanation": explanation
                }

        except Exception as e:
            print("⚠️ Failed to parse response from Ollama model:", raw_output)
            print("Error:", str(e))
            return {
                "labels": {},
                "explanation": "Parsing failed"
            }