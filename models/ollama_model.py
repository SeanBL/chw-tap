import requests
from models.base_model import BaseModel
from typing import List, Dict
import re
import ast

class OllamaModel(BaseModel):
    def __init__(self, model_name="mistral", temperature: float = 0.7):
        self.api_url = "http://localhost:11434/api/generate"
        self.model_name = model_name
        self.temperature = temperature

def classify(self, text: str, labels: List[str]) -> Dict:
    prompt = (
        f"Classify the following testimonial into the categories: {', '.join(labels)}.\n\n"
        f"Testimonial:\n\"{text}\"\n\n"
        f"Return a Python dictionary with each category as a key and a confidence score (0 to 1) as the value. "
        f"Only include categories that apply. Then provide a short explanation below the dictionary."
    )

    try:
        response = requests.post(self.api_url, json={
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False
        })

        raw_output = response.json().get("response", "")

        match = re.search(r'(\{.*?\})', raw_output, re.DOTALL)
        if match:
            dict_str = match.group(1)
            explanation = raw_output.replace(dict_str, '').strip()
            output_dict = ast.literal_eval(dict_str)

            # Normalize and bin scores
            parsed_scores = {
                label: float(output_dict.get(label, 0.0)) for label in labels
            }
            binned_scores = {
                label: 1 if parsed_scores[label] >= 0.5 else 0 for label in labels
            }

            return {
                "labels": parsed_scores,
                "binned_labels": binned_scores,
                "explanation": explanation
            }

    except Exception as e:
        print("⚠️ Failed to parse response from Ollama model:", raw_output)
        print("Error:", str(e))
        return {
            "labels": {label: 0.0 for label in labels},
            "binned_labels": {label: 0 for label in labels},
            "explanation": f"Parsing failed or API call failed: {str(e)}"
        }
