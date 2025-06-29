import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from models.base_model import BaseModel
from typing import List, Dict
from utils.prompt_template import generate_prompt

load_dotenv()

class GeminiModel(BaseModel):
    def __init__(self, api_key: str = None, temperature: float = 0.7, model_name: str = "gemini-1.5-pro-latest"):
        self.model_name = model_name
        self.temperature = temperature

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found in .env under GOOGLE_API_KEY.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def classify(self, text: str, labels: List[str]) -> Dict:
        prompt = generate_prompt(text, labels)

        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()

            # Try parsing the JSON content from the response
            json_str = self._extract_json(raw_text)
            result = json.loads(json_str)

            # Normalize scores and create binned labels
            parsed_scores = {
                label: float(result.get("labels", {}).get(label, 0.0))
                for label in labels
            }
            binned_scores = {
                label: 1 if parsed_scores[label] >= 0.5 else 0
                for label in labels
            }

            explanation = result.get("explanation", "")
            return {"labels": parsed_scores, "binned_labels": binned_scores, "explanation": explanation}

        except Exception as e:
            print(f"Gemini classification failed: {e}")
            return {
                "labels": {label: 0.0 for label in labels},
                "binned_labels": {label: 0 for label in labels},
                "explanation": "Parsing failed or API call failed"
            }

    def _extract_json(self, text: str) -> str:
        """
        Attempts to extract a valid JSON object from the model's raw response.
        """
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            return text[start:end]
        raise ValueError("Failed to extract JSON object from Gemini response.")

