import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from models.base_model import BaseModel
from typing import List, Dict

load_dotenv()

class GeminiModel(BaseModel):
    def __init__(self, model_name="gemini-pro"):
        self.model_name = model_name
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found in .env under GOOGLE_API_KEY.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)

def classify(self, text: str, labels: List[str]) -> Dict:
    prompt = f"""
        Classify the following testimonial using the provided labels.

        Return only a JSON object in this format:
        {{
          "labels": {{
            "label1": score (0 to 1),
            ...
          }},
          "explanation": "brief explanation"
        }}

        Testimonial:
        \"{text}\"

        Labels: {labels}
    """

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
