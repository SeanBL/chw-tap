import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from models.base_model import BaseModel
from typing import List, Dict
from utils.prompt_template import generate_prompt
from utils.model_safety_mixin import ModelSafetyMixin  # NEW

load_dotenv()

class GeminiModel(BaseModel, ModelSafetyMixin):
    def __init__(self, api_key: str = None, temperature: float = 0.0, model_name: str = "gemini-1.5-pro-latest"):
        self.model_name = model_name
        self.temperature = temperature

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found in .env under GOOGLE_API_KEY.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def classify(self, text: str, labels: List[str], normalized_labels: Dict[str, str]) -> Dict:
        prompt = generate_prompt(text, labels)

        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()

            print(f"\n[DEBUG] Raw Gemini output:\n{raw_text}\n")

            json_str = self._extract_json(raw_text)
            result = json.loads(json_str)

            score_block = result.get("labels", result)
            explanation = result.get("explanation", raw_text.replace(json_str, "").strip())

            normalized_block = {}
            for k, v in score_block.items():
                norm_key = self._normalize_label(k)
                for defined_label in labels:
                    if self._normalize_label(defined_label) == norm_key:
                        normalized_block[defined_label] = v
                        break
                else:
                    print(f"⚠️ Unexpected label from model: '{k}' → normalized as '{norm_key}'")

            for k in score_block:
                print(f"Normalized '{k}' → '{self._normalize_label(k)}'")

            parsed_scores = {
                canonical_label: float(normalized_block.get(norm_label, 0.0))
                for norm_label, canonical_label in normalized_labels.items()
            }

            binned_scores = {
                label: 1 if parsed_scores.get(label, 0.0) >= 0.5 else 0 for label in labels
            }

            self._warn_on_low_scores(parsed_scores, explanation, normalized_labels)

            return {
                "labels": parsed_scores,
                "binned_labels": binned_scores,
                "explanation": explanation
            }

        except Exception as e:
            print(f"⚠️ Gemini classification failed: {e}")
            return {
                "labels": {label: 0.0 for label in labels},
                "binned_labels": {label: 0 for label in labels},
                "explanation": "Parsing failed or API call failed"
            }

