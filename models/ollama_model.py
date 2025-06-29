import requests
from models.base_model import BaseModel
from typing import List, Dict
import re
import json
from utils.prompt_template import generate_prompt

class OllamaModel(BaseModel):
    def __init__(self, model_name="mistral", temperature: float = 0.0):
        self.api_url = "http://localhost:11434/api/generate"
        self.model_name = model_name
        self.temperature = temperature

    def classify(self, text: str, labels: List[str]) -> Dict:
        prompt = generate_prompt(text, labels)

        try:
            response = requests.post(self.api_url, json={
                "model": self.model_name,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False
            })

            raw_output = response.json().get("response", "")

            print(f"\n[DEBUG] Raw Ollama output:\n{raw_output}\n")

            json_str = self._extract_json(raw_output)
            output_dict = json.loads(json_str)

            # Handle nested vs flat JSON
            score_block = output_dict.get("labels", output_dict)  # fallback if not nested
            explanation = output_dict.get("explanation", raw_output.replace(json_str, "").strip())

            parsed_scores = {
                label: float(score_block.get(label, 0.0)) for label in labels
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

    def _extract_json(self, text: str) -> str:
        """
        Extract the first JSON object from possibly noisy text output.
        Attempts slicing first, then regex as a fallback.
        """
        text = text.strip("` \n")

        # Try slicing from first { to last }
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1 and end > start:
            try:
                return text[start:end]
            except Exception:
                pass

        # Fallback: Regex (slower but sometimes more forgiving)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)

        raise ValueError("No valid JSON object found in Ollama output.")