import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
from models.base_model import BaseModel
from utils.prompt_template import generate_prompt
from utils.model_safety_mixin import ModelSafetyMixin  # Shared mixin

load_dotenv()

class GPTModel(BaseModel, ModelSafetyMixin):
    def __init__(self, api_key: str = None, model: str = "gpt-4", temperature: float = 0.7):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key)

    def classify(self, text: str, labels: List[str], normalized_labels: Dict[str, str]) -> Dict:
        prompt = generate_prompt(text, labels)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": "You are a helpful classifier."},
                          {"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            reply = response.choices[0].message.content.strip()
            print("\n[DEBUG] GPT raw output:\n", reply)

            return self._parse_output(reply, labels, normalized_labels)

        except Exception as e:
            print("[ERROR] GPT API call failed:", e)
            return {
                "labels": {label: 0.0 for label in labels},
                "binned_labels": {label: 0 for label in labels},
                "explanation": f"API call failed: {str(e)}"
            }

    def _parse_output(self, text: str, labels: List[str], normalized_labels: Dict[str, str]) -> Dict:
        def bin_score(score: float) -> int:
            return 1 if score >= 0.5 else 0

        try:
            json_str = self._extract_json(text)
            data = json.loads(json_str)

            score_block = data.get("labels", data)
            explanation = data.get("explanation", text.replace(json_str, "").strip())

            normalized_block = {}
            for k, v in score_block.items():
                norm_key = self._normalize_label(k)
                for defined_label in labels:
                    if self._normalize_label(defined_label) == norm_key:
                        normalized_block[defined_label] = v
                        break
                else:
                    print(f"⚠️ Unexpected label from GPT: '{k}' → normalized as '{norm_key}'")

            for k in score_block:
                print(f"Normalized '{k}' → '{self._normalize_label(k)}'")

            parsed_scores = {
                canonical_label: float(normalized_block.get(norm_label, 0.0))
                for norm_label, canonical_label in normalized_labels.items()
            }

            binned_scores = {
                label: bin_score(parsed_scores.get(label, 0.0)) for label in labels
            }

            self._warn_on_low_scores(parsed_scores, explanation, normalized_labels)

            return {
                "labels": parsed_scores,
                "binned_labels": binned_scores,
                "explanation": explanation
            }

        except Exception as e:
            return {
                "labels": {label: 0.0 for label in labels},
                "binned_labels": {label: 0 for label in labels},
                "explanation": f"Failed to parse JSON: {str(e)}"
            }


