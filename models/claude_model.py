import anthropic
import os
import json
import re
from typing import List, Dict
from models.base_model import BaseModel
from dotenv import load_dotenv
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from utils.prompt_template import generate_prompt
from utils.label_utils import normalize_label, explanation_contains_label_stem  # Make sure this exists

load_dotenv()

class ClaudeModel(BaseModel):
    def __init__(self, api_key: str = None, temperature: float = 0.7, model: str = "claude-opus-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model_name = model
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def classify(self, text: str, labels: List[str], normalized_labels: Dict[str, str]) -> Dict:
        prompt = generate_prompt(text, labels)

        try:
            response = self.client.messages.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            reply = response.content[0].text
            print("\n[DEBUG] Claude raw output:\n", reply)
            return self._parse_output(reply, labels, normalized_labels)

        except Exception as e:
            print("[ERROR] Claude API call failed:", e)
            return {
                "labels": {label: 0.0 for label in labels},
                "binned_labels": {label: 0 for label in labels},
                "explanation": f"API call failed: {str(e)}"
            }

    def _parse_output(self, text: str, labels: List[str], normalized_labels: Dict[str, str]) -> Dict:
        try:
            json_str = self._extract_json(text)
            data = json.loads(json_str)

            score_block = data.get("labels", data)
            explanation = data.get("explanation", text.replace(json_str, "").strip())

            # Normalize model keys and map to config labels
            normalized_block = {}
            for k, v in score_block.items():
                norm_key = normalize_label(k)
                for defined_label in labels:
                    if normalize_label(defined_label) == norm_key:
                        normalized_block[defined_label] = v
                        break
                else:
                    print(f"⚠️ Unexpected label from Claude: '{k}' → normalized as '{norm_key}'")

            # Final score dict
            parsed_scores = {
                canonical_label: float(normalized_block.get(norm_label, 0.0))
                for norm_label, canonical_label in normalized_labels.items()
            }

            binned = {
                label: 1 if parsed_scores.get(label, 0.0) >= 0.5 else 0 for label in labels
            }

            # Explanation-based warning
            for norm_label, canonical_label in normalized_labels.items():
                if explanation_contains_label_stem(norm_label, explanation) and parsed_scores.get(canonical_label, 0.0) < 0.1:
                    print(f"⚠️ Warning: '{canonical_label}' mentioned in explanation (stem match) but has very low score ({parsed_scores[canonical_label]})")

            return {
                "labels": parsed_scores,
                "binned_labels": binned,
                "explanation": explanation
            }

        except Exception as e:
            return {
                "labels": {label: 0.0 for label in labels},
                "binned_labels": {label: 0 for label in labels},
                "explanation": f"Failed to parse JSON: {str(e)}"
            }

    def _extract_json(self, text: str) -> str:
        text = text.strip("` \n")
        text = re.sub(r'//.*', '', text)
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1 and end > start:
            return text[start:end]
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        raise ValueError("No valid JSON object found in Claude output.")


