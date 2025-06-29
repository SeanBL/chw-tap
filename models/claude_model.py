import anthropic
import os
import json
from typing import List, Dict
from models.base_model import BaseModel
from dotenv import load_dotenv
from utils.prompt_template import generate_prompt

load_dotenv()

class ClaudeModel(BaseModel):
    def __init__(self, api_key: str = None, temperature: float = 0.7, model: str = "claude-opus-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model_name = model
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def classify(self, text: str, labels: List[str]) -> Dict:
        prompt = generate_prompt(text, labels)

        try:
            response = self.client.messages.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            reply = response.content[0].text
            print("\n[DEBUG] Claude raw output:\n", reply)
            return self.parse_output(reply, labels)
        except Exception as e:
            print("[ERROR] Failed to parse Claude output:\n", text)
            return {
                "labels": {label: 0.0 for label in labels},
                "binned_labels": {label: 0 for label in labels},
                "explanation": f"API call failed: {str(e)}"
            }

    def parse_output(self, text: str, labels: List[str]) -> Dict:
        def bin_score(score: float) -> int:
            if score >= 0.66:
                return 2
            elif score >= 0.33:
                return 1
            else:
                return 0

        try:
            data = json.loads(text)
            parsed_scores = {
                label: float(data.get("labels", {}).get(label, 0.0))
                for label in labels
            }
            binned = {label: bin_score(score) for label, score in parsed_scores.items()}
            explanation = data.get("explanation", "")
            return {"labels": parsed_scores, "binned_labels": binned, "explanation": explanation}
        except Exception as e:
            return {
                "labels": {label: 0.0 for label in labels},
                "binned_labels": {label: 0 for label in labels},
                "explanation": f"Failed to parse JSON: {str(e)}"
            }

