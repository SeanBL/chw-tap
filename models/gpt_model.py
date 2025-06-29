from openai import OpenAI
from dotenv import load_dotenv
import os
from models.base_model import BaseModel
from typing import List, Dict
import json
from utils.prompt_template import generate_prompt

load_dotenv()

class GPTModel(BaseModel):
    def __init__(self, api_key: str = None, model: str = "gpt-4", temperature: float = 0.7):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key)

    def classify(self, text: str, labels: List[str]) -> Dict:
        prompt = generate_prompt(text, labels)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": "You are a helpful classifier."}, prompt],
                temperature=self.temperature
            )
            reply = response.choices[0].message.content
            return self.parse_output(reply, labels)
        except Exception as e:
            return {
                "labels": {label: 0.0 for label in labels},
                "binned_labels": {label: 0 for label in labels},
                "explanation": f"Parsing failed or API call failed: {str(e)}"
            }

    def parse_output(self, text: str, labels: List[str]) -> Dict:
        def bin_score(score: float) -> int:
            if score < 0.34:
                return 0
            elif score < 0.67:
                return 1
            else:
                return 2

        try:
            data = json.loads(text)
            parsed_scores = {
                label: float(data.get("labels", {}).get(label, 0.0))
                for label in labels
            }
            binned_scores = {
                label: bin_score(parsed_scores[label])
                for label in labels
            }
            explanation = data.get("explanation", "")
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

