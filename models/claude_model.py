import anthropic
import os
from models.base_model import BaseModel
from typing import List, Dict
import re, ast
from dotenv import load_dotenv

load_dotenv()

class ClaudeModel(BaseModel):
    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def classify(self, text: str, labels: List[str]) -> Dict[str, any]:
        prompt = (
            f"Classify the following testimonial into any of these categories: {', '.join(labels)}.\n\n"
            f"Testimonial:\n\"{text}\"\n\n"
            f"Return a valid Python dictionary with categories as keys and confidence scores (0 to 1) as values. "
            f"Only include categories that apply. Then provide a brief explanation of your choices."
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.content[0].text.strip()

            match = re.search(r'(\{.*?\})', content, re.DOTALL)
            if match:
                dict_str = match.group(1)
                explanation = content.replace(dict_str, "").strip()
                output_dict = ast.literal_eval(dict_str)
                return {
                    "labels": {k: float(v) for k, v in output_dict.items()},
                    "explanation": explanation
                }

        except Exception as e:
            print("⚠️ Claude classification error:", str(e))
            return {
                "labels": {},
                "explanation": "Parsing failed or API call failed"
            }
