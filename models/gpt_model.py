import openai
from models.base_model import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
import os

load_dotenv()

class GPTModel(BaseModel):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def classify(self, text: str, labels: List[str]) -> Dict[str, any]:
        prompt = (
            f"Classify the following testimonial into any of these categories: {', '.join(labels)}.\n\n"
            f"Testimonial:\n\"{text}\"\n\n"
            f"Return a valid Python dictionary with categories as keys and confidence scores (0 to 1) as values. "
            f"Only include categories that apply. Then provide a brief explanation of your choices."
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes health worker testimonials."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            content = response["choices"][0]["message"]["content"].strip()

            # Attempt to parse the output
            import re, ast
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
            print("⚠️ GPT-4 classification error:", str(e))
            return {
                "labels": {},
                "explanation": "Parsing failed or API call failed"
            }