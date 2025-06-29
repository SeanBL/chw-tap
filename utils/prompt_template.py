from typing import List

def generate_prompt(text: str, labels: list[str]) -> str:
    return f"""
You are a helpful assistant. Your task is to classify the testimonial into relevant categories.

Return only a JSON object in this format:
{{
  "labels": {{
    "label1": score (float between 0 and 1),
    "label2": score,
    ...
  }},
  "explanation": "A short explanation of how the labels were assigned"
}}

Testimonial:
\"\"\"{text}\"\"\"

Available categories: {', '.join(labels)}
""".strip()
