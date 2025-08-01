import anthropic
import os
import json
import time
from math import ceil
from typing import List, Dict
from dotenv import load_dotenv
from models.base_model import BaseModel
from utils.prompt_template import generate_prompt
from utils.model_safety_mixin import ModelSafetyMixin  # Shared mixin

load_dotenv()

class ClaudeModel(BaseModel, ModelSafetyMixin):
    def __init__(self, api_key: str = None, temperature: float = 0.0, model: str = "claude-opus-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model_name = model
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.cache = {}  # Simple in-memory cache for prompt -> result

    def classify(
        self,
        text: str,
        labels: List[str],
        normalized_labels: Dict[str, str],
        concept_definitions: Dict[str, Dict] = None,
        include_explanations: bool = False
    ) -> Dict:

        all_scores = {}
        explanations = []

        try:
            for i, label in enumerate(labels):
                print(f"\n📌 Scoring concept {i+1}/{len(labels)} → {label}")
                concept_label = [label] 
                
                prompt = generate_prompt(
                    text=text,
                    labels=concept_label,
                    concept_definitions=concept_definitions or {}
                )

                if prompt in self.cache:
                    print("✅ Cache hit — skipping API call.")
                    raw_output = self.cache[prompt]
                else:
                    response = self.client.messages.create(
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=2048,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    print("ANTHROPIC USAGE: ", response.usage)
                    raw_output = response.content[0].text.strip()
                    self.cache[prompt] = raw_output
                    print("📤 Claude API call made.")
                    reply = raw_output

                print(f"\n[DEBUG] Claude raw chunk:\n{raw_output[:500]}...\n")  # trimmed preview

                try:
                    json_str = self._extract_json(raw_output)
                    result = json.loads(json_str)

                    score_block = result.get("labels", {})
                    if not isinstance(score_block, dict):
                        print(f"⚠️ No valid 'labels' returned for concept '{label}' in model {self.model_name}")
                        print(f"Raw GPT output (trimmed):\n{reply[:500]}...\n")
                        raise ValueError("Expected 'labels' field to be a dictionary.")

                    if include_explanations:
                        explanation = self._extract_explanation(result, raw_output)
                    else:
                        explanation = ""

                    normalized_block = {}
                    for k, v in score_block.items():
                        norm_key = self._normalize_label(k)
                        for defined_label in concept_label:
                            if self._normalize_label(defined_label) == norm_key:
                                normalized_block[defined_label] = v
                                break
                        else:
                            print(f"⚠️ Unexpected label from Claude: '{k}' → normalized as '{norm_key}'")

                    parsed_scores = {
                        label: float(normalized_block.get(label, 0.0)) for label in concept_label
                    }

                    all_scores.update(parsed_scores)

                    if include_explanations:
                        explanations.append(explanation)
                        self._warn_on_low_scores(parsed_scores, explanation, normalized_labels)

                except Exception as pe:
                    print(f"⚠️ Claude parsing failed for concept '{label}': {pe}")
                    explanations.append(f"Parsing failed for concept '{label}'")
                    continue

            binned_scores = {
                label: 1 if all_scores.get(label, 0.0) >= 0.5 else 0 for label in labels
            }

            return {
                "labels": all_scores,
                "binned_labels": binned_scores,
                "explanation": " | ".join(explanations) if include_explanations else ""
            }

        except Exception as e:
            print(f"[ERROR] Claude classify() failed: {e}")
            return {
                "labels": {label: 0.0 for label in labels},
                "binned_labels": {label: 0 for label in labels},
                "explanation": f"Error during classification: {str(e)}"
            }


