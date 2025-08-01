import os
import json
import time
from dotenv import load_dotenv
import google.generativeai as genai
from models.base_model import BaseModel
from typing import List, Dict
from math import ceil
from utils.prompt_template import generate_prompt
from utils.model_safety_mixin import ModelSafetyMixin

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
        prompt_cache = {}

        try:
            for i, label in enumerate(labels):
                for i, label in enumerate(labels):
                    print(f"\n📌 Scoring concept {i + 1}/{len(labels)} → {label}")
                    concept_label = [label]

                    prompt = generate_prompt(text, concept_label, concept_definitions or {})

                if prompt in prompt_cache:
                    print("✅ Cache hit — skipping Gemini API call.")
                    raw_text = prompt_cache[prompt]
                else:
                    response = self.model.generate_content(prompt)
                    raw_text = response.text.strip()
                    prompt_cache[prompt] = raw_text
                    print("📤 Gemini API call made.")

                    reply = raw_text

                print(f"\n[DEBUG] Raw Gemini output (trimmed):\n{raw_text[:500]}...\n")

                try:
                    json_str = self._extract_json(raw_text)
                    result = json.loads(json_str)

                    score_block = result.get("labels", {})
                    if not isinstance(score_block, dict):
                        print(f"⚠️ No valid 'labels' returned for concept '{label}' in model {self.model_name}")
                        print(f"Raw GPT output (trimmed):\n{reply[:500]}...\n")
                        raise ValueError("Expected 'labels' field to be a dictionary.")
                    
                    # Conditionally extract explanation
                    if include_explanations:
                        explanation = self._extract_explanation(result, raw_text)
                    else:
                        explanation = ""

                except json.JSONDecodeError as je:
                    print(f"❌ JSON decode error on concept '{label}': {je}")
                    explanations.append(f"Parsing failed: invalid JSON for concept '{label}'")
                    continue
                except Exception as pe:
                    print(f"❌ Parsing error for concept '{label}': {pe}")
                    explanations.append(f"Parsing failed for concept '{label}'")
                    continue

                normalized_block = {}
                for k, v in score_block.items():
                    norm_key = self._normalize_label(k)
                    for defined_label in concept_label:
                        if self._normalize_label(defined_label) == norm_key:
                            normalized_block[defined_label] = v
                            break
                    else:
                        print(f"⚠️ Unexpected label from Gemini: '{k}' → normalized as '{norm_key}'")

                for k in score_block:
                    print(f"Normalized '{k}' → '{self._normalize_label(k)}'")

                parsed_scores = {
                    label: float(normalized_block.get(label, 0.0))
                    for label in concept_label
                }

                all_scores.update(parsed_scores)
                explanations.append(explanation)

            binned_scores = {
                label: 1 if all_scores.get(label, 0.0) >= 0.5 else 0 for label in labels
            }

            if include_explanations:
                self._warn_on_low_scores(all_scores, " | ".join(explanations), normalized_labels)

            return {
                "labels": all_scores,
                "binned_labels": binned_scores,
                "explanation": " | ".join(explanations) if include_explanations else ""
            }

        except Exception as e:
            print(f"⚠️ Gemini classify() failed: {e}")
            return {
                "labels": {label: 0.0 for label in labels},
                "binned_labels": {label: 0 for label in labels},
                "explanation": f"Classification error: {str(e)}" if include_explanations else ""
            }


