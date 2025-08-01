import requests
from models.base_model import BaseModel
from utils.model_safety_mixin import ModelSafetyMixin
from typing import List, Dict, Tuple
import re
import json
import time
from utils.prompt_template import generate_prompt
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from math import ceil
from ollama import Client

class OllamaModel(BaseModel, ModelSafetyMixin):  # ✅ Inherit from ModelSafetyMixin
    def __init__(self, model_name="mistral", temperature: float = 0.0):
        self.api_url = "http://localhost:11434/api/generate"
        self.model_name = model_name
        self.temperature = temperature

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

        for i, label in enumerate(labels):
            print(f"\n📌 Scoring concept {i + 1}/{len(labels)} → {label}")
            concept_label = [label]

            prompt = generate_prompt(text, concept_label, concept_definitions or {})

            if prompt in prompt_cache:
                print("✅ Cache hit — skipping Ollama API call.")
                raw_output = prompt_cache[prompt]
            else:
                response = requests.post(self.api_url, json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False
                })
                raw_output = response.json().get("response", "")

                # Fix common formatting issues
                raw_output = re.sub(r'(\})(\s*\{)', r'\1,\2', raw_output)  # add comma between objects
                raw_output = raw_output.strip("` \n")
                raw_output = re.sub(r'//.*', '', raw_output)  # strip JS-style comments

                last_brace = raw_output.rfind('}')
                if last_brace != -1:
                    raw_output = raw_output[:last_brace + 1]

                prompt_cache[prompt] = raw_output
                print("📤 Ollama API call made.")

                reply = raw_output

            print(f"\n[DEBUG] Raw Ollama output before parsing:\n{raw_output}\n")

            try:
                json_str = self._extract_json(raw_output)
                output_dict = json.loads(json_str)

                score_block = output_dict.get("labels")
                if not isinstance(score_block, dict):
                    print(f"⚠️ No valid 'labels' returned for concept '{label}' in model {self.model_name}")
                    print(f"Raw GPT output (trimmed):\n{reply[:500]}...\n")
                    raise ValueError("Expected 'labels' field to be a dictionary.")

                # Conditionally extract explanation
                if include_explanations:
                    explanation = self._extract_explanation(output_dict, raw_output)
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
                        print(f"⚠️ Unexpected label from Ollama: '{k}' → normalized as '{norm_key}'")

                for k in score_block:
                    print(f"Normalized '{k}' → '{self._normalize_label(k)}'")

                parsed_scores = {
                    label: float(normalized_block.get(label, 0.0))
                    for label in concept_label
                }

                all_scores.update(parsed_scores)

                if include_explanations:
                    explanations.append(explanation)
                    self._warn_on_low_scores(parsed_scores, explanation, normalized_labels)

            except Exception as e:
                print("⚠️ Failed to parse response from Ollama model:", raw_output)
                print("Error:", str(e))
                explanations.append(f"Parsing failed for concept '{label}'")

        binned_scores = {
            label: 1 if all_scores.get(label, 0.0) >= 0.5 else 0 for label in labels
        }

        return {
            "labels": all_scores,
            "binned_labels": binned_scores,
            "explanation": " | ".join(explanations) if include_explanations else ""
        }
