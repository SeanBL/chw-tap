import requests
from models.base_model import BaseModel
from typing import List, Dict
import re
import json
from utils.prompt_template import generate_prompt
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re

class OllamaModel(BaseModel):
    def __init__(self, model_name="mistral", temperature: float = 0.0):
        self.api_url = "http://localhost:11434/api/generate"
        self.model_name = model_name
        self.temperature = temperature

    def classify(self, text: str, labels: List[str], normalized_labels: Dict[str, str]) -> Dict:
        prompt = generate_prompt(text, labels)

        try:
            response = requests.post(self.api_url, json={
                "model": self.model_name,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False
            })

            raw_output = response.json().get("response", "")

            print(f"\n[DEBUG] Raw Ollama output:\n{raw_output}\n")

            json_str = self._extract_json(raw_output)
            output_dict = json.loads(json_str)

            # Handle nested vs flat JSON and normalize label keys
            score_block = output_dict.get("labels", output_dict)  # fallback if not nested
            explanation = output_dict.get("explanation", raw_output.replace(json_str, "").strip())

            # Normalize keys in score_block and map them to config-defined labels
            normalized_block = {}
            for k, v in score_block.items():
                norm_key = self._normalize_label(k)
                for defined_label in labels:
                    if self._normalize_label(defined_label) == norm_key:
                        normalized_block[defined_label] = v
                        break
                else:
                    # Log unexpected keys if no match is found
                    print(f"⚠️ Unexpected label from model: '{k}' → normalized as '{norm_key}'")

            # DEBUG: Show how each label key was normalized
            for k in score_block.keys():
                print(f"Normalized '{k}' → '{self._normalize_label(k)}'")

            # Build parsed_scores using canonical label names
            parsed_scores = {}
            for norm_label, canonical_label in normalized_labels.items():
                parsed_scores[canonical_label] = float(normalized_block.get(norm_label, 0.0))
                
            binned_scores = {
                label: 1 if parsed_scores.get(label, 0.0) >= 0.5 else 0 for label in labels
            }

            # Warn about unexpected label keys
            for key in score_block:
                normalized = self._normalize_label(key)
                if normalized not in [self._normalize_label(label) for label in labels]:
                    print(f"⚠️ Unexpected label from model: '{key}' → normalized as '{normalized}'")

            # Stem explanation and labels to catch morphological variants
            stemmed_expl = self._stemmed_words(explanation)
            stemmer = PorterStemmer()

            for norm_label, canonical_label in normalized_labels.items():
                # Stem each word in the normalized label (e.g., "knowledge sharing" → ["knowledg", "share"])
                label_stems = {stemmer.stem(word) for word in norm_label.split()}

                # If all stemmed words are in the explanation, warn if score is low
                if label_stems.issubset(stemmed_expl) and parsed_scores.get(canonical_label, 0.0) < 0.1:
                    print(f"⚠️ Warning: '{canonical_label}' mentioned in explanation (stem match) but has very low score ({parsed_scores[canonical_label]})")

            return {
                "labels": parsed_scores,
                "binned_labels": binned_scores,
                "explanation": explanation
            }

        except Exception as e:
            print("⚠️ Failed to parse response from Ollama model:", raw_output)
            print("Error:", str(e))
            return {
                "labels": {label: 0.0 for label in labels},
                "binned_labels": {label: 0 for label in labels},
                "explanation": f"Parsing failed or API call failed: {str(e)}"
            }

    def _normalize_label(self, label: str) -> str:
        """Standardize label for matching (lowercase, camelCase → spaced, dashes/underscores → space)."""
        label = re.sub(r'([a-z])([A-Z])', r'\1 \2', label)  # split camelCase
        label = label.replace("_", " ").replace("-", " ")
        return label.strip().lower()
    
    def _stemmed_words(self, text: str) -> set:
        stemmer = PorterStemmer()
        tokens = word_tokenize(text.lower())
        return {stemmer.stem(token) for token in tokens if token.isalpha()}

    def _extract_json(self, text: str) -> str:
        """
        Extract the first JSON object from possibly noisy text output.
        Removes comments and extracts clean JSON.
        """
        text = text.strip("` \n")

        # Remove inline comments (e.g., // trust not mentioned)
        text = re.sub(r'//.*', '', text)

        # Try bracket slicing
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1 and end > start:
            return text[start:end]

        # Fallback regex
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)

        raise ValueError("No valid JSON object found in Ollama output.")
