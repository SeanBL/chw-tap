import re
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from typing import Dict, List

stemmer = PorterStemmer()


class ModelSafetyMixin:
    def _normalize_label(self, label: str) -> str:
        """Standardize label for comparison."""
        label = re.sub(r'([a-z])([A-Z])', r'\1 \2', label)
        label = label.replace("_", " ").replace("-", " ")
        return label.strip().lower()

    def _stemmed_words(self, text: str) -> set:
        tokens = word_tokenize(text.lower())
        return {stemmer.stem(token) for token in tokens if token.isalpha()}

    def _explanation_contains_label_stem(self, label: str, explanation: str) -> bool:
        label_stems = {stemmer.stem(word) for word in self._normalize_label(label).split()}
        expl_stems = self._stemmed_words(explanation)
        return label_stems.issubset(expl_stems)

    def _warn_on_low_scores(self, parsed_scores: Dict[str, float], explanation: str, normalized_labels: Dict[str, str]):
        for norm_label, canonical_label in normalized_labels.items():
            if self._explanation_contains_label_stem(norm_label, explanation) and parsed_scores.get(canonical_label, 0.0) < 0.1:
                print(f"⚠️ Warning: '{canonical_label}' mentioned in explanation (stem match) but has very low score ({parsed_scores[canonical_label]})")

    def _extract_json(self, text: str) -> str:
        """Extract the first JSON object from potentially noisy text output."""
        text = text.strip("` \n")
        text = re.sub(r'//.*', '', text)
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1 and end > start:
            return text[start:end]
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        raise ValueError("No valid JSON object found in model output.")
