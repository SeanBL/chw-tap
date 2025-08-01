from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import re
import json

class BaseModel(ABC):
    @abstractmethod
    def classify(self, text: str, labels: List[str]) -> Dict[str, float]:
        """Return a dictionary of label → score"""
        pass

    def _extract_json(self, text: str) -> str:
        text = text.strip("` \n")
        text = re.sub(r'//.*', '', text)

        # Fix missing comma between labels and explanation
        text = re.sub(r'(\})\s*("explanation")', r'\1,\n\2', text)

        # Try matching a full object with "labels"
        match = re.search(r'(\{[\s\S]*?"labels"[\s\S]*?\})', text)
        if match:
            candidate = match.group(1)
            if candidate.count('{') > candidate.count('}'):
                candidate += '}'
            return candidate

        # Fallback: slice from first { to last }
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            candidate = text[start:end]
            if candidate.count('{') > candidate.count('}'):
                candidate += '}'
            return candidate

        raise ValueError("No valid JSON object found in Ollama output.")
    
    def _extract_explanation(self, output_dict: dict, raw_output: str) -> str:
        # Try structured access first
        explanation = output_dict.get("explanation", "")
        if explanation:
            return explanation.strip()

        # Fallback: Regex extraction from raw text
        match = re.search(r'"explanation"\s*:\s*"(.+?)"', raw_output, re.DOTALL)
        if match:
            return match.group(1).strip()

        return "[No explanation returned]"