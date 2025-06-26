from abc import ABC, abstractmethod
from typing import List, Dict

class BaseModel(ABC):
    @abstractmethod
    def classify(self, text: str, labels: List[str]) -> Dict[str, float]:
        """Return a dictionary of label → score"""
        pass
