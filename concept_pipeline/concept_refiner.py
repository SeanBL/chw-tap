import os
import json
from dotenv import load_dotenv
from models.model_loader import get_model_response 

load_dotenv()

# Input/output paths
GENERAL_PATH = "data/outputs/bertopic_top_concepts.json"
NUANCED_PATH = "data/outputs/nuanced_concepts.json"
OUTPUT_PATH = "data/outputs/refined_labels.json"

SYSTEM_PROMPT = """
You are an expert qualitative researcher helping refine concepts extracted from community health worker testimonials.

Your responsibilities:
1. Suggest clearer or more academic names for each concept label.
2. Identify and merge overlapping or redundant concepts.
3. Group related concepts into broader categories or themes.
4. Propose any important concepts that may be missing.
5. Optionally suggest a hierarchy using parent-child structure if patterns emerge.

Return only a valid JSON object with the following fields:
{
  "refined_labels": {
    "original_label": "refined academic label"
  },
  "merged_concepts": [
    {
      "merged_label": "New Merged Concept",
      "includes": ["original_label_1", "original_label_2"]
    }
  ],
  "grouped": {
    "Parent Theme 1": ["refined_label_1", "refined_label_2"],
    "Parent Theme 2": ["refined_label_3"]
  },
  "new_suggestions": ["Possible concept 1", "Possible concept 2"]
}
"""

def load_general_concepts(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return [item["Name"] for item in data if "Name" in item and item["Name"].lower() != "others"]

def load_nuanced_concepts(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            return list(data.keys())
        elif isinstance(data, list):
            return [item["concept"] for item in data if "concept" in item]
        else:
            return []

def call_llm_refiner(concept_list: list, model="gpt-4", provider="openai") -> str:
    user_prompt = f"""
The following is a combined list of concepts extracted from community health worker testimonials:

{json.dumps(concept_list, indent=2)}

Please refine, group, and improve them.
"""
    return get_model_response(
        prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        model=model,
        provider=provider
    )

def save_json(content: str, output_path: str):
    try:
        parsed = json.loads(content)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        print(f"✅ Refined labels saved to {output_path}")
    except json.JSONDecodeError:
        print("⚠️ Failed to parse LLM output as JSON. Raw output:")
        print(content)

def main():
    general_labels = load_general_concepts(GENERAL_PATH)
    nuanced_labels = load_nuanced_concepts(NUANCED_PATH)
    all_concepts = list(set(general_labels + nuanced_labels))

    result = call_llm_refiner(
        all_concepts,
        model="gpt-4",
        provider="openai"  # You can switch to "gemini", "claude", or "ollama"
    )
    save_json(result, OUTPUT_PATH)

if __name__ == "__main__":
    main()
