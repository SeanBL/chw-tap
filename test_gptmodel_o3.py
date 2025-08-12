import os, json
from models.gpt_model import GPTModel  # adjust import if your path differs

def main():
    # Minimal sample
    text = "I taught my community to use safe water and handwashing to prevent cholera."
    labels = ["Cholera Education and Prevention"]
    concept_defs = {
        "Cholera Education and Prevention": {
            "definition": "Teaching about cholera symptoms and prevention (water, sanitation, hygiene).",
            "inclusion": "CHW-led education on cholera prevention.",
            "exclusion": "General health info not related to cholera."
        }
    }

    model = GPTModel(model="o3")  # uses OPENAI_API_KEY + OPENAI_ORG_ID from env
    result = model.classify(
        text=text,
        labels=labels,
        normalized_labels={l.lower(): l for l in labels},
        concept_definitions=concept_defs,
        include_explanations=True
    )

    print("\n=== GPTModel o3 Integration Test ===")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
