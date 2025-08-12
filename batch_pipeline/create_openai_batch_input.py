import json
import yaml
from pathlib import Path
from utils.prompt_template import generate_prompt

def load_testimonials_from_jsonl(path: str, limit: int = None) -> list:
    testimonials = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            obj = json.loads(line)
            if "content" in obj:
                obj["text"] = " ".join(obj["content"])
                testimonials.append(obj)
    return testimonials

def load_labels_and_concepts():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open("config/concept_definitions.yaml", "r", encoding="utf-8") as f:
        concept_definitions = yaml.safe_load(f)

    labels = config["labels"]
    return labels, concept_definitions

def create_batch_input_file(
    testimonial_path: str,
    output_path: str,
    model_name: str = "o3",
    temperature: float = None,
    limit: int = None
):
    labels, concept_definitions = load_labels_and_concepts()
    testimonials = load_testimonials_from_jsonl(testimonial_path, limit)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for i, entry in enumerate(testimonials):
            testimonial_id = entry.get("id", i + 1)
            text = entry["text"]

            # ONE request per (testimonial × label)
            for label in labels:
                # Use your existing prompt template with a single-label list
                prompt = generate_prompt(
                    text=text,
                    labels=[label],  # <-- single label only
                    concept_definitions=concept_definitions
                )

                # Pick correct endpoint/body based on model
                if str(model_name).lower().startswith("o3"):
                    url = "/v1/responses"
                    body = {
                        "model": model_name,
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": prompt}
                                ]
                            }
                        ],
                        # o3 ignores temperature; don't send it
                    }
                else:
                    url = "/v1/chat/completions"
                    body = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if temperature is not None:
                        body["temperature"] = float(temperature)

                jsonl_line = {
                    "custom_id": f"{testimonial_id}::{label}",  # <-- encode label in ID
                    "method": "POST",
                    "url": url,
                    "body": body,
                }
                outfile.write(json.dumps(jsonl_line) + "\n")

    print(f"✅ Batch input file written to: {output_path}")

if __name__ == "__main__":
    create_batch_input_file(
        testimonial_path="data/processed/testimonials.jsonl",
        output_path="data/batch/inputs/openai_batch_input.jsonl",
        model_name="o3",
        limit=None  # or set a small number for testing
    )
