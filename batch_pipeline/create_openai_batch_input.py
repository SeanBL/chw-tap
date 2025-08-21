import json
import yaml
from pathlib import Path
from typing import Optional, Set
from utils.prompt_template import generate_prompt


def load_testimonials_from_jsonl(path: str, limit: Optional[int] = None) -> list:
    """
    Expects each line to be a JSON object. If it has a "content" list,
    we join it into a "text" field for convenience.
    """
    testimonials = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            if "content" in obj and isinstance(obj["content"], list):
                obj["text"] = " ".join(obj["content"])
            if "text" in obj and obj["text"]:
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
    temperature: Optional[float] = None,   # ignored for o3 responses API
    limit: Optional[int] = None,
    ids_filter: Optional[Set[str]] = None, # <-- NEW: include only these testimonial IDs (as strings)
):
    labels, concept_definitions = load_labels_and_concepts()
    testimonials = load_testimonials_from_jsonl(testimonial_path, limit)

    # Normalize filter to strings
    if ids_filter is not None:
        ids_filter = {str(x) for x in ids_filter}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as outfile:
        for i, entry in enumerate(testimonials):
            tid = str(entry.get("id", i + 1))
            if ids_filter and tid not in ids_filter:
                continue
            text = entry["text"]

            # ONE request per (testimonial × label)
            for label in labels:
                prompt = generate_prompt(
                    text=text,
                    labels=[label],  # single-label prompt
                    concept_definitions=concept_definitions,
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
                        # temperature is ignored by o3 Responses API; don't include
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
                    "custom_id": f"{tid}::{label}",  # encode label in ID; loaders extract leading integer
                    "method": "POST",
                    "url": url,
                    "body": body,
                }
                outfile.write(json.dumps(jsonl_line, ensure_ascii=False) + "\n")
                written += 1

    print(f"✅ OpenAI batch input written: {output_path}  "
          f"(requests: {written}, testimonials included: "
          f"{len(ids_filter) if ids_filter else 'ALL'})")


if __name__ == "__main__":
    create_batch_input_file(
        testimonial_path="data/processed/testimonials.jsonl",
        output_path="data/batch/inputs/openai_batch_input.jsonl",
        model_name="o3",
        limit=None,
        ids_filter=None,  # set to a set of string IDs for chunked runs, e.g. {'1','2',...}
    )
