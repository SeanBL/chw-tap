import json
import yaml
import re
from pathlib import Path
from datetime import datetime
from utils.prompt_template import generate_prompt
from utils.config import load_config

_ID_SAFE_RE = re.compile(r'[^a-zA-Z0-9_-]+')

def _slug_for_anthropic(label: str) -> str:
    s = label.lower().strip().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_-]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s or "label"

def _sanitize_id(value: str) -> str:
    # Anthropic custom_id must match ^[a-zA-Z0-9_-]{1,64}$
    v = _ID_SAFE_RE.sub('_', str(value)).strip('_-')
    return v[:32] or "id"  # keep base short to leave room for label tail

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

def create_anthropic_batch_file(
    testimonial_path: str,
    output_path: str,
    model_name: str = "claude-opus-4-20250514",
    batch_id: str = None,
    max_tokens: int = 1024,
    temperature: float = None,   # optional; omit if None
    limit: int = None,
):
    # read toggle from config so prompts match your REST runs
    cfg = load_config()
    include_explanations = cfg.get("include_explanations", True)

    labels, concept_definitions = load_labels_and_concepts()
    testimonials = load_testimonials_from_jsonl(testimonial_path, limit)

    if not batch_id:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        batch_id = f"anthropic-batch-{timestamp}"

    requests = []
    for i, entry in enumerate(testimonials):
        base_id = _sanitize_id(entry.get("id", i + 1))
        text = entry["text"]

        # ONE request per (testimonial × label) with a single-label prompt
        for label in labels:
            prompt = generate_prompt(
                text=text,
                labels=[label],                 # <-- single label only
                concept_definitions=concept_definitions,
                include_explanations=include_explanations,
            )

            slug = _slug_for_anthropic(label)
            safe_id = f"{base_id}_{slug}"

            # Enforce 64-char max custom_id
            if len(safe_id) > 64:
                tail_len = 64 - len(base_id) - 1
                safe_id = f"{base_id}_{slug[-max(tail_len, 1):]}"

            # Ensure first char is alnum
            if not re.match(r"[A-Za-z0-9]", safe_id):
                safe_id = f"x{safe_id}"

            # Anthropic batch requires params field
            params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                params["temperature"] = float(temperature)

            requests.append({
                "custom_id": safe_id,
                "params": params
            })

    batch_payload = {"requests": requests}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch_payload, f, indent=2)

    print(f"✅ Anthropic batch input file saved to: {output_path}")
    print(f"📦 Batch ID: {batch_id}")
    return batch_id

if __name__ == "__main__":
    create_anthropic_batch_file(
        testimonial_path="data/processed/testimonials.jsonl",
        output_path="data/batch/inputs/anthropic_batch_input.json",
        model_name="claude-opus-4-20250514",
        limit=None
    )

