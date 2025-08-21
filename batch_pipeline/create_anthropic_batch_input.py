import json
import yaml
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Set
from utils.prompt_template import generate_prompt
from utils.config import load_config

_ID_SAFE_RE = re.compile(r'[^a-zA-Z0-9_-]+')

def _slug_for_anthropic(label: str) -> str:
    s = label.lower().strip().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_-]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s or "label"

def _sanitize_id(value: str) -> str:
    """
    Anthropic custom_id must match ^[a-zA-Z0-9_-]{1,64}$.
    Keep the base short so there's room for the label tail.
    """
    v = _ID_SAFE_RE.sub('_', str(value)).strip('_-')
    return v[:32] or "id"

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
    temperature: Optional[float] = None,   # optional; omit if None
    limit: int = None,
    ids_filter: Optional[Set[str]] = None, # <-- NEW: only include these IDs if provided
):
    """
    Writes a single Anthropic batch JSON with:
      {
        "requests": [
          {"custom_id": "<id>_<label_slug>", "params": {...}},
          ...
        ]
      }
    One request per (testimonial × label). Use `ids_filter` to emit chunked subsets.
    """
    # read toggle from config so prompts match your analysis run
    cfg = load_config()
    include_explanations = cfg.get("include_explanations", True)

    labels, concept_definitions = load_labels_and_concepts()
    # deterministic label order helps with reproducibility
    labels = list(labels)

    testimonials = load_testimonials_from_jsonl(testimonial_path, limit)

    if not batch_id:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        batch_id = f"anthropic-batch-{timestamp}"

    requests = []
    for i, entry in enumerate(testimonials):
        # Keep a numeric-leading base id so your loader's regex picks it up
        tid = str(entry.get("id", i + 1))
        if ids_filter and tid not in ids_filter:
            continue

        base_id = _sanitize_id(tid)  # still starts with the numeric tid in your data
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
                tail_len = max(1, 64 - len(base_id) - 1)
                safe_id = f"{base_id}_{slug[-tail_len:]}"

            # Ensure first char is alnum (spec requirement)
            if not re.match(r"[A-Za-z0-9]", safe_id):
                safe_id = f"x{safe_id}"
                safe_id = safe_id[:64]

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
        json.dump(batch_payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Anthropic batch input file saved to: {output_path}")
    print(f"📦 Batch ID (local tag): {batch_id}  |  Requests: {len(requests)}")
    return batch_id

if __name__ == "__main__":
    # Full file (no chunking) example:
    create_anthropic_batch_file(
        testimonial_path="data/processed/testimonials.jsonl",
        output_path="data/batch/inputs/anthropic_batch_input.json",
        model_name="claude-opus-4-20250514",
        limit=None,
        ids_filter=None,  # or set: ids_filter={"1","2",...} for a chunk
    )

