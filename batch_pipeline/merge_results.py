import json
import re
import os, glob, json
import pandas as pd

# Default fallback paths (used only if no path is passed in)
OPENAI_PATH = "data/batch/openai_batch_output.jsonl"
ANTHROPIC_PATH = "data/batch/anthropic_batch_output.json"

ID_RE = re.compile(r"^\s*(\d+)")  # grab leading integer id

def parse_model_response(content: str):
    """
    Try to parse the model text into a dict with keys:
      { "labels": {label: float, ...}, "explanation": str }
    """
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            labels = parsed.get("labels", {}) or {}
            # coerce scores to float if possible
            labels = {k: float(v) for k, v in labels.items() if _is_number(v)}
            explanation = parsed.get("explanation", "")
            return {"labels": labels, "explanation": explanation}
    except Exception:
        pass
    # Fallback if not valid JSON
    return {"labels": {}, "explanation": ""}

def _extract_openai_response_text(obj):
    """
    Pull the model text out of OpenAI batch line:
      line["response"]["body"] is the real payload.
    Prefer the convenient 'text' field. Fall back to Responses 'output' blocks,
    then to Chat 'choices'.
    """
    resp = obj.get("response", {}) or {}

    # If request failed, bail
    sc = resp.get("status_code")
    if sc and int(sc) != 200:
        return ""

    body = resp.get("body")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            body = {}
    body = body or {}

    # 0) Easiest/most robust: OpenAI often includes concatenated assistant text here
    if isinstance(body.get("text"), str) and body["text"].strip():
        return body["text"]

    # 1) Responses API shape: output[].content[] with type="output_text"
    out = body.get("output")
    if isinstance(out, list) and out:
        for item in out:
            # Some accounts have message-like items; some have tool-steps first.
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "output_text" and block.get("text"):
                        return block["text"]

    # 2) Convenience sometimes present
    if isinstance(body.get("output_text"), str):
        return body["output_text"]

    # 3) Chat fallback
    try:
        return body["choices"][0]["message"]["content"]
    except Exception:
        return ""

def _is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def _extract_testimonial_id(custom_id: str) -> str:
    m = ID_RE.match(custom_id or "")
    if not m:
        raise ValueError(f"Could not extract testimonial_id from custom_id: {custom_id!r}")
    return m.group(1)  # string id like "1"

def _safe_json_from_text_block(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        # Strip ```json fences if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
        return json.loads(text)

def load_openai_results(path: str) -> dict:
    """
    Returns: { testimonial_id: { 'labels': {concept: score, ...}, 'explanations': {concept: explanation, ...} } }
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            t_id = _extract_testimonial_id(obj.get("custom_id", ""))
            text = _extract_openai_response_text(obj)
            if not text:
                continue
            payload = _safe_json_from_text_block(text)
            labels = payload.get("labels", {}) or {}
            explanation = payload.get("explanation") or ""

            d = out.setdefault(t_id, {"labels": {}, "explanations": {}})
            
            for k, v in labels.items():
                try:
                    d["labels"][k] = float(v)
                except (TypeError, ValueError):
                    d["labels"][k] = None
            if explanation:
                for k in labels.keys():
                    d["explanations"].setdefault(k, explanation)
    return out

def _extract_anthropic_text(obj: dict) -> str:
    """
    Handle the /v1/messages/batches/{id}/results NDJSON shapes.
    Prefer result.message.content[*].text; fall back to other likely places.
    """
    r = obj.get("result") or obj.get("response") or {}

    # If it's an error row, there won't be message content to parse.
    if isinstance(r, dict) and r.get("type") == "error":
        return ""

    # result.message.content = [{type:"text", text:"..."}]
    msg = r.get("message")
    if isinstance(msg, dict):
        content = msg.get("content", [])
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
            txt = "\n".join(t for t in texts if t)
            if txt:
                return txt

    # Sometimes the content is at result.content directly
    content = r.get("content")
    if isinstance(content, list):
        texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
        txt = "\n".join(t for t in texts if t)
        if txt:
            return txt

    # Very old shapes might put content at top level (unlikely here)
    content = obj.get("content")
    if isinstance(content, list):
        texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
        txt = "\n".join(t for t in texts if t)
        if txt:
            return txt

    return ""


def load_anthropic_results(path: str) -> dict:
    """
    Returns: { testimonial_id: { 'labels': {concept: score, ...}, 'explanations': {concept: explanation, ...} } }
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            t_id = _extract_testimonial_id(obj.get("custom_id", ""))
            text = _extract_anthropic_text(obj)
            if not text:
                continue
            payload = _safe_json_from_text_block(text)
            labels = payload.get("labels", {}) or {}
            explanation = payload.get("explanation") or ""

            d = out.setdefault(t_id, {"labels": {}, "explanations": {}})
            for k, v in labels.items():
                try:
                    d["labels"][k] = float(v)
                except (TypeError, ValueError):
                    d["labels"][k] = None
            if explanation:
                for k in labels.keys():
                    d["explanations"].setdefault(k, explanation)
    return out

def merge_all_results(
        openai_results, 
        anthropic_results, 
        testimonial_map, 
        label_list, 
        include_explanations=True,
        model_display_map=None, 
):
    if model_display_map is None:
        model_display_map = {"openai": "openai", "anthropic": "anthropic"}

    # Stable order + normalized label keys for safer joins
    label_list = list(label_list)
    def norm_label(s: str) -> str:
        return s.lower().strip().replace("-", " ").replace("_", " ")

    norm_label_map = {lbl: norm_label(lbl) for lbl in label_list}
    inv_norm = {}  # norm -> canonical
    for k, v in norm_label_map.items():
        inv_norm.setdefault(v, k)  # first wins as canonical

    ratings = []
    combined_results = []
    explanations_log = []

    missing_openai_ids = []
    missing_anthropic_ids = []

    for base_id, entry in testimonial_map.items():
        text = entry.get("text", "")
        topic = entry.get("topic", "unknown")
        gender = entry.get("gender", "unknown")

        # Ensure nested structure for IRR with None (missing) by default
        row_bundle = {"id": base_id, "text": text, "labels": {lbl: {} for lbl in label_list}}

        def parse_payload(model_payload):
            """Return dict: canonical_label -> score, and explanations dict (same keys)"""
            if not model_payload:
                return None, None
            raw_scores = (model_payload.get("labels") or {})
            raw_expls  = (model_payload.get("explanations") or {})
            # normalize keys to canonical labels when possible
            scores = {}
            for k, v in raw_scores.items():
                nk = norm_label(k)
                canonical = inv_norm.get(nk, k)  # fall back to original if unseen label
                try:
                    scores[canonical] = float(v)
                except Exception:
                    scores[canonical] = None
            expls = {}
            for k, v in raw_expls.items():
                nk = norm_label(k)
                canonical = inv_norm.get(nk, k)
                expls[canonical] = v
            return scores, expls

        def emit_model_row(provider_key, model_payload):
            pretty = model_display_map.get(provider_key, provider_key)
            scores, expls = parse_payload(model_payload)

            if scores is None:
                # Mark truly missing as None (not 0.0) so IRR can detect gaps
                for lbl in label_list:
                    row_bundle["labels"][lbl][provider_key] = None
                # don't add an Excel row for this model if it has no payload
                return

            # Fill ratings structure using provider_key
            for lbl in label_list:
                row_bundle["labels"][lbl][provider_key] = scores.get(lbl)

            # Build Explanation cell (display only)
            if include_explanations and expls:
                explanation_joined = " ".join(
                    f"[{k}] {v}" for k, v in expls.items() if v
                ).strip()
            else:
                explanation_joined = ""

            # Excel row (use pretty display name)
            combined_results.append({
                "ID": base_id,
                "Model": pretty,
                "Topic": topic,
                "Gender": gender,
                "Testimonial": text,
                **{lbl: (scores.get(lbl, None)) for lbl in label_list},
                "Explanation": explanation_joined,
            })

            if include_explanations:
                explanations_log.append({
                    "testimonial": text,
                    "model": pretty,
                    "label_scores": json.dumps(
                        {lbl: (scores.get(lbl, None)) for lbl in label_list},
                        indent=2
                    ),
                    "explanation": explanation_joined,
                })

        # Emit both models
        o_payload = openai_results.get(base_id)
        a_payload = anthropic_results.get(base_id)

        if o_payload is None:
            missing_openai_ids.append(base_id)
        if a_payload is None:
            missing_anthropic_ids.append(base_id)

        emit_model_row("openai",    o_payload)
        emit_model_row("anthropic", a_payload)

        ratings.append(row_bundle)

    partial_missing_rows = []
    for r in ratings:
        for lbl, scores in r["labels"].items():
            if scores.get("openai") is None or scores.get("anthropic") is None:
                partial_missing_rows.append({
                    "id": r["id"],
                    "label": lbl,
                    "missing_openai": scores.get("openai") is None,
                    "missing_anthropic": scores.get("anthropic") is None
                })

    if partial_missing_rows:
        example = partial_missing_rows[:5]
        raise RuntimeError(
            f"Detected {len(partial_missing_rows)} (testimonial,label) pairs with missing provider data. "
            f"Examples: {example}"
        )

    # Optional: visibility on partial missing
    if missing_openai_ids:
        print(f"⚠️ {len(missing_openai_ids)} testimonials missing OpenAI results")
    if missing_anthropic_ids:
        print(f"⚠️ {len(missing_anthropic_ids)} testimonials missing Anthropic results")

    return ratings, combined_results, explanations_log
