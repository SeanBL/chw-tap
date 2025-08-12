import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

METADATA_PATH = "data/batch/batch_metadata.json"
POLL_INTERVAL = 30  # kept for the outer loop; don't sleep in these functions


def _normalize_openai_status(raw):
    if not raw:
        return "unknown"
    s = str(raw).lower()
    if s in {"validating", "in_progress", "finalizing", "queued"}:
        return "in_progress"
    if s in {"completed"}:
        return "completed"
    if s in {"failed", "cancelled", "canceled", "expired"}:
        return "failed"
    return s


def _normalize_anthropic_status(raw):
    if not raw:
        return "unknown"
    s = str(raw).lower()
    if s in {"queued", "running", "in_progress", "processing", "validating"}:
        return "in_progress"
    if s in {"succeeded", "completed", "complete", "finished", "ended"}:  # <-- add ended
        return "completed"
    if s in {"failed", "error", "canceled", "cancelled", "expired"}:
        return "failed"
    return s


def poll_openai_batch(batch_id: str) -> str:
    """
    Single-shot status check for an OpenAI batch.
    Returns: 'in_progress' | 'completed' | 'failed' | 'unknown'
    """
    url = f"https://api.openai.com/v1/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 404:
            # right after submit it can briefly 404; treat as still cooking
            print("🔁 OpenAI batch status: (transient 404) -> in_progress")
            return "in_progress"
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"⚠️ OpenAI poll error: {e}")
        return "unknown"

    status = _normalize_openai_status(data.get("status"))
    print(f"🔁 OpenAI batch status: {status}")
    return status


def poll_anthropic_batch(batch_id: str) -> str:
    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}"
    headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"}

    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 404:
            print("🔁 Anthropic batch status: (transient 404) -> in_progress")
            return "in_progress"
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"⚠️ Anthropic poll error: {e}")
        return "unknown"

    raw_status = data.get("processing_status") or data.get("status") or data.get("state")
    status = _normalize_anthropic_status(raw_status)

    if status not in {"in_progress", "completed", "failed"}:
        print(f"ℹ️ Unexpected Anthropic status='{raw_status}'. Full payload:\n{json.dumps(data, indent=2)[:2000]}")

    print(f"🔁 Anthropic batch status: {status}")
    return status


# if __name__ == "__main__":
#     metadata = load_metadata()

#     openai_id = metadata.get("openai_batch_id")
#     anthropic_id = metadata.get("anthropic_batch_id")

#     if not openai_id and not anthropic_id:
#         print("⚠️ No batch IDs found in metadata.")
#         exit(1)

#     print("⏳ Polling for batch completion...")

#     if openai_id:
#         result = poll_openai_batch(openai_id)
#         print(f"✅ OpenAI batch completed with status: {result}")

#     if anthropic_id:
#         result = poll_anthropic_batch(anthropic_id)
#         print(f"✅ Anthropic batch completed with status: {result}")
