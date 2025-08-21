import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from utils.logging import log_to_file

load_dotenv()

LOG_PATH = "logs/batch_log.txt"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# === Config Paths ===
OPENAI_INPUT_PATH = "data/batch/inputs/openai_batch_input.jsonl"
ANTHROPIC_INPUT_PATH = "data/batch/inputs/anthropic_batch_input.json"
METADATA_OUTPUT_PATH = "data/batch/batch_metadata.json"

os.makedirs("data/batch", exist_ok=True)

# === Submit to OpenAI ===
def _infer_openai_endpoint(jsonl_path: str) -> str:
    # Look at the first real line and use its "url"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            url = obj.get("url")
            if url in ("/v1/chat/completions", "/v1/responses"):
                return url
            break
    # Fallback to chat if not detectable
    return "/v1/chat/completions"

def _preview_first_lines(path, n=3):
    lines = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for _ in range(n):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())
    except Exception as e:
        lines = [f"<preview failed: {e}>"]
    return lines

def submit_openai_batch(input_path: str | None = None, completion_window: str = "24h") -> str:
    log_to_file(LOG_PATH, f"🔵 [OpenAI] Preparing to submit batch job...")
    jsonl_path = input_path or OPENAI_INPUT_PATH   # <— NEW
    if not os.path.exists(jsonl_path):
        msg = f"❌ [OpenAI] Input file not found: {jsonl_path}"
        log_to_file(LOG_PATH, msg)
        raise FileNotFoundError(msg)

    # Preview input
    preview = _preview_first_lines(jsonl_path, n=3)
    log_to_file(LOG_PATH, f"🔵 [OpenAI] Input path: {jsonl_path}")
    log_to_file(LOG_PATH, "🔵 [OpenAI] First 3 lines of input:")
    for i, line in enumerate(preview, 1):
        log_to_file(LOG_PATH, f"   {i:02d}: {line}")

    # Upload the JSONL file
    log_to_file(LOG_PATH, "🔵 [OpenAI] Uploading file to /v1/files ...")
    try:
        with open(jsonl_path, "rb") as f:
            upload_response = requests.post(
                "https://api.openai.com/v1/files",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files={"file": (Path(jsonl_path).name, f)},
                data={"purpose": "batch"}
            )
        upload_data = upload_response.json()
    except Exception as e:
        log_to_file(LOG_PATH, f"❌ [OpenAI] Upload request error: {repr(e)}")
        raise
    if upload_response.status_code != 200:
        log_to_file(LOG_PATH, f"❌ [OpenAI] Upload failed: {upload_data}")
        raise RuntimeError(f"Failed to upload OpenAI file: {upload_data}")

    file_id = upload_data["id"]
    log_to_file(LOG_PATH, f"✅ [OpenAI] File uploaded. File ID: {file_id}")

    # Detect endpoint from the JSONL (supports both chat + responses)
    endpoint = _infer_openai_endpoint(jsonl_path)   # <— use the actual path
    log_to_file(LOG_PATH, f"🔵 [OpenAI] Using endpoint: {endpoint}")

    payload = {"input_file_id": file_id, "endpoint": endpoint, "completion_window": completion_window}
    log_to_file(LOG_PATH, "🔵 [OpenAI] Submitting batch to /v1/batches ...")
    try:
        submit_response = requests.post(
            "https://api.openai.com/v1/batches",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        batch_data = submit_response.json()
    except Exception as e:
        log_to_file(LOG_PATH, f"❌ [OpenAI] Submit request error: {repr(e)}")
        raise
    if submit_response.status_code != 200:
        log_to_file(LOG_PATH, f"❌ [OpenAI] Submit failed: {batch_data}")
        raise RuntimeError(f"Failed to submit OpenAI batch: {batch_data}")

    batch_id = batch_data["id"]
    log_to_file(LOG_PATH, f"🚀 [OpenAI] Batch submitted. Batch ID: {batch_id}")
    return batch_id



# === Submit to Anthropic ===
def submit_anthropic_batch(input_path: str | None = None) -> str:
    log_to_file(LOG_PATH, "🟠 [Anthropic] Preparing to submit batch job...")
    json_path = input_path or ANTHROPIC_INPUT_PATH   # <— NEW
    if not os.path.exists(json_path):
        msg = f"❌ [Anthropic] Input file not found: {json_path}"
        log_to_file(LOG_PATH, msg)
        raise FileNotFoundError(msg)

    preview = _preview_first_lines(json_path, n=3)
    log_to_file(LOG_PATH, f"🟠 [Anthropic] Input path: {json_path}")
    log_to_file(LOG_PATH, "🟠 [Anthropic] First 3 lines of input:")
    for i, line in enumerate(preview, 1):
        log_to_file(LOG_PATH, f"   {i:02d}: {line}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        log_to_file(LOG_PATH, f"❌ [Anthropic] Failed reading JSON: {repr(e)}")
        raise

    log_to_file(LOG_PATH, "🟠 [Anthropic] Submitting to /v1/messages/batches ...")
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages/batches",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            data=json.dumps(payload)
        )
        data = response.json()
    except Exception as e:
        log_to_file(LOG_PATH, f"❌ [Anthropic] Submit request error: {repr(e)}")
        raise
    if response.status_code != 200:
        log_to_file(LOG_PATH, f"❌ [Anthropic] Submit failed: {data}")
        raise RuntimeError(f"Failed to submit Anthropic batch: {data}")

    batch_id = data["id"]
    log_to_file(LOG_PATH, f"🚀 [Anthropic] Batch submitted. Batch ID: {batch_id}")
    return batch_id


# === Save metadata to disk ===
def save_batch_metadata(openai_id=None, anthropic_id=None):
    metadata = {}
    if openai_id:
        metadata["openai_batch_id"] = openai_id
    if anthropic_id:
        metadata["anthropic_batch_id"] = anthropic_id

    with open(METADATA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log_to_file(LOG_PATH, f"🗂️ Batch metadata saved to: {METADATA_OUTPUT_PATH}")


# === Entry point ===
# if __name__ == "__main__":
#     openai_id = None
#     anthropic_id = None

#     if os.path.exists(OPENAI_INPUT_PATH):
#         openai_id = submit_openai_batch()

#     if os.path.exists(ANTHROPIC_INPUT_PATH):
#         anthropic_id = submit_anthropic_batch()

#     if not openai_id and not anthropic_id:
#         print("⚠️ No batch input files found.")
#     else:
#         save_batch_metadata(openai_id, anthropic_id)
