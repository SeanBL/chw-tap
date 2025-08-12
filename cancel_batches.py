import os
import time
import requests

# Get API key from environment or ask user
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    api_key = input("🔑 Enter your OPENAI_API_KEY: ").strip()
    if not api_key:
        raise SystemExit("❌ No API key provided. Exiting...")

headers = {"Authorization": f"Bearer {api_key}"}

def list_batches():
    r = requests.get("https://api.openai.com/v1/batches", headers=headers, timeout=20)
    if r.status_code != 200:
        raise SystemExit(f"❌ Failed to list batches: {r.status_code} {r.text}")
    return r.json().get("data", [])

while True:
    batches = list_batches()
    to_cancel = [
        b["id"] for b in batches
        if b["status"] in ("in_progress", "validating", "queued")
    ]
    if not to_cancel:
        print("✅ All queued/in-progress batches cleared.")
        break

    print(f"🔍 Found {len(to_cancel)} batches to cancel: {to_cancel}")
    for bid in to_cancel:
        r = requests.post(f"https://api.openai.com/v1/batches/{bid}/cancel", headers=headers, timeout=20)
        print(f"⏹ Cancelled {bid}: {r.status_code} {r.text}")

    print("⏳ Waiting 5s before checking again...")
    time.sleep(5)
