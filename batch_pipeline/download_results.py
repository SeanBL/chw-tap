import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def _write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def download_results(provider: str, job_id: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    if provider == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        print("⬇️ Downloading OpenAI batch results...")

        batch_url = f"https://api.openai.com/v1/batches/{job_id}"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

        meta = requests.get(batch_url, headers=headers, timeout=30)
        if meta.status_code != 200:
            raise RuntimeError(f"❌ Failed to fetch OpenAI batch metadata: {meta.text}")

        info = meta.json()
        status = info.get("status")
        if status != "completed":
            raise RuntimeError(f"❌ OpenAI batch not completed (status={status}).")

        output_file_id = info.get("output_file_id")
        error_file_id  = info.get("error_file_id")

        if output_file_id:
            content_url = f"https://api.openai.com/v1/files/{output_file_id}/content"
            # stream large files line-by-line
            r = requests.get(content_url, headers=headers, stream=True, timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"❌ Failed to download OpenAI results: {r.status_code} {r.text}")

            out_path = os.path.join(output_dir, f"openai_{job_id}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for line in r.iter_lines(decode_unicode=True):
                    if line:
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="replace")
                        f.write(line + "\n")
            print(f"✅ OpenAI results saved to: {out_path}")

            if error_file_id:
                err_url = f"https://api.openai.com/v1/files/{error_file_id}/content"
                er = requests.get(err_url, headers=headers, stream=True, timeout=60)
                if er.status_code == 200:
                    err_path = os.path.join(output_dir, f"openai_{job_id}_errors.jsonl")
                    with open(err_path, "w", encoding="utf-8") as f:
                        wrote_any = False
                        for line in er.iter_lines(decode_unicode=True):
                            if not line:
                                continue
                            if isinstance(line, bytes):
                                line = line.decode("utf-8", errors="replace")
                            f.write(line + "\n")
                            wrote_any = True
                    if wrote_any:
                        print(f"ℹ️ OpenAI error file saved to: {err_path}")

            return out_path

        if error_file_id:
            err_url = f"https://api.openai.com/v1/files/{error_file_id}/content"
            er = requests.get(err_url, headers=headers, stream=True, timeout=60)
            if er.status_code != 200:
                raise RuntimeError(f"❌ Failed to download OpenAI error file: {er.status_code} {er.text}")

            err_path = os.path.join(output_dir, f"openai_{job_id}_errors.jsonl")
            with open(err_path, "w", encoding="utf-8") as f:
                wrote_any = False
                for line in er.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    f.write(line + "\n")
                    wrote_any = True
            preview_msgs = []
            if wrote_any:
                with open(err_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= 5:
                            break
                        try:
                            obj = json.loads(line)
                            msg = (obj.get("error") or {}).get("message") or str(obj)
                        except Exception:
                            msg = line[:400]
                        preview_msgs.append(f"{i+1:02d}: {msg}")
                print("🔎 OpenAI error preview:\n" + "\n".join(preview_msgs))
            raise RuntimeError(
                f"❌ OpenAI batch produced no outputs. Saved error details to: {err_path}"
            )

        raise RuntimeError("❌ No output_file_id or error_file_id found for OpenAI batch.")

    elif provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        print("⬇️ Downloading Anthropic batch results...")

        base = f"https://api.anthropic.com/v1/messages/batches/{job_id}"
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "accept": "application/json",
        }

        meta = requests.get(base, headers=headers, timeout=30)
        try:
            meta.raise_for_status()
        except Exception:
            raise RuntimeError(f"❌ Failed to fetch Anthropic metadata: {meta.status_code} {meta.text}")

        results_url = f"{base}/results"
        r = requests.get(results_url, headers=headers, stream=True, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"❌ Failed to download Anthropic results: {r.status_code} {r.text}")

        results_path = os.path.join(output_dir, f"anthropic_{job_id}_results.jsonl")
        with open(results_path, "w", encoding="utf-8") as f:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                f.write(line + "\n")
        print(f"✅ Anthropic results saved to: {results_path}")

        errors_url = f"{base}/errors"
        e = requests.get(errors_url, headers=headers, stream=True, timeout=60)
        if e.status_code == 200:
            errors_path = os.path.join(output_dir, f"anthropic_{job_id}_errors.jsonl")
            wrote_any = False
            with open(errors_path, "w", encoding="utf-8") as f:
                for line in e.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    f.write(line + "\n")
                    wrote_any = True
            if wrote_any:
                print(f"ℹ️ Anthropic errors (if any) saved to: {errors_path}")
            else:
                try:
                    os.remove(errors_path)
                except FileNotFoundError:
                    pass

        return results_path
