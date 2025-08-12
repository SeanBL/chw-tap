import os
import json
import time
import requests
from datetime import datetime
from utils.config import load_config
from utils.logging import log_to_file
from utils.retry import retry_with_backoff
from batch_pipeline.submit_batches import submit_openai_batch, submit_anthropic_batch
from batch_pipeline.poll_batches import poll_openai_batch, poll_anthropic_batch
from batch_pipeline.download_results import download_results
from batch_pipeline.run_batch_analysis import run_batch_analysis_both

# Constants
LOG_PATH = "logs/batch_log.txt"
CACHE_DIR = "data/batch/cached_results"


def submit_batch_job(provider):
    log_to_file(LOG_PATH, f"Submitting batch job to {provider}...")
    # Add retry on submission failure
    @retry_with_backoff(retries=3)
    def _submit():
        if provider == "openai":
            return submit_openai_batch()
        elif provider == "anthropic":
            return submit_anthropic_batch()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    return _submit()

def debug_openai_batch_failure(batch_id):
    import requests, os
    url = f"https://api.openai.com/v1/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    r = requests.get(url, headers=headers)
    data = r.json()
    # log the high-level errors, if present
    from utils.logging import log_to_file
    LOG_PATH = "logs/batch_log.txt"
    log_to_file(LOG_PATH, f"🔍 [OpenAI] Batch detail: status={data.get('status')}, errors={data.get('errors')}")
    err_file = data.get("error_file_id")
    if err_file:
        # fetch the error file for per-line reasons
        fr = requests.get(f"https://api.openai.com/v1/files/{err_file}/content", headers=headers)
        log_to_file(LOG_PATH, f"🔍 [OpenAI] error_file_id={err_file} contents:\n{fr.text[:4000]}")  # first ~4KB

def poll_until_complete(provider, job_id, poll_interval=10, timeout=1800):
    log_to_file(LOG_PATH, f"Polling {provider} job {job_id} every {poll_interval}s (timeout={timeout}s)...")
    start_time = time.time()
    elapsed = 0

    while elapsed < timeout:
        @retry_with_backoff(retries=3)
        def _poll():
            if provider == "openai":
                return poll_openai_batch(job_id)
            elif provider == "anthropic":
                return poll_anthropic_batch(job_id)
            else:
                raise ValueError(f"Unknown provider: {provider}")

        status = _poll()
        if status == "completed":
            log_to_file(LOG_PATH, f"✅ Job {job_id} completed!")
            return True
        elif status in ["failed", "cancelled"]:
            log_to_file(LOG_PATH, f"❌ Job {job_id} ended with status: {status}")
            if provider == "openai":
                debug_openai_batch_failure(job_id)
            return False

        time.sleep(poll_interval)
        elapsed = time.time() - start_time
        log_to_file(LOG_PATH, f"⏳ Still waiting... Elapsed: {int(elapsed)}s")

    log_to_file(LOG_PATH, f"⚠️ Timeout reached for job {job_id} after {timeout}s")
    return False


def run_batch_job():
    config = load_config()
    batch_providers = config.get("batch_providers", [])
    os.makedirs(CACHE_DIR, exist_ok=True)

    results_paths = {}  # {"openai": "...jsonl", "anthropic": "...jsonl"}

    for provider in batch_providers:
        log_to_file(LOG_PATH, f"📤 Submitting job for provider: {provider}")
        job_id = submit_batch_job(provider)
        log_to_file(LOG_PATH, f"📤 Submitted {provider} job. Job ID: {job_id}")

        if not poll_until_complete(provider, job_id):
            log_to_file(LOG_PATH, f"⚠️ Skipping analysis for failed {provider} job.")
            continue

        log_to_file(LOG_PATH, f"⬇️ Downloading results for {provider} job {job_id}")
        try:
            path = download_results(provider, job_id, CACHE_DIR)
            if not path:
                log_to_file(LOG_PATH, f"⚠️ No results path returned for {provider} job {job_id}.")
                continue
            results_paths[provider] = path
            log_to_file(LOG_PATH, f"✅ {provider} results saved to: {path}")
        except Exception as e:
            log_to_file(LOG_PATH, f"❌ Download failed for {provider} job {job_id}: {e}")

    # 👉 Only analyze once we have BOTH files
    if {"openai", "anthropic"}.issubset(results_paths.keys()):
        log_to_file(LOG_PATH, f"📊 Running merged IRR analysis with "
                              f"OpenAI={results_paths['openai']} and Anthropic={results_paths['anthropic']}")
        try:
            # Use the combined-analysis entry point (see prior message)
            run_batch_analysis_both(
                openai_results_path=results_paths["openai"],
                anthropic_results_path=results_paths["anthropic"],
                output_dir="data/outputs",
            )
        except Exception as e:
            log_to_file(LOG_PATH, f"❌ Merged analysis failed: {e}")
    else:
        missing = {"openai", "anthropic"} - results_paths.keys()
        log_to_file(LOG_PATH, f"⚠️ Skipping IRR: missing providers: {', '.join(sorted(missing))}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    run_batch_job()
