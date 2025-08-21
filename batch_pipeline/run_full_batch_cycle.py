import os
import json
import time
import requests
from datetime import datetime
import glob
from pathlib import Path
from utils.config import load_config
from utils.logging import log_to_file
from utils.retry import retry_with_backoff
from batch_pipeline.create_openai_batch_input import (
    create_batch_input_file as build_openai_input,
    load_testimonials_from_jsonl,
)
from batch_pipeline.create_anthropic_batch_input import (
    create_anthropic_batch_file as build_anthropic_input
)
from batch_pipeline.submit_batches import submit_openai_batch, submit_anthropic_batch
from batch_pipeline.poll_batches import poll_openai_batch, poll_anthropic_batch
from batch_pipeline.download_results import download_results
from batch_pipeline.run_batch_analysis import run_batch_analysis_both, run_final_merged_analysis
from batch_pipeline.merge_results import load_openai_results, load_anthropic_results




# Constants
LOG_PATH = "logs/batch_log.txt"
CACHE_DIR = "data/batch/cached_results"

def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _load_all_ids(testimonial_path="data/processed/testimonials.jsonl"):
    items = load_testimonials_from_jsonl(testimonial_path)
    return [str(t.get("id", i + 1)) for i, t in enumerate(items)]

def _norm_label(s: str) -> str:
    return str(s).lower().strip().replace("-", " ").replace("_", " ")

def _coverage_report(cache_dir, expected_ids, labels):
    # discover cached files
    o_paths = sorted(glob.glob(os.path.join(cache_dir, "openai_*.jsonl")))
    a_paths = sorted(glob.glob(os.path.join(cache_dir, "anthropic_*_results.jsonl")))

    # aggregate quickly (re-using your loaders)
    agg_o, agg_a = {}, {}
    def _accumulate(agg, part):
        for tid, payload in part.items():
            d = agg.setdefault(tid, {"labels": {}, "explanations": {}})
            for k, v in (payload.get("labels") or {}).items():
                if d["labels"].get(k) is None and v is not None:
                    d["labels"][k] = v

    for p in o_paths:
        _accumulate(agg_o, load_openai_results(p))
    for p in a_paths:
        _accumulate(agg_a, load_anthropic_results(p))

    # compute coverage
    exp_pairs = {(tid, lbl) for tid in expected_ids for lbl in labels}
    canon = { _norm_label(lbl): lbl for lbl in labels }

    def _pairs_from(agg):
        have = set()
        for tid, payload in agg.items():
            for k, v in (payload.get("labels") or {}).items():
                if v is None: 
                    continue
                lbl = canon.get(_norm_label(k))
                if lbl:
                    have.add((tid, lbl))
        return have

    have_o = _pairs_from(agg_o)
    have_a = _pairs_from(agg_a)
    total = len(exp_pairs)
    done_o = len(have_o)
    done_a = len(have_a)

    print(f"[Coverage] OpenAI: {done_o}/{total} ({done_o/total:.1%}) | Anthropic: {done_a}/{total} ({done_a/total:.1%})")
    return (done_o == total) and (done_a == total)

def run_chunked_job(batch_size=25):
    config = load_config()
    labels = config["labels"]
    pm = config.get("provider_models", {})  # adjust keys if your config differs
    openai_model    = pm.get("openai", "o3")
    anthropic_model = pm.get("anthropic", "claude-opus-4-20250514")
    testimonial_path = "data/processed/testimonials.jsonl"

    ids = _load_all_ids(testimonial_path)
    os.makedirs("data/batch/inputs", exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    for idx, id_chunk in enumerate(_chunk(ids, batch_size), start=1):
        ids_filter = set(id_chunk)

        # 1) Build per-chunk input files (both providers)
        oai_in = f"data/batch/inputs/openai_chunk_{idx:03}.jsonl"
        ant_in = f"data/batch/inputs/anthropic_chunk_{idx:03}.json"

        build_openai_input(
            testimonial_path=testimonial_path,
            output_path=oai_in,
            model_name=openai_model,
            ids_filter=ids_filter
        )
        build_anthropic_input(
            testimonial_path=testimonial_path,
            output_path=ant_in,
            model_name=anthropic_model,
            ids_filter=ids_filter
        )

        # 2) Submit both using the per-chunk paths
        log_to_file(LOG_PATH, f"📤 Submitting chunk {idx} (size={len(id_chunk)})")
        o_job = submit_openai_batch(input_path=oai_in)
        a_job = submit_anthropic_batch(input_path=ant_in)

        # 3) Poll and download both
        for provider, job in (("openai", o_job), ("anthropic", a_job)):
            if poll_until_complete(provider, job):
                path = download_results(provider, job, CACHE_DIR)
                log_to_file(LOG_PATH, f"✅ {provider} results saved to: {path}")
            else:
                log_to_file(LOG_PATH, f"⚠️ Skipping download for failed {provider} job {job}")

        # 4) Optional: show dynamic coverage after each chunk
        _coverage_report(CACHE_DIR, expected_ids=ids, labels=labels)

    # 5) After all chunks: only run final analysis if coverage is complete
    complete = _coverage_report(CACHE_DIR, expected_ids=ids, labels=labels)
    if not complete:
        log_to_file(LOG_PATH, "⏸️ Coverage incomplete; skipping final IRR/exports for now.")
        return
    log_to_file(LOG_PATH, "📊 Running final merged analysis across all cached chunks")
    run_final_merged_analysis(batch_cache_dir=CACHE_DIR, output_dir="data/outputs")

# Single batch job submission

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
    USE_CHUNKED = True   # set False to use single-shot flow
    if USE_CHUNKED:
        run_chunked_job(batch_size=25)
    else:
        run_batch_job()
