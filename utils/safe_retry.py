import time
import random
import inspect

def safe_classify_with_retries(
    model,
    model_name: str,
    text: str,
    labels: list,
    normalized_labels: dict,
    concept_definitions: dict = None,
    include_explanations: bool = False,
    max_retries: int = 3,
    delay_overrides: dict = None,
) -> dict:
    """
    Handles retries and rate limits when classifying a testimonial with a model.
    """

    # Normalize model key for throttling
    key = model_name.lower()
    if key.startswith("gpt-"):
        key = "gpt"  # collapse all GPT variants unless overridden

    delay_map = {
        "gemini": 4,
        "claude": 3,
        # "gpt": 1,
        "o3": 4
    }

    if delay_overrides:
        delay_map.update(delay_overrides)
    
    all_scores = {}
    all_binned = {}
    explanations = []

    for label in labels:
        retries = 0
        while retries <= max_retries:
            try:
                start = time.time()
                classify_sig = inspect.signature(model.classify)

                kwargs = dict(
                    text=text,
                    labels=[label],
                    normalized_labels=normalized_labels,
                    concept_definitions=concept_definitions,
                )
                if "include_explanations" in classify_sig.parameters:
                    kwargs["include_explanations"] = include_explanations

                result = model.classify(**kwargs)

                elapsed = time.time() - start
                print(f"✅ {model_name.upper()} completed in {elapsed:.2f}s")

                # Rate-limit pause
                delay = delay_map.get(key, 0)
                if delay > 0:
                    time.sleep(delay)

                score = result["labels"].get(label, 0.0)
                binned = result["binned_labels"].get(label, 0)
                explanation = result.get("explanation", "")

                all_scores[label] = score
                all_binned[label] = binned
                if include_explanations:
                    explanations.append(f"{label}: {explanation}")

                break

            except Exception as e:
                retries += 1
                print(f"⚠️ Error from {model_name.upper()} ({label}, attempt {retries}): {e}")
                if retries > max_retries:
                    all_scores[label] = 0.0
                    all_binned[label] = 0
                    if include_explanations:
                        explanations.append(f"{label}: Retry limit exceeded or error.")
                    break
                wait = 2 ** retries + random.uniform(0, 1)
                print(f"⏳ Retrying after {wait:.1f}s...")
                time.sleep(wait)

    return {
        "labels": all_scores,
        "binned_labels": all_binned,
        "explanation": " | ".join(explanations) if include_explanations else ""
    }