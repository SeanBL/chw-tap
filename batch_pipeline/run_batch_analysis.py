import os
import json
import yaml
import pandas as pd
from datetime import datetime
from utils.config import load_config
from main import load_testimonials_from_jsonl
from batch_pipeline.merge_results import (
    load_openai_results,
    load_anthropic_results,
    merge_all_results
)
from pipeline.irr import compute_irr_scores
from pipeline.visualize import (
    visualize_irr_scores,
    print_irr_table,
    export_irr_to_excel
)
from pipeline.disagreement import (
    compute_model_disagreements,
    summarize_disagreements,
    flag_high_disagreement_testimonials,
    model_disagreement_percentages
)
from pipeline.aggregate import (
    aggregate_concept_frequencies,
    compute_consensus_labels
)

def export_full_excel_report(results, labels, model_names, output_dir="data/outputs", 
                             include_explanations=True): 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/concept_analysis_full_output_{timestamp}.xlsx"

    df = pd.DataFrame(results)
    if df.empty:
        print("⚠️ No rows to export. Skipping full Excel report.")
        return

    if "ID" not in df.columns:
        raise ValueError("Missing 'ID' column in results.")

    # If explanations are disabled, drop the column before writing
    if not include_explanations:
        df = df.drop(columns=["Explanation"], errors="ignore")

    concept_counts = {label: {} for label in labels}
    concept_ids = {label: {} for label in labels}
    for label in labels:
        for model in model_names:
            if label in df.columns:
                filtered = df[(df["Model"] == model) & (df[label] >= 0.5)]
            else:
                # if a label column is missing, count is 0
                filtered = pd.DataFrame(columns=df.columns)
            concept_counts[label][model] = len(filtered)
            concept_ids[label][model] = ", ".join(map(str, filtered["ID"])) if not filtered.empty else ""

    concept_count_df = pd.DataFrame.from_dict(concept_counts, orient="index").reset_index().rename(columns={"index": "Concept"})
    concept_ids_df = pd.DataFrame.from_dict(concept_ids, orient="index").reset_index().rename(columns={"index": "Concept"})

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Scores + Explanations", index=False, float_format="%.3f")
        concept_count_df.to_excel(writer, sheet_name="Concept Count Summary", index=False, float_format="%.3f")
        concept_ids_df.to_excel(writer, sheet_name="Testimonial IDs by Concept", index=False, float_format="%.3f")

    print(f"📁 Full concept analysis output saved to {output_path}")

def run_batch_analysis_both(openai_results_path: str,
                            anthropic_results_path: str,
                            output_dir="data/outputs"):
    config = load_config()
    labels = config["labels"]

    pm = config.get("provider_models", {})
    model_display_map = {"openai": pm.get("openai", "openai"),
                         "anthropic": pm.get("anthropic", "anthropic")}
    include_explanations = config.get("include_explanations", True)

    testimonial_path = "data/processed/testimonials.jsonl"
    os.makedirs(output_dir, exist_ok=True)

    testimonials = load_testimonials_from_jsonl(testimonial_path)
    testimonial_map = {str(t.get("id", i)): t for i, t in enumerate(testimonials)}

    # Load both sides explicitly
    openai_data = load_openai_results(openai_results_path)
    anthropic_data = load_anthropic_results(anthropic_results_path)

    # 👇 Sanity check BEFORE merge_all_results
    tid = "1"
    if tid in openai_data and tid in anthropic_data:
        print("OpenAI labels for 1:", sorted(openai_data[tid]["labels"].keys())[:5], "…", len(openai_data[tid]["labels"]))
        print("Anthropic labels for 1:", sorted(anthropic_data[tid]["labels"].keys())[:5], "…", len(anthropic_data[tid]["labels"]))
    else:
        # fallback: pick any id present in both
        common_ids = [t for t in openai_data.keys() if t in anthropic_data]
        if common_ids:
            tid = common_ids[0]
            print(f"OpenAI labels for {tid}:", sorted(openai_data[tid]["labels"].keys())[:5], "…", len(openai_data[tid]["labels"]))
            print(f"Anthropic labels for {tid}:", sorted(anthropic_data[tid]["labels"].keys())[:5], "…", len(anthropic_data[tid]["labels"]))
        else:
            print("⚠️ No overlapping testimonial IDs between OpenAI and Anthropic.")
            return

    ratings, results, explanations_log = merge_all_results(
        openai_data,
        anthropic_data,
        testimonial_map,
        labels,
        include_explanations=include_explanations,
        model_display_map=model_display_map,
    )

    # Verify both providers actually present
    present = set()
    for row in ratings:
        for _, mm in row["labels"].items():
            for provider_key, score in mm.items():
                if score is not None:
                    present.add(provider_key)

    required = {"openai", "anthropic"}
    missing = required - present
    if missing:
        raise RuntimeError(f"Missing scores for: {', '.join(sorted(missing))}. "
                           f"Check your parsed JSONL and merge_all_results alignment.")

    # Save merged ratings
    with open(f"{output_dir}/classification_ratings.json", "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2)

    # Excel (pretty names)
    pretty_model_names = [model_display_map[m] for m in ["openai", "anthropic"]]
    export_full_excel_report(
        results, labels, pretty_model_names, output_dir,
        include_explanations=include_explanations
    )

    # Frequencies/consensus (provider keys)
    provider_model_names = ["openai", "anthropic"]
    concept_frequencies = aggregate_concept_frequencies(ratings, provider_model_names)
    consensus_labels = compute_consensus_labels(ratings, method="vote", model_names=provider_model_names)
    concept_output_path = f"{output_dir}/concept_frequency_consensus.xlsx"
    with pd.ExcelWriter(concept_output_path, engine="openpyxl") as writer:
        pd.DataFrame(concept_frequencies).to_excel(writer, sheet_name="Concept Frequencies", index=False)
        pd.DataFrame(consensus_labels).to_excel(writer, sheet_name="Consensus Labels", index=False)

    # IRR uses the merged two-rater ratings
    irr_scores = compute_irr_scores(ratings)
    with open(f"{output_dir}/irr_scores.json", "w", encoding="utf-8") as f:
        json.dump(irr_scores, f, indent=2)
    visualize_irr_scores(irr_scores)
    print_irr_table(irr_scores)
    export_irr_to_excel(irr_scores)

    # Disagreements
    disagreement_records = compute_model_disagreements(ratings)
    disagreement_df = pd.DataFrame(disagreement_records)
    disagreement_summary = summarize_disagreements(disagreement_df)
    flagged_testimonials = flag_high_disagreement_testimonials(disagreement_df, provider_model_names)
    model_disagreement_summary = model_disagreement_percentages(disagreement_df)
    explanations_df = pd.DataFrame(explanations_log)

    disagreement_output_path = f"{output_dir}/model_disagreements.xlsx"
    with pd.ExcelWriter(disagreement_output_path, engine="openpyxl") as writer:
        disagreement_df.to_excel(writer, sheet_name="Disagreements", index=False, float_format="%.3f")
        disagreement_summary.to_excel(writer, sheet_name="Summary", index=False, float_format="%.3f")
        model_disagreement_summary.to_excel(writer, sheet_name="Model Summary", index=False, float_format="%.3f")
        if not flagged_testimonials.empty:
            flagged_testimonials.to_excel(writer, sheet_name="Flagged", index=False, float_format="%.3f")
        if include_explanations and not explanations_df.empty:
            explanations_df.to_excel(writer, sheet_name="Explanations", index=False, float_format="%.3f")

    print(f"📉 Disagreement log saved to {disagreement_output_path}")

if __name__ == "__main__":
    # Temporary manual test
    run_batch_analysis_both(
    openai_results_path="data/batch/cached_results/openai_batch_....jsonl",
    anthropic_results_path="data/batch/cached_results/anthropic_msgbatch_....jsonl",
)
