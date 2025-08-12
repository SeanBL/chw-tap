import os
import json
import pandas as pd
import yaml
from datetime import datetime
from utils.config import load_config
from models.model_loader import load_models_from_config
from pipeline.irr import compute_irr_scores
from pipeline.visualize import visualize_irr_scores, print_irr_table, export_irr_to_excel
from utils.safe_retry import safe_classify_with_retries
from batch_pipeline.create_openai_batch_input import create_batch_input_file
from batch_pipeline.create_anthropic_batch_input import create_anthropic_batch_file
from pipeline.disagreement import (
    compute_model_disagreements,
    summarize_disagreements,
    flag_high_disagreement_testimonials,
    model_disagreement_percentages,
)
from pipeline.aggregate import (
    aggregate_concept_frequencies,
    compute_consensus_labels,
)


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

def lookup_label_score(label: str, label_scores: dict) -> float:
    if label in label_scores:
        return label_scores[label]
    norm = label.lower().strip().replace("-", " ").replace("_", " ")
    for k in label_scores:
        k_norm = k.lower().strip().replace("-", " ").replace("_", " ")
        if norm == k_norm:
            return label_scores[k]
    return 0.0

def run_conceptual_analysis():
    config = load_config()
    include_explanations = config.get("include_explanations", False)

    # Load concept definitions and batch size
    with open("config/concept_definitions.yaml", "r", encoding="utf-8") as f:
        concept_definitions = yaml.safe_load(f)

    testimonial_path = "data/processed/testimonials.jsonl"
    models = load_models_from_config()
    model_names = list(models.keys())

    os.makedirs("data/outputs", exist_ok=True)

    testimonials = load_testimonials_from_jsonl(testimonial_path)

    labels = config["labels"]
    normalized_labels = {label.strip().lower().replace("-", " ").replace("_", " "): label for label in labels}

    ratings = []
    explanations_log = []
    results = []

    for entry in testimonials:
        text = entry["text"]
        topic = entry.get("topic", "unknown")
        gender = entry.get("gender", "unknown")
        testimonial_id = entry.get("id", "unknown")
        print(f"\n📝 Testimonial {testimonial_id}:\n{text}")
        testimonial_ratings = {"id": testimonial_id, "text": text, "labels": {}}

        for model_name, model in models.items():
            result = safe_classify_with_retries(
                model=model,
                model_name=model_name,
                text=text,
                labels=labels,
                normalized_labels=normalized_labels,
                concept_definitions=concept_definitions,
                include_explanations=include_explanations
            )

            if not result or "labels" not in result:
                print(f"⚠️ Skipping model {model_name} due to invalid result.")
                continue

            label_scores = result["labels"]
            explanation = result["explanation"]

            # for label in labels:
            #     testimonial_ratings["labels"].setdefault(label, {})[model_name] = label_scores.get(label, 0.0)
            for label, score in label_scores.items():
                testimonial_ratings["labels"].setdefault(label, {})[model_name] = score
                print(f"- {label}: {score:.3f}")


            explanations_log.append({
                "testimonial": text,
                "model": model_name,
                "label_scores": json.dumps(label_scores, indent=2),
                "explanation": explanation
            })

            print(f"\n🤖 {model_name.upper()} Label Scores:")
            print("📌 Available keys from model result:", list(label_scores.keys()))
            for label in labels:
                print(f"- {label}: {score:.3f}")

            print("🧠 Explanation:", explanation)

            results.append({
                "ID": testimonial_id,
                "Model": model_name,
                "Topic": topic,
                "Gender": gender,
                "Testimonial": text,
                **{label: score for label, score in label_scores.items()},
                "Explanation": explanation
            })

        ratings.append(testimonial_ratings)

    with open("data/outputs/classification_ratings.json", "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2)
    
    export_full_excel_report(results, labels, model_names)

    # Concept analysis
    concept_frequencies = aggregate_concept_frequencies(ratings, model_names=model_names)
    consensus_labels = compute_consensus_labels(ratings, method="vote", model_names=model_names)
    concept_output_path = "data/outputs/concept_frequency_consensus.xlsx"
    with pd.ExcelWriter(concept_output_path, engine="openpyxl") as writer:
        pd.DataFrame(concept_frequencies).to_excel(writer, sheet_name="Concept Frequencies", index=False)
        pd.DataFrame(consensus_labels).to_excel(writer, sheet_name="Consensus Labels", index=False)
    print(f"📊 Concept frequency and consensus saved to {concept_output_path}")

    # IRR
    irr_scores = compute_irr_scores(ratings)
    irr_path = "data/outputs/irr_scores.json"
    with open(irr_path, "w", encoding="utf-8") as f:
        json.dump(irr_scores, f, indent=2)
    print(f"📊 IRR scores saved to {irr_path}")

    visualize_irr_scores(irr_scores)
    print_irr_table(irr_scores)
    export_irr_to_excel(irr_scores)

    # Disagreements
    disagreement_records = compute_model_disagreements(ratings)
    disagreement_df = pd.DataFrame(disagreement_records)
    disagreement_summary = summarize_disagreements(disagreement_df)
    flagged_testimonials = flag_high_disagreement_testimonials(disagreement_df, model_names)
    model_disagreement_summary = model_disagreement_percentages(disagreement_df)
    explanations_df = pd.DataFrame(explanations_log)

    disagreement_output_path = "data/outputs/model_disagreements.xlsx"
    with pd.ExcelWriter(disagreement_output_path, engine="openpyxl") as writer:
        disagreement_df.to_excel(writer, sheet_name="Disagreements", index=False, float_format="%.3f")
        disagreement_summary.to_excel(writer, sheet_name="Summary", index=False, float_format="%.3f")
        model_disagreement_summary.to_excel(writer, sheet_name="Model Summary", index=False, float_format="%.3f")
        if not flagged_testimonials.empty:
            flagged_testimonials.to_excel(writer, sheet_name="Flagged", index=False, float_format="%.3f")
        explanations_df.to_excel(writer, sheet_name="Explanations", index=False, float_format="%.3f")
    print(f"📉 Disagreement log saved to {disagreement_output_path}")

def export_full_excel_report(results, labels, model_names, output_dir="data/outputs"):
    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/concept_analysis_full_output_{timestamp}.xlsx"

    df = pd.DataFrame(results)

    # Ensure 'ID' is included
    if "ID" not in df.columns:
        raise ValueError("Missing 'ID' column in results.")

    # Sheet 2: Count of concept scores >= 0.5 per model
    concept_counts = {label: {} for label in labels}
    concept_ids = {label: {} for label in labels}
    for label in labels:
        for model in model_names:
            filtered = df[(df["Model"] == model) & (df[label] >= 0.5)]
            concept_counts[label][model] = len(filtered)
            concept_ids[label][model] = ", ".join(map(str, filtered["ID"])) if not filtered.empty else ""

    concept_count_df = pd.DataFrame.from_dict(concept_counts, orient="index").reset_index().rename(columns={"index": "Concept"})
    concept_ids_df = pd.DataFrame.from_dict(concept_ids, orient="index").reset_index().rename(columns={"index": "Concept"})

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Scores + Explanations", index=False, float_format="%.3f")
        concept_count_df.to_excel(writer, sheet_name="Concept Count Summary", index=False, float_format="%.3f")
        concept_ids_df.to_excel(writer, sheet_name="Testimonial IDs by Concept", index=False, float_format="%.3f")

    print(f"📁 Full concept analysis output saved to {output_path}")

if __name__ == "__main__":
    run_conceptual_analysis()
