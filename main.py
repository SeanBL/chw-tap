import os
import json
import pandas as pd
from utils.config import load_config
from models.model_loader import load_models_from_config
from pipeline.irr import compute_irr_scores
from pipeline.visualize import visualize_irr_scores, print_irr_table, export_irr_to_excel
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
                testimonials.append(" ".join(obj["content"]))
    return testimonials


def run_conceptual_analysis():
    config = load_config()
    output_path = config.get("output_csv", "data/outputs/conceptual_analysis_output.csv")
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

    for i, text in enumerate(testimonials):
        print(f"\n📝 Testimonial {i + 1}:\n{text}")
        testimonial_ratings = {"text": text, "labels": {}}

        for model_name, model in models.items():
            result = model.classify(text, labels, normalized_labels)

            if not result or "labels" not in result:
                print(f"⚠️ Skipping model {model_name} due to invalid result.")
                continue

            label_scores = result["labels"]
            explanation = result["explanation"]

            for label in labels:
                testimonial_ratings["labels"].setdefault(label, {})[model_name] = label_scores.get(label, 0.0)

            explanations_log.append({
                "testimonial": text,
                "model": model_name,
                "label_scores": json.dumps(label_scores, indent=2),
                "explanation": explanation
            })

            print(f"\n🤖 {model_name.upper()} Label Scores:")
            for label in labels:
                print(f"- {label}: {label_scores.get(label, 0.0):.2f}")
            print("🧠 Explanation:", explanation)

            results.append({
                "Model": model_name,
                "Testimonial": text,
                **{label: label_scores.get(label, 0.0) for label in labels},
                "Explanation": explanation
            })

        ratings.append(testimonial_ratings)

    # Save core outputs
    df = pd.DataFrame(results)
    csv_path = output_path
    xlsx_path = output_path.replace(".csv", ".xlsx")
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    print(f"\n✅ CSV output saved to {csv_path}")
    print(f"✅ Excel output saved to {xlsx_path}")

    with open("data/outputs/classification_ratings.json", "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2)

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
        disagreement_df.to_excel(writer, sheet_name="Disagreements", index=False)
        disagreement_summary.to_excel(writer, sheet_name="Summary", index=False)
        model_disagreement_summary.to_excel(writer, sheet_name="Model Summary", index=False)
        if not flagged_testimonials.empty:
            flagged_testimonials.to_excel(writer, sheet_name="Flagged", index=False)
        explanations_df.to_excel(writer, sheet_name="Explanations", index=False)
    print(f"📉 Disagreement log saved to {disagreement_output_path}")


if __name__ == "__main__":
    run_conceptual_analysis()
