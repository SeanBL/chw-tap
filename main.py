import csv
import os
import json
from utils.config import load_config
from models.model_loader import load_models_from_config
from pipeline.irr import compute_irr_scores
from pipeline.visualize import visualize_irr_scores, print_irr_table
from pipeline.visualize import export_irr_to_excel
from pipeline.disagreement import (
    compute_model_disagreements,
    summarize_disagreements,
    flag_high_disagreement_testimonials,
    export_disagreements_to_excel,
    model_disagreement_percentages,
)
from pipeline.aggregate import (
    aggregate_concept_frequencies,
    compute_consensus_labels,
    export_consensus_to_excel
)
import pandas as pd


# Load config
config = load_config()
selected_models = config["models"]
output_path = config.get("output_csv", "conceptual_analysis_output.csv")

# Load model instances from config
models = load_models_from_config()

# Testimonials to classify
testimonials = [
    "After receiving training from WiRED, I was able to teach others in my village about malaria prevention. The community now trusts me and people ask me for advice all the time.",
    "I visit homes and share information about clean water. People now recognize me as a health worker.",
    "I had no previous experience, but after training I felt confident talking to people about disease prevention."
]

# Determine label set
if config.get("use_generated_labels"):
    from pipeline.topic_modeling import generate_labels_from_topic_model
    labels = generate_labels_from_topic_model(testimonials, method=config.get("label_source", "bertopic"))
else:
    labels = config["labels"]

# Create normalized label map for use during classification and warnings
normalized_labels = {label.strip().lower().replace("-", " ").replace("_", " "): label for label in labels}

# Prepare output directory
os.makedirs("data/outputs", exist_ok=True)

# Initialize ratings for IRR
ratings = []

# Collect model explanations
explanations_log = []

# Run analysis
with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Model", "Testimonial", *labels, "Explanation"])

    for i, text in enumerate(testimonials):
        print(f"\n📝 Testimonial {i + 1}:\n{text}")
        testimonial_ratings = {
            "text": text,
            "labels": {}
        }

        for model_name, model in models.items():
            result = model.classify(text, labels, normalized_labels)

            if not result or "labels" not in result:
                print(f"⚠️ Skipping model {model_name} due to invalid result.")
                continue

            label_scores = result["labels"]
            explanation = result["explanation"]

            # Collect scores for IRR
            for label in labels:
                testimonial_ratings["labels"].setdefault(label, {})[model_name] = label_scores.get(label, 0.0)

            # Save explanation log
            explanations_log.append({
                "testimonial": text,
                "model": model_name,
                "label_scores": json.dumps(label_scores, indent=2),
                "explanation": explanation
            })

            # Console output
            print(f"\n🤖 {model_name.upper()} Label Scores:")
            for label in labels:
                print(f"- {label}: {label_scores.get(label, 0.0):.2f}")
            print("🧠 Explanation:", explanation)

            # CSV row
            row = [model_name, text] + [label_scores.get(label, 0.0) for label in labels] + [explanation]
            writer.writerow(row)

        print(f"✅ Collected ratings for testimonial {i + 1}: {len(testimonial_ratings['labels'])} labels")

        ratings.append(testimonial_ratings)

        # Concept Frequency Aggregation
        concept_frequencies = aggregate_concept_frequencies(ratings, model_names=list(models.keys()))

        # Consensus Labeling
        consensus_labels = compute_consensus_labels(ratings, method="vote", model_names=list(models.keys()))


print(f"\n✅ Results saved to {output_path}")

# Compute IRR scores
irr_scores = compute_irr_scores(ratings)

# Save to JSON
irr_path = "data/outputs/irr_scores.json"
with open(irr_path, "w", encoding="utf-8") as f:
    json.dump(irr_scores, f, indent=2)

print(f"📊 IRR scores saved to {irr_path}")

# Generate IRR visualizations and print table
visualize_irr_scores(irr_scores)
print_irr_table(irr_scores)

export_irr_to_excel(irr_scores)

# Analyze model disagreements
disagreement_records = compute_model_disagreements(ratings)
disagreement_df = pd.DataFrame(disagreement_records)
disagreement_summary = summarize_disagreements(disagreement_df)
flagged_testimonials = flag_high_disagreement_testimonials(disagreement_df, list(models.keys()))
model_disagreement_summary = model_disagreement_percentages(disagreement_df)

# Convert explanations log to DataFrame
explanations_df = pd.DataFrame(explanations_log)

# Export Concept Frequency and Consensus to Excel
concept_output_path = "data/outputs/concept_frequency_consensus.xlsx"
with pd.ExcelWriter(concept_output_path, engine="openpyxl") as writer:
    pd.DataFrame(concept_frequencies).to_excel(writer, sheet_name="Concept Frequencies", index=False)
    pd.DataFrame(consensus_labels).to_excel(writer, sheet_name="Consensus Labels", index=False)

print(f"📊 Concept frequency and consensus saved to {concept_output_path}")

# Save disagreement logs to Excel with summary and flags
disagreement_output_path = "data/outputs/model_disagreements.xlsx"
with pd.ExcelWriter(disagreement_output_path, engine="openpyxl") as writer:
    disagreement_df.to_excel(writer, sheet_name="Disagreements", index=False)
    disagreement_summary.to_excel(writer, sheet_name="Summary", index=False)
    model_disagreement_summary.to_excel(writer, sheet_name="Model Summary", index=False)
    if not flagged_testimonials.empty:
        flagged_testimonials.to_excel(writer, sheet_name="Flagged", index=False)
    explanations_df.to_excel(writer, sheet_name="Explanations", index=False)

print(f"📉 Disagreement log saved to {disagreement_output_path}")
