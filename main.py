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

# Prepare output directory
os.makedirs("data/outputs", exist_ok=True)

# Initialize ratings for IRR
ratings = []

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
            result = model.classify(text, labels)

            if not result or "labels" not in result:
                print(f"⚠️ Skipping model {model_name} due to invalid result.")
                continue

            label_scores = result["labels"]
            explanation = result["explanation"]

            # Collect scores for IRR
            for label in labels:
                testimonial_ratings["labels"].setdefault(label, {})[model_name] = label_scores.get(label, 0.0)

            # Console output
            print(f"\n🤖 {model_name.upper()} Label Scores:")
            for label in labels:
                print(f"- {label}: {label_scores.get(label, 0.0):.2f}")
            print("🧠 Explanation:", explanation)

            # CSV row
            row = [model_name, text] + [label_scores.get(label, 0.0) for label in labels] + [explanation]
            writer.writerow(row)

        ratings.append(testimonial_ratings)

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

# # Save disagreement logs to Excel
# disagreement_output_path = "data/outputs/model_disagreements.xlsx"
# with pd.ExcelWriter(disagreement_output_path, engine="openpyxl") as writer:
#     disagreement_df.to_excel(writer, sheet_name="Raw Disagreements", index=False)
#     disagreement_summary.to_excel(writer, sheet_name="Summary", index=False)
    
# Save disagreement logs to Excel with summary and flags
disagreement_output_path = "data/outputs/model_disagreements.xlsx"
export_disagreements_to_excel(
    disagreement_df, 
    disagreement_summary, 
    flagged_testimonials,
    model_disagreement_summary, 
    disagreement_output_path
)

print(f"📉 Disagreement log saved to {disagreement_output_path}")
