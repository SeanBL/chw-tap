from utils.config import load_config
from models.model_loader import load_models_from_config

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

# Run analysis
with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Model", "Testimonial", *labels, "Explanation"])

    for i, text in enumerate(testimonials):
        print(f"\n📝 Testimonial {i + 1}:")
        print(text)

        for model_name, model in models.items():
            result = model.classify(text, labels)

            label_scores = result["labels"]
            explanation = result["explanation"]

            # Console output
            print(f"\n🤖 {model_name.upper()} Label Scores:")
            for label in labels:
                print(f"- {label}: {label_scores.get(label, 0.0):.2f}")
            print("🧠 Explanation:", explanation)

            # CSV row
            row = [model_name, text] + [label_scores.get(label, 0.0) for label in labels] + [explanation]
            writer.writerow(row)

print(f"\n✅ Results saved to {output_path}")



