import os
import json
import matplotlib.pyplot as plt
import pandas as pd

def load_irr_scores(path="data/outputs/irr_scores.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def visualize_irr_scores(irr_scores: dict, output_dir="data/outputs/visualizations"):
    os.makedirs(output_dir, exist_ok=True)

    per_label = irr_scores["per_label"]
    overall = irr_scores.get("overall", {})

    metrics = ["icc", "fleiss", "cohen", "krippendorff", "percent_agreement"]

    # Create one bar chart per metric across all labels
    for metric in metrics:
        labels = []
        values = []
        for label, scores in per_label.items():
            labels.append(label)
            values.append(scores[metric])
        
        # DEBUG
        print(f"\n[DEBUG] Plotting IRR metric: {metric}")
        print("[DEBUG] Labels:", labels)
        print("[DEBUG] Values:", values)

        # Convert values to float if necessary
        try:
            values = [float(v) for v in values]
        except ValueError as e:
            print("⚠️ Error: IRR score values must be numeric!", e)
            continue


        plt.figure(figsize=(10, 5))
        plt.bar(labels, values)
        plt.ylim(0, 1)
        plt.title(f"{metric.upper()} Scores by Label")
        plt.ylabel("Score")
        plt.xlabel("Label")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_by_label.png"))
        plt.close()

    # Display overall Krippendorff’s Alpha
    overall_score = overall.get("krippendorff", None)
    if overall_score is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Overall Krippendorff's Alpha"], [overall_score], color='green')
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Overall Krippendorff’s Alpha")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "overall_krippendorff.png"))
        plt.close()

def print_irr_table(irr_scores: dict):
    df = pd.DataFrame(irr_scores["per_label"]).T
    df.index.name = "Label"
    print("\n📋 Inter-Rater Reliability Table:\n")
    print(df.round(3))

def export_irr_to_excel(irr_scores: dict, output_path="data/outputs/irr_scores.xlsx"):
    df = pd.DataFrame(irr_scores["per_label"]).T
    df.index.name = "Label"

    # Add overall Krippendorff’s Alpha as a separate row
    overall_kripp = irr_scores.get("overall", {}).get("krippendorff", None)
    if overall_kripp is not None:
        overall_df = pd.DataFrame([{
            "icc": None,
            "fleiss": None,
            "cohen": None,
            "krippendorff": overall_kripp,
            "percent_agreement": None
        }], index=["OVERALL"])
        df = pd.concat([df, overall_df])

    df.to_excel(output_path)
    print(f"\n📁 IRR scores exported to {output_path}")

