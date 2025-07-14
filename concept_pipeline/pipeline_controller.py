import os
import json
from validate_docx import validate_docs
from preprocessing import preprocess_raw_testimonials
from concept_generator import generate_concepts
from concept_extraction_toolkit import (
    approach_1_cluster_then_prompt,
    approach_2_summarize_then_prompt,
    approach_3_bulk_prompt
)
from concept_refiner import (
    load_general_concepts,
    load_nuanced_concepts,
    call_llm_refiner,
    save_json
)

# Configuration
RAW_DOC_PATH = "data/raw/"
PROCESSED_JSONL = "data/processed/testimonials.jsonl"
BER_TOPIC_OUTPUT = "data/outputs/bertopic_top_concepts.json"
TOOLKIT_OUTPUT = "outputs/cluster_then_prompt.json"
REFINED_OUTPUT = "data/outputs/refined_labels.json"

MODEL = "gpt-4"
PROVIDER = "gpt"  # Options: gpt, claude, gemini, mistral, etc.


def run_pipeline():
    print("\n✅ Step 1: Validate raw testimonial documents")
    validate_docs(RAW_DOC_PATH)

    print("\n✅ Step 2: Preprocess validated testimonials")
    preprocess_raw_testimonials(RAW_DOC_PATH, PROCESSED_JSONL)

    print("\n✅ Step 3: Run BERTopic to generate surface-level concepts")
    generate_concepts(PROCESSED_JSONL, top_n=10)  # saves to BER_TOPIC_OUTPUT

    print("\n✅ Step 4: Run concept extraction with LLMs")
    testimonials = []
    with open(PROCESSED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "content" in obj:
                testimonials.append(" ".join(obj["content"]))

    approach_1_cluster_then_prompt(testimonials, model=MODEL, provider=PROVIDER)
    approach_2_summarize_then_prompt(testimonials, model=MODEL, provider=PROVIDER)
    approach_3_bulk_prompt(testimonials, model=MODEL, provider=PROVIDER)

    print("\n🛑 Step 5: Please manually review the generated concepts before proceeding.")
    input("Press Enter to continue after manual review...")

    print("\n✅ Step 6: Refine concepts using selected model")
    general_labels = load_general_concepts(BER_TOPIC_OUTPUT)
    nuanced_labels = load_nuanced_concepts(TOOLKIT_OUTPUT)
    all_concepts = list(set(general_labels + nuanced_labels))
    result = call_llm_refiner(all_concepts, model=MODEL, provider=PROVIDER)
    save_json(result, REFINED_OUTPUT)

    print("\n🎉 Pipeline completed. Refined concepts saved to:", REFINED_OUTPUT)


if __name__ == "__main__":
    run_pipeline()
