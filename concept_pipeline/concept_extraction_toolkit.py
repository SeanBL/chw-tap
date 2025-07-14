import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from models.model_loader import get_model_response
from utils.config import load_config


load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load testimonials and return two parallel lists:
# - texts_for_embedding: plain content for clustering
# - texts_for_prompting: enriched metadata version for LLM prompt
def load_testimonials_with_metadata(path: str) -> Tuple[List[str], List[str]]:
    texts_for_embedding = []
    texts_for_prompting = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            content = " ".join(obj.get("content", []))
            if not content.strip():
                continue

            speaker = obj.get("speaker", "unknown")
            gender = obj.get("gender", "unknown")
            date = obj.get("date", "unknown")
            topic = obj.get("topic", "unknown")

            metadata_line = f"{speaker} ({gender}, {date}) on {topic}: {content}"

            texts_for_embedding.append(content)
            texts_for_prompting.append(metadata_line)

    return texts_for_embedding, texts_for_prompting

# Sanitize LLM output to ensure it is valid JSON

def sanitize_llm_output(text: str) -> str:
    # Step 1: Normalize smart quotes
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Step 2: Fix malformed "examples" list
    match = re.search(r'"examples"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if match:
        raw_block = match.group(1)

        pieces = re.split(r'(?<!\\)",\s*', raw_block)
        merged = []
        buffer = ""

        for part in pieces:
            part = part.strip().strip('"').strip("'")
            if part.endswith(".") and ":" in part:
                if buffer:
                    buffer += " " + part
                    merged.append(buffer.strip())
                    buffer = ""
                else:
                    merged.append(part.strip())
            else:
                buffer += " " + part

        if buffer:
            merged.append(buffer.strip())

        # Safely insert using re.escape
        fixed_examples = '"examples": [' + ", ".join(json.dumps(s) for s in merged) + ']'
        pattern = r'"examples"\s*:\s*\[.*?\]'
        text = re.sub(pattern, lambda _: fixed_examples, text, flags=re.DOTALL)

    # Step 3: Ensure balanced braces
    open_braces = text.count("{")
    close_braces = text.count("}")
    if open_braces > close_braces:
        text += "}" * (open_braces - close_braces)

    return text.strip()

# Prompt the LLM
def generate_concept(texts: List[str], model="gpt-4", provider="gpt") -> Dict:
    system_prompt = """
    You are an expert in health communication and qualitative research.

    Given a set of community health worker testimonials, identify one **nuanced, unified concept** that meaningfully connects them. The concept should reflect an underlying theme, motivation, or experience present across the testimonials.

    Please return your response as a **valid, minified JSON object** with the following structure:

    {
    "concept": "A short, clear title summarizing the central concept.",
    "rationale": "A concise explanation of why this concept applies to the group of testimonials.",
    "examples": [
        "Speaker: Example sentence or quote.",
        "Speaker: Another example supporting the concept."
    ]
    }

    ❗ Do not include markdown, bullet points, commentary, or introductory text.
    ✅ Return **only** the JSON object — no additional explanation.
    """


    prompt = "Testimonials:\n"
    for i, text in enumerate(texts, start=1):
        prompt += f"{i}. \"{text}\"\n"

    result_str = get_model_response(prompt=prompt, system_prompt=system_prompt, model=model, provider=provider)
    print(f"🔍 Using model: {model} (provider: {provider})")
    print("\n📤 Raw LLM Output:")
    print(result_str) 
    try:
        cleaned = sanitize_llm_output(result_str)
        result = json.loads(cleaned)
    except Exception as e:
        print("❌ Failed to parse cleaned JSON:")
        print(result_str)  # print original response instead
        raise e

    # Safeguard: if the model returns fewer or no examples, fill them manually
    if "examples" not in result or not isinstance(result["examples"], list) or not result["examples"]:
        result["examples"] = texts[:5]

    return result

# Save results
def save_outputs(data: List[Dict], json_path: str, xlsx_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    df = pd.DataFrame(data)
    df.to_excel(xlsx_path, index=False)

# Approach 1: Cluster then prompt
def approach_1_cluster_then_prompt(embedding_texts, prompt_texts, n_clusters=20, model="gpt-4", provider="gpt"):
    embeddings = embedding_model.encode(embedding_texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)

    clustered_prompt_texts = [[] for _ in range(n_clusters)]
    for emb_text, label in zip(prompt_texts, kmeans.labels_):
        clustered_prompt_texts[label].append(emb_text)

    results = []
    for cluster in clustered_prompt_texts:
        if len(cluster) >= 2:
            result = generate_concept(cluster[:10], model=model, provider=provider)
            results.append(result)
    save_outputs(results, "data/outputs/llm_concepts/cluster_then_prompt.json", "data/outputs/llm_concepts/cluster_then_prompt.xlsx")

# Approach 2: Summarize then prompt
def approach_2_summarize_then_prompt(prompt_texts, batch_size=10, model="gpt-4", provider="gpt"):
    batches = [prompt_texts[i:i + batch_size] for i in range(0, len(prompt_texts), batch_size)]
    results = []
    for batch in batches:
        result = generate_concept(batch, model=model, provider=provider)
        results.append(result)
    save_outputs(results, "data/outputs/llm_concepts/summarize_then_prompt.json", "data/outputs/llm_concepts/summarize_then_prompt.xlsx")

# Approach 3: Bulk prompt
def approach_3_bulk_prompt(prompt_texts, max_batch=30, model="gpt-4", provider="gpt"):
    trimmed = prompt_texts[:max_batch]
    result = generate_concept(trimmed, model=model, provider=provider)
    save_outputs([result], "data/outputs/llm_concepts/bulk_concept_extraction.json", "data/outputs/llm_concepts/bulk_concept_extraction.xlsx")

if __name__ == "__main__":
    os.makedirs("data/outputs/llm_concepts", exist_ok=True)
    
    # Choose model and provider
    config = load_config()
    MODEL = config.get("model", "gpt")
    PROVIDER = config.get("provider", "openai")
    N_CLUSTERS = config.get("n_clusters", 20)

    # Load testimonials with metadata
    embedding_texts, prompt_texts = load_testimonials_with_metadata("data/processed/testimonials.jsonl")

    approach_1_cluster_then_prompt(embedding_texts, prompt_texts, model=MODEL, provider=PROVIDER)
    print("✅ Finished Approach 1")
    print("✅ Saved results to outputs/approach_1.json")
    approach_2_summarize_then_prompt(prompt_texts, model=MODEL, provider=PROVIDER)
    print("✅ Finished Approach 2")
    print("✅ Saved results to outputs/approach_2.json")
    approach_3_bulk_prompt(prompt_texts, model=MODEL, provider=PROVIDER)
    print("✅ Finished Approach 3")
    print("✅ Saved results to outputs/approach_3.json")
