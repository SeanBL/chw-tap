import os
import json
import pandas as pd
import tiktoken
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from utils.llm_helpers import summarize_cluster_with_llm

### ─────────────────────────────────
### Load Testimonials
### ─────────────────────────────────

def load_testimonials(jsonl_path: str) -> List[str]:
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "content" in obj:
                # Sanitize: join, strip, and replace stray newlines or excessive spaces
                content = " ".join(obj["content"]).strip().replace("\n", " ")
                texts.append(content)
    return texts

### ─────────────────────────────────
### Track A: BERTopic – Common Concepts
### ─────────────────────────────────

def generate_common_concepts(texts: List[str], top_n: int = 10) -> BERTopic:
    vectorizer = CountVectorizer(stop_words="english")
    umap_model = umap.UMAP(n_neighbors=15, n_components=2, metric="cosine")
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean", prediction_data=True)

    topic_model = BERTopic(
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model
    )

    topics, _ = topic_model.fit_transform(texts)

    freq_df = topic_model.get_topic_info().head(top_n)
    print("\n📊 Top Common Concepts:")
    print(freq_df[["Topic", "Name", "Count"]])
    return topic_model

def save_topic_summaries_with_texts(model: BERTopic, texts: List[str], out_path: str, top_n: int = 10):
    topic_info = model.get_topic_info()
    topics = model.get_topics()
    doc_topics, _ = model.transform(texts)

    summaries = []
    for i in range(1, min(len(topic_info), top_n + 1)):
        topic_row = topic_info.iloc[i]
        topic_id = topic_row["Topic"]
        name = topic_row["Name"]
        count = topic_row["Count"]
        words = [w.split(":")[0] for w in name.split("_") if ":" not in w]

        # Collect all texts assigned to this topic
        examples = [texts[j] for j in range(len(doc_topics)) if doc_topics[j] == topic_id]

        summaries.append({
            "topic_id": int(topic_id),
            "label": name,
            "top_words": words,
            "count": int(count),
            "examples": examples  # Include full text examples here
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved top {top_n} common topics with examples → {out_path}")

def save_common_concepts_xlsx(common_concepts_path: str, xlsx_path: str):
    with open(common_concepts_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    rows = []
    for concept in concepts:
        row = {
            "Topic ID": concept["topic_id"],
            "Label": concept["label"],
            "Top Words": ", ".join(concept["top_words"]),
            "Count": concept["count"]
        }
        for i, ex in enumerate(concept["examples"]):
            row[f"Example {i+1}"] = ex
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(xlsx_path, index=False)
    print(f"✅ Saved common concepts Excel → {xlsx_path}")

### ─────────────────────────────────
### Track B: Nuanced Concepts via Semantic Clustering
### ─────────────────────────────────

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def truncate_cluster_by_tokens(cluster_texts: List[str], max_tokens: int = 7500) -> List[str]:
    selected = []
    total_tokens = 0
    for text in cluster_texts:
        token_count = estimate_tokens(text)
        if total_tokens + token_count > max_tokens:
            break
        selected.append(text)
        total_tokens += token_count
    return selected

def extract_nuanced_concepts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> Tuple[Dict, Dict]:
    print("\n🔍 Extracting nuanced clusters...")
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, convert_to_tensor=False)

    reducer = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine').fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean').fit(reducer)

    nuanced_concepts = {}
    cluster_map = {}

    for label in set(clusterer.labels_):
        if label == -1:
            continue

        cluster_texts = [texts[i] for i in range(len(texts)) if clusterer.labels_[i] == label]
        print(f"🔍 Processing cluster {label} with {len(cluster_texts)} docs")

        # Use token-aware truncation instead of naive truncation
        trimmed = truncate_cluster_by_tokens(cluster_texts, max_tokens=7500)

        try:
            concept_name, explanation = summarize_cluster_with_llm(trimmed)
        except Exception as e:
            print(f"❌ Failed to summarize cluster {label}: {e}")
            continue

        cluster_key = f"cluster_{label}"
        nuanced_concepts[cluster_key] = {
            "cluster_id": int(label),
            "concept_name": concept_name,
            "explanation": explanation,
            "examples": trimmed[:3]
        }
        cluster_map[cluster_key] = cluster_texts

    print(f"✅ Extracted {len(nuanced_concepts)} nuanced concepts")
    return nuanced_concepts, cluster_map

def save_nuanced_concepts(concepts: Dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved nuanced concepts → {out_path}")

def save_nuanced_concepts_xlsx(concepts: Dict, xlsx_path: str):
    rows = []
    for title, data in concepts.items():
        row = {
            "Concept": title,
            "Explanation": data["explanation"]
        }
        for i, ex in enumerate(data["examples"]):
            row[f"Example {i+1}"] = ex
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(xlsx_path, index=False)
    print(f"✅ Saved nuanced concepts Excel → {xlsx_path}")

### ─────────────────────────────────
### Run Both Tracks
### ─────────────────────────────────

if __name__ == "__main__":
    jsonl_path = "data/processed/testimonials.jsonl"
    model_path = "models/bertopic_model"
    common_path = "data/concepts/raw_concepts.json"
    nuanced_path = "data/concepts/nuanced_concepts.json"

    texts = load_testimonials(jsonl_path)

    if not texts:
        print("❌ No testimonial content found.")
    else:
        # Track A
        model = generate_common_concepts(texts, top_n=10)
        model.save(model_path)
        print(f"✅ BERTopic model saved → {model_path}")
        save_topic_summaries_with_texts(model, texts, common_path, top_n=10)
        save_common_concepts_xlsx(common_path, "data/concepts/raw_concepts.xlsx")

        # Track B
        nuanced_concepts, cluster_map = extract_nuanced_concepts(texts)
        save_nuanced_concepts(nuanced_concepts, nuanced_path)
        save_nuanced_concepts_xlsx(nuanced_concepts, "data/concepts/nuanced_concepts.xlsx")
        save_nuanced_concepts(cluster_map, "data/concepts/clustered_testimonials.json")
