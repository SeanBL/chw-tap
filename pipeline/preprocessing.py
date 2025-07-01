import os
import re
import json
import hashlib
from docx import Document
from typing import List, Dict
from collections import OrderedDict


def clean_line(line: str) -> str:
    return re.sub(r'\s+', ' ', line).strip()


def generate_unique_id(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def merge_short_lines(lines: List[str], min_len: int = 40) -> List[str]:
    paragraph = ""
    merged = []
    for line in lines:
        line = clean_line(line)
        if not line:
            continue
        paragraph += (" " if paragraph else "") + line
        if len(paragraph) >= min_len:
            merged.append(paragraph.strip())
            paragraph = ""
    if paragraph:
        merged.append(paragraph.strip())
    return merged


def extract_testimonials(doc_path: str) -> List[Dict]:
    doc = Document(doc_path)
    raw_text = [clean_line(p.text) for p in doc.paragraphs if p.text.strip()]

    entries = []
    entry = OrderedDict({"topic": "unknown", "speaker": "unknown", "date": "unknown", "content": []})

    for line in raw_text:
        if line.lower().startswith("topic title:"):
            if entry["content"]:
                entry["content"] = merge_short_lines(entry["content"])
                entries.append(entry)
                entry = OrderedDict({"topic": "unknown", "speaker": "unknown", "date": "unknown", "content": []})
            entry["topic"] = line.split(":", 1)[-1].strip() or "unknown"
        elif line.lower().startswith("speaker:"):
            entry["speaker"] = line.split(":", 1)[-1].strip() or "unknown"
        elif line.lower().startswith("date:"):
            entry["date"] = line.split(":", 1)[-1].strip() or "unknown"
        else:
            entry["content"].append(line)

    if entry["content"]:
        entry["content"] = merge_short_lines(entry["content"])
        entries.append(entry)

    return entries


def save_as_jsonl(entries: List[Dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    input_folder = "data/validated"
    output_jsonl = "data/processed/testimonials.jsonl"
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    all_entries = []
    seen_hashes = set()

    for filename in os.listdir(input_folder):
        if filename.endswith(".docx"):
            file_path = os.path.join(input_folder, filename)
            try:
                entries = extract_testimonials(file_path)
                for entry in entries:
                    text_blob = " ".join(entry["content"])
                    content_hash = generate_unique_id(text_blob)
                    if content_hash not in seen_hashes:
                        seen_hashes.add(content_hash)
                        all_entries.append(entry)
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

    # Assign unique IDs starting from 1
    for idx, entry in enumerate(all_entries, 1):
        entry["id"] = idx

    save_as_jsonl(all_entries, output_jsonl)
    print(f"\n✅ Extracted {len(all_entries)} total unique testimonials → {output_jsonl}")
