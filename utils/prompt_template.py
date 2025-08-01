from typing import List, Dict

# prompt with explanation
def generate_prompt(text: str, labels: List[str], concept_definitions: Dict[str, Dict[str, str]]) -> str:
    if len(labels) != 1:
      raise ValueError(f"Expected exactly one concept, but got {len(labels)}: {labels}")

    label = labels[0]
    info = concept_definitions.get(label, {})
    definition = info.get("definition", "")
    inclusion = info.get("inclusion", "")
    exclusion = info.get("exclusion", "")
    examples = info.get("examples", "")  # Optionally pull example block

    label_block = (
        f"- {label}:\n"
        f"  Definition: {definition}\n"
        f"  Inclusion: {inclusion}\n"
        f"  Exclusion: {exclusion}"
    )

    if examples:
        label_block += f"\n  Examples:\n{examples}"

    return f"""
You are a researcher classifying community health worker (CHW) testimonials using predefined concepts.

Each concept includes a definition, inclusion criteria, exclusion criteria, and examples. Your task is to assign a **score from 0.0 to 1.0** to the concept indicating how well it applies to the testimonial.

Scoring Rules:
- 1.0 = Perfect, unambiguous match
- 0.8–0.9 = Strong direct alignment
- 0.5–0.7 = Moderate or partial alignment
- 0.1–0.4 = Weak alignment (minimal evidence)
- 0.0 = Concept clearly not present or explicitly excluded

You must return a valid JSON object in the following format:
{{
  "labels": {{
    "{label}": score
  }},
  "explanation": "Brief justification for the score. Reference the concept definition or example where applicable."
}}

Only include 'labels' and 'explanation'. Do not include any other output.

Concept:
{label_block}

Testimonial:
\"\"\"{text}\"\"\"
""".strip()

# prompt without explanation
# def generate_prompt(text: str, labels: List[str], concept_definitions: Dict[str, Dict[str, str]]) -> str:
#     label_details = []
#     for label in labels:
#         info = concept_definitions.get(label, {})
#         definition = info.get("definition", "")
#         inclusion = info.get("inclusion", "")
#         exclusion = info.get("exclusion", "")
#         label_details.append(
#             f"- {label}:\n"
#             f"  Definition: {definition}\n"
#             f"  Inclusion: {inclusion}\n"
#             f"  Exclusion: {exclusion}"
#         )

#     return f"""
# You are a researcher classifying community health worker testimonials.

# Use the following concept definitions to assign each label a score between 0.0 and 1.0 based on how well it applies to the testimonial.

# You must return a valid JSON object **exactly** in the following format:
# {{
#   "labels": {{
#     "label1": score,
#     ...
#   }}
# }}

# Concepts:
# {chr(10).join(label_details)}

# Testimonial:
# \"\"\"{text}\"\"\"
# """.strip()