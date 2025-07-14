import openai
import os
from typing import List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_cluster_with_llm(texts, model="gpt-4") -> tuple[str, str]:
    prompt = "You are a health research expert. Given the following testimonials, identify a **nuanced concept** that connects them. Then explain your reasoning.\n\n"
    prompt += "\n".join([f"- {text}" for text in texts])

    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in health communication."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()

        # Optional: basic heuristic split into name and rationale
        if "\n" in content:
            lines = content.split("\n", 1)
            concept_name = lines[0].replace("Concept:", "").strip()
            explanation = lines[1].strip()
        else:
            concept_name = "Unnamed Concept"
            explanation = content

        return concept_name, explanation

    except Exception as e:
        print("❌ Error calling OpenAI:", e)
        return "Error", str(e)
