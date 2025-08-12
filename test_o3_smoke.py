import os
from openai import OpenAI

# Make sure these are set in your shell or .env:
# OPENAI_API_KEY=sk-...
# OPENAI_ORG_ID=org_...

def main():
    api = os.getenv("OPENAI_API_KEY")
    org = os.getenv("OPENAI_ORG_ID")
    assert api and org, "Set OPENAI_API_KEY and OPENAI_ORG_ID first."

    client = OpenAI(api_key=api, organization=org)

    r = client.responses.create(
        model="o3",
        instructions="You are a helpful classifier.",
        input="Say 'ok' if you can see this.",
        reasoning={"effort": "low"},
        # response_format={"type": "json_object"},  # optional strict JSON
    )

    print("\n=== o3 Smoke Test ===")
    print("output_text:", getattr(r, "output_text", None))
    print("raw (truncated):", str(r.to_dict())[:400], "...\n")

if __name__ == "__main__":
    main()