import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env first
load_dotenv()

# Grab key safely

api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env or environment variables.")

# Create client using the key explicitly
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.2
)

print(response.choices[0].message.content)




