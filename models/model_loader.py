import yaml
from dotenv import load_dotenv
from models.ollama_model import OllamaModel
from models.gpt_model import GPTModel
from models.claude_model import ClaudeModel
from models.gemini_model import GeminiModel
import os

load_dotenv()

def load_models_from_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_names = config.get("models", [])
    model_settings = config.get("model_settings", {})
    loaded_models = {}

    for name in model_names:
        temperature = model_settings.get(name, {}).get("temperature", 0.7)

        if name in ["mistral", "llama3", "qwen:7b", "mixtral"]:
            loaded_models[name] = OllamaModel(model_name=name, temperature=temperature)

        elif name == "gpt":
            api_key = os.getenv("OPENAI_API_KEY")
            print("Loaded GPT API Key:", api_key[:8], "...")  # confirm
            loaded_models[name] = GPTModel(api_key=api_key, model="gpt-4", temperature=temperature)

        elif name == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            loaded_models[name] = ClaudeModel(api_key=api_key, temperature=temperature)

        elif name == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            loaded_models[name] = GeminiModel(api_key=api_key, temperature=temperature)

        else:
            raise ValueError(f"Unsupported model: {name}")

    return loaded_models

