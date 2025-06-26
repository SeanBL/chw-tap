import yaml
from models.claude_model import ClaudeModel
from models.ollama_model import OllamaModel
from models.gemini_model import GeminiModel
from models.gpt_model import GPTModel
from dotenv import load_dotenv
import os
# Later: add ClaudeModel, GeminiModel, etc.
load_dotenv()

def load_models_from_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_names = config.get("models", [])
    loaded_models = {}

    for name in model_names:
        if name in ["mistral", "llama3", "qwen:7b", "mixtral"]:
            loaded_models[name] = OllamaModel(model_name=name)
        elif name == "gpt":
            api_key = os.getenv("OPENAI_API_KEY")
            loaded_models[name] = GPTModel(api_key=api_key, model="gpt-4")

        elif name == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            loaded_models[name] = ClaudeModel(api_key=api_key)

        elif name == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            loaded_models[name] = GeminiModel(api_key=api_key)
        else:
            raise ValueError(f"Unsupported model: {name}")

    return loaded_models
