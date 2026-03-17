from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from ...setting import RAGSettings
from dotenv import load_dotenv
import requests

load_dotenv()


class LocalRAGModel:
    """Factory class for creating LLM instances for the RAG chatbot."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def set(
        model_name: str = "llama3:8b-instruct-q8_0",
        system_prompt: str | None = None,
        host: str = "host.docker.internal",
        setting: RAGSettings | None = None,
    ):
        """Create and return an LLM instance based on the model name.

        Args:
            model_name: Name of the model to use (e.g., 'gpt-4', 'llama3.1:8b').
            system_prompt: Optional system prompt for the model.
            host: Host address for Ollama server.
            setting: RAGSettings instance with configuration.

        Returns:
            LLM instance (OpenAI or Ollama) configured with appropriate settings.

        Raises:
            ValueError: If model creation fails due to invalid config or missing API key.
        """
        setting = setting or RAGSettings()
        try:
            if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4-turbo"]:
                return OpenAI(
                    model=model_name,
                    temperature=setting.openai.temperature,
                    max_tokens=setting.openai.max_tokens,
                    api_key=setting.openai.api_key or None,
                )
            else:
                settings_kwargs = {
                    "tfs_z": setting.ollama.tfs_z,
                    "top_k": setting.ollama.top_k,
                    "top_p": setting.ollama.top_p,
                    "repeat_last_n": setting.ollama.repeat_last_n,
                    "repeat_penalty": setting.ollama.repeat_penalty,
                }
                return Ollama(
                    model=model_name,
                    system_prompt=system_prompt,
                    base_url=f"http://{host}:{setting.ollama.port}",
                    temperature=setting.ollama.temperature,
                    context_window=setting.ollama.context_window,
                    request_timeout=setting.ollama.request_timeout,
                    additional_kwargs=settings_kwargs,
                )
        except Exception as e:
            raise ValueError(f"Failed to create LLM instance for model '{model_name}': {str(e)}")

    @staticmethod
    def pull(host: str, model_name: str):
        """Pull a model from the Ollama server.

        Args:
            host: Host address of the Ollama server.
            model_name: Name of the model to pull.

        Returns:
            Response object from the pull request.

        Raises:
            ConnectionError: If unable to connect to Ollama server.
        """
        setting = RAGSettings()
        try:
            response = requests.post(
                f"http://{host}:{setting.ollama.port}/api/pull", json={"name": model_name}, stream=True
            )
            response.raise_for_status()  # Raise for bad status
            return response
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to pull model '{model_name}' from Ollama at {host}:{setting.ollama.port}: {str(e)}")

    @staticmethod
    def check_model_exist(host: str, model_name: str) -> bool:
        setting = RAGSettings()
        data = requests.get(f"http://{host}:{setting.ollama.port}/api/tags").json()
        if data["models"] is None:
            return False
        list_model = [d["name"] for d in data["models"]]
        if model_name in list_model:
            return True
        return False
