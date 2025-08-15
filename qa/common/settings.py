# common/settings.py
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Telegram Bot
    TELEGRAM_TOKEN: str = Field(default="", description="Telegram Bot Token from @BotFather")

    # AgeentOps
    AGENTOPS_TOKEN: str = Field(default="", description="AgentOps monitoring service token")

    # RAG CrewAI Service
    RAG_CREWAI_URL: str = Field(
        default="http://rag:8000/process",
        description="URL to RAG CrewAI service endpoint"
    )

    # Ollama Services
    OLLAMA_LLM_URL: str = Field(
        default="http://localhost:11434",
        description="URL for Ollama LLM service"
    )
    OLLAMA_EMBEDDING_URL: str = Field(
        default="http://localhost:11435",
        description="URL for Ollama Embedding service"
    )

    # Data Paths
    DATA_DIR: str = Field(
        default="/app/data",
        description="Directory with data files"
    )

    # Models
    LLM_MODEL: str = Field(
        default="hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q6_K_L",
        description="Ollama model for LLM"
    )
    EMBEDDING_MODEL: str = Field(
        default="hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
        description="Ollama model for embeddings"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()