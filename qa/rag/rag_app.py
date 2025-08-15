import os

from embedchain import App
from embedchain.loaders.directory_loader import DirectoryLoader

from common.settings import settings
from parser import parse


# Парсим PDF перед инициализацией
print("Starting data parsing...")
from parser import parse
try:
    parse()
    print("Data parsing completed!")
except Exception as e:
    print(f"Error during parsing: {str(e)}")

# Конфигурация RAG
lconfig = {"extensions": [".txt"]}
loader = DirectoryLoader(config=lconfig)

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": settings.LLM_MODEL,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False,
            "base_url": settings.OLLAMA_LLM_URL
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": settings.EMBEDDING_MODEL,
            "vector_dimension": 1024,
            "base_url": settings.OLLAMA_EMBEDDING_URL,
        },
    },
}

print("Initializing RAG app...")
app = App.from_config(config=config)

# Добавляем данные
if os.path.exists(settings.DATA_DIR):
    app.add(settings.DATA_DIR, loader=loader)
    print(f"Loaded data from {settings.DATA_DIR}")
else:
    print(f"Data directory not found: {settings.DATA_DIR}")

print("RAG system is ready!")
