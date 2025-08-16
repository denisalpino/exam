import os
from typing import Any, Dict

from embedchain import App
from embedchain.loaders.directory_loader import DirectoryLoader
from embedchain.utils.misc import DataType

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

# Add data
if os.path.exists(settings.DATA_DIR):
    # Collect all files with .txt extention
    txt_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(settings.DATA_DIR)
        for file in files
        if file.endswith('.txt')
    ]
    # If there is no .txt files, flag it
    if not txt_files:
        print("В репозитории не найдено файлов с расширением .txt")
    else:
        mapping = {
            "ai": "Искуственный интеллект",
            "ai_product": "Управление ИИ-продуктами / AI Product",
        }
        for fname in txt_files:
            # File name without extention
            name = fname.split(".")[0]

            # Make metadata about file (program name
            # and is it program or just overview)
            metadata: Dict[str, Any] = {
                "program": mapping[name.rstrip("_program").split("/")[-1]],
                "plan": name.endswith("_program")
            }
            print(fname)
            with open(fname, encoding="utf-8") as f: content = f.read()

            app.add(content, data_type=DataType.TEXT, metadata=metadata)
    print(f"Loaded data from {settings.DATA_DIR}")
else:
    print(f"Data directory not found: {settings.DATA_DIR}")

print("RAG system is ready!")
