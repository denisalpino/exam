FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    ghostscript \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Копируем только requirements.txt сначала
COPY ./rag/requirements.txt /app/requirements.txt

# Устанавливаем зависимости поэтапно
RUN pip install --no-cache-dir --upgrade pip && \
    pip install crewai && \
    pip install 'crewai[tools]' && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --upgrade embedchain && \
    pip install 'camelot-py[cv]' && \
    pip install ollama

# Копируем остальные файлы
COPY ./common /app/common
COPY ./rag /app

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]