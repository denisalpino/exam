#!/bin/bash
# запускаем сервер в фоне
ollama serve &
SERVER_PID=$!
# ждём, пока сервер ответит
while ! curl -s http://localhost:11434/api/tags; do sleep 1; done
# скачиваем модель
ollama pull "$GENERATIVE_MODEL"
# перезапускаем сервер, чтобы подхватить модель
kill $SERVER_PID
exec ollama serve