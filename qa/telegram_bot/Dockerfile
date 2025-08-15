FROM python:3.10-slim

WORKDIR /app

# Копируем общие модули
COPY ./common /app/common

# Копируем код Telegram бота
COPY ./telegram_bot /app

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["python", "bot.py"]