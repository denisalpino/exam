import os
import requests
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode

from common.settings import settings
from common.utils import escape_markdown

bot = Bot(token=settings.TELEGRAM_TOKEN)
dp = Dispatcher()

MAX_MESSAGE_LENGTH = 4096

async def send_long_message(message: Message, text: str, parse_mode=None):
    """
    Отправка длинного текста в Telegram по частям
    """
    # Разбиваем на куски, стараясь резать по границам предложений
    while text:
        chunk = text[:MAX_MESSAGE_LENGTH]

        # Ищем последний перенос строки или точку в пределах куска
        cut_idx = max(chunk.rfind("\n"), chunk.rfind("."))
        if cut_idx == -1:
            cut_idx = len(chunk)

        part = chunk[:cut_idx+1]
        await message.answer(part, parse_mode=parse_mode)

        text = text[cut_idx+1:]



@dp.message(Command("start"))
async def start(message: Message):
    text = escape_markdown(
        "🎓 Привет! Я твой помощник по выбору магистерской программы.\n\n"
        "Расскажи о себе:\n"
        "- Твоё предыдущее образование\n"
        "- Профессиональные цели\n"
        "- Технические навыки\n"
        "- Предпочтения по учебной нагрузке\n\n"
        "Я помогу выбрать подходящую программу и дисциплины!"
    )
    await message.answer(text, parse_mode=ParseMode.MARKDOWN_V2)

@dp.message()
async def handle_message(message: Message):
    try:
        response = requests.post(
            settings.RAG_CREWAI_URL,
            json={"message": message.text},
            timeout=600
        )

        if response.status_code == 200:
            result = response.json()
            escaped_response = escape_markdown(result['response'])
            await send_long_message(message, escaped_response, parse_mode=ParseMode.MARKDOWN_V2)
        else:
            error_text = escape_markdown("⚠️ Произошла ошибка при обработке запроса. Попробуйте позже.")
            await message.answer(error_text, parse_mode=ParseMode.MARKDOWN_V2)

    except Exception as e:
        error_text = escape_markdown(f"🚫 Ошибка: {str(e)}")
        await message.answer(error_text, parse_mode=ParseMode.MARKDOWN_V2)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())