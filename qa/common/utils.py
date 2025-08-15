import re

def escape_markdown(text) -> str:
    """
    Экранирует спецсимволы Telegram MarkdownV2.
    Принимает любой тип данных, приводит к строке.
    """
    if text is None:
        return ""

    # Приводим к строке на всякий случай
    text = str(text)

    # Список символов, которые надо экранировать в MarkdownV2
    escape_chars = r"_*[]()~`>#+-=|{}.!<>"

    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)