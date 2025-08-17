import os
from typing import Literal, Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from crewai import Agent, Task, Crew, Process, LLM
import agentops

from common.settings import settings
from rag_app import app as rag_app

agentops.init(api_key=settings.AGENTOPS_TOKEN, default_tags=["crewai"])


PROGRAM_NAMES = ["Искусственный интеллект", "Управление ИИ-продуктами / AI Product"]

# Инструмент 1: Запрос к базе знаний
class ProgramInput(BaseModel):
    query:   str = Field(..., description="Запрос о программе, в целом, ИЛИ ее дисциплинах и учебном плане")
    program: str = Field(..., description=f"Название одной из программ, к которой относится запрос: {' или '.join(PROGRAM_NAMES)}")


class ProgramTool(BaseTool):
    name: str = "ask_about_program"
    description: str = "Используется для фактологических вопросов о конкретной программе"
    args_schema: Type[BaseModel] = ProgramInput

    def _run(self, query: str, program: str) -> str:
        if program not in PROGRAM_NAMES:
            return (
                "Введено неправильное название программы. "
                f"Доступны только следующие: {', '.join(PROGRAM_NAMES)}"
            )

        _, cits = rag_app.query(
            input_query=query,
            where={"$and": [dict(program=program), dict(plan=False)]},
            citations=True
        )
        print(_)
        print(cits)
        answer = ["Relevant content found:"]
        for i, item in enumerate(cits):
            answer.append(f'{i}. {item[0]}')
        return "\n".join(answer)

# Инструмент 2: Рекомендация курсов
class FetchProgramPlanInput(BaseModel):
    program: str = Field(..., description=f"Название одной из программ, к которой относится запрос: {', '.join(PROGRAM_NAMES)}")


class FetchProgramPlanTool(BaseTool):
    name: str = "fetch_full_program_plan"
    description: str = "Возвращает полный план указанной программы"
    args_schema: Type[BaseModel] = FetchProgramPlanInput

    def _run(self, program: str) -> str:
        inversive_mapping = {
            "Искусственный интеллект": "ai",
            "Управление ИИ-продуктами / AI Product": "ai_product"
        }

        # Try too map program name
        try:
            program_name = inversive_mapping[program]
        except KeyError:
            return (
                "Введено неправильное название программы. "
                f"Доступны только следующие: {', '.join(inversive_mapping.keys())}"
            )

        # Try to find and return file content
        fname = f'{program_name}_program.txt'
        file_path = os.path.join(settings.DATA_DIR, fname)
        if os.path.exists(file_path):
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return f"Вот, полный учебный план программы {program}:\n\n{content}"
        return (
            f"Не смог найти нужного файла с учебным планом программы {program}. "
            f"Наверное, его еще не успели загрузить или он обновляется."
        )


# Создаем экземпляры инструментов
program_tool = ProgramTool()
fetch_program_plan_tool = FetchProgramPlanTool()


async def run_crew(user_input: str):
    # Агент 1: Менеджер программ
    manager = Agent(
        role="Менеджер программ ИТМО",
        goal="Используя предоставленную помощниками информацию, способствующую удовлетворению пользоваетльского запроса, сформировать структурированный и компактый окончательный ответ пользователю, удалив все Markdown форматирование и добавив смайлики",
        backstory=f"Как дружелюбный и ответственный менеджер университета ИТМО, ты ведешь разговор с абитуриентом только о следующих магистрских программах: {' и '.join(PROGRAM_NAMES)}. Если абитуриент спрашивает об отдаленных от программ тем, корректно сообщи о своем предназначении",
        llm=LLM(
            model=f"ollama/{settings.LLM_MODEL}",
            base_url=settings.OLLAMA_LLM_URL,
            temperature=0.2,
            timeout=600,
            top_p=0.95,
            top_k=20,
            seed=42,
            presence_penalty=-2
        ),
        tools=[],
        verbose=True,
        allow_delegation=False
    )

    # Агент 2: Аналитик программ
    program_analyst = Agent(
        role="Аналитик магистерских программ",
        goal="Используя испключительно информацию о магистрских программах ИТМО сформировать компактный и персонализированный ответ",
        backstory=f"Ты эксперт по академическим программам и умеешь при помощи инструментов итеративно извлекать точную информацию о программах. В твоем распоряжении лишь 2 программы {' и '.join(PROGRAM_NAMES)}",
        llm=LLM(
            model=f"ollama/{settings.LLM_MODEL}",
            base_url=settings.OLLAMA_LLM_URL,
            temperature=0.5,
            timeout=600,
            top_p=0.95,
            top_k=20,
            seed=42,
            presence_penalty=-2
        ),
        tools=[program_tool],
        verbose=True,
        allow_delegation=True
    )

    # Агент 3: Советник по курсам
    course_advisor = Agent(
        role="Советник по учебному плану",
        goal="Используя исключительно учебный план по интересуемой дисциплине сформировать компактный и персонализированный ответ",
        backstory=f"Как опытный специалист по академическому планированию в университете ИТМО, ты умеешь четко выполнять поставленные задачи используя для их решения предпочтения абитуриента и инструмент, который предоставляет полнуое описание дисциплин, входящих в кокретную программу. В твоем распоряжении лишь 2 программы {' и '.join(PROGRAM_NAMES)}",
        llm=LLM(
            model=f"ollama/{settings.LLM_MODEL}",
            base_url=settings.OLLAMA_LLM_URL,
            temperature=0.5,
            timeout=600,
            top_p=0.95,
            top_k=20,
            seed=42,
            presence_penalty=-2,
            num_ctx=16_384
        ),
        tools=[fetch_program_plan_tool],
        verbose=True,
        allow_delegation=True
    )

    main_task = Task(
        description=(
            "{user_input}\n\n"
            "Оцени, относится ли последний пользовательский вопрос, к одной "
            f"из магистрских программ университета ИТМО: {' или '.join(PROGRAM_NAMES)}!\n"
            "- Если тема запроса далека от данных двух программ, сразу же передай запрос с Менеджеру программ, чтобы тот обозначил неприемлемость\n"
            "- Если пользовательский запрос подразумевает интерес в указанных программах, выполни инструкции ниже\n\n"
            "На основе предоставленного контекста взамодействия:\n"
            "1. Выдели бэкграунд абитуриента, его цели и навыки\n"
            "2. Идентифицировать ключевую задачу, которую пользователь поручил решить.\n"
            "3. Решить задачу, делегируя подзадачи ответственным асистентам.\n"
            "4. Оформить результаты в компактное и персонализированное сообщение\n\n"
            "Правила управления:\n"
            '- Если задача касается дисциплин, входящих в какую-либо программу, используй помощника "Советник по учебному плану"\n' \
            '- Для получения любо другой информации о программах, используй "Аналитик магистерских программ"\n' \
            '- А чтобы агрегировать информацию и сформировать окончательный ответ. используй "Менеджер программ ИМТО"'
        ),
        expected_output="Ответ должен иметь менее 300 слов. Ответ должен быть четко и красиво стркутурирован обычным текстом БЕЗ ИСПОЛЬЗОВАНИЯ Markdown синтаксиса, предпочитая использование смайликов"
    )

    # Создаем экипаж
    academic_crew = Crew(
        agents=[manager, program_analyst, course_advisor], # type: ignore
        tasks=[main_task],
        process=Process.hierarchical,
        manager_llm=LLM(
            model=f"ollama/{settings.LLM_MODEL}",
            base_url=settings.OLLAMA_LLM_URL,
            temperature=0.5,
            timeout=600,
            top_p=0.95,
            top_k=20,
            seed=42,
            presence_penalty=-2,
            num_ctx=12_000
        ),
        verbose=True,
        memory=True,
        embedder={
            "provider": "ollama",
            "config": {
                "model": settings.EMBEDDING_MODEL,
                "vector_dimension": 1024,
                "url": settings.OLLAMA_EMBEDDING_URL,
            }
        }
    )
    res = await academic_crew.kickoff_async(inputs={'user_input': user_input})
    return res.raw