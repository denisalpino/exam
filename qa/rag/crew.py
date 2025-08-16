import os
from typing import Literal, Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool, tool
from crewai import Agent, Task, Crew, Process, LLM
import agentops

from common.settings import settings
from rag_app import app as rag_app

agentops.init(api_key=settings.AGENTOPS_TOKEN, default_tags=["crewai"])


PROGRAM_NAMES = ["Искуственный интеллект", "Управление ИИ-продуктами / AI Product"]

# Инструмент 1: Запрос к базе знаний
class ProgramInput(BaseModel):
    query:   str  = Field(..., description="Запрос о программе, в целом, ИЛИ ее дисциплинах и учебном плане")
    program: Literal["Искуственный интеллект", "Управление ИИ-продуктами / AI Product"] = Field(..., description=f"Название одной из программ, к которой относится запрос: {', '.join(PROGRAM_NAMES)}")
    plan:    bool = Field(..., description="Необходим ли учебный план программы (True) или ее общее описание (False)")

class ProgramTool(BaseTool):
    name: str = "ask_about_program"
    description: str = "Используется для фактологических вопросов о конкретной программе: ее общем описании ИЛИ дисциплинах и учебном плане"
    args_schema: Type[BaseModel] = ProgramInput

    def _run(self, query: str, program: Literal["Искуственный интеллект", "Управление ИИ-продуктами / AI Product"], plan: bool) -> str:
        # If the program name is incorrect, the agent will respond
        # that it was unable to find such a program.
        return rag_app.query(
            input_query=query,
            where={"$and": [dict(program=program), dict(plan=plan)]}
        ) # type: ignore

# Инструмент 2: Рекомендация курсов
class FetchProgramPlanInput(BaseModel):
    program: Literal["Искуственный интеллект", "Управление ИИ-продуктами / AI Product"] = Field(..., description=f"Название одной из программ, к которой относится запрос: {', '.join(PROGRAM_NAMES)}")

class FetchProgramPlanTool(BaseTool):
    name: str = "fetch_full_program_plan"
    description: str = "Возвращает полный план указанной программы"
    args_schema: Type[BaseModel] = FetchProgramPlanInput

    def _run(self, program: str) -> str:
        inversive_mapping = {
            "Искуственный интеллект": "ai",
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
        if os.path.exists(file_path) :
            with open(file_path, encoding="utf-8") as f: content = f.read()
            return f"Вот, полный учебный план программы {program}:\n\n{content}"
        return (
            f"Не смог найти нужного файла с учебным планом программы {program}. "
            f"Наверное, его еще не успели загрузить или он обновляется."
        )

        prompt = f"""
        Контекст: {context}

        Сгенерируй 3-5 ключевых рекомендаций по выборным дисциплинам. Формат:
        - [Название дисциплины]: [Краткое обоснование (1 предложение)]

        Требования:
        1. Только релевантные дисциплины для целей абитуриента
        2. Максимально конкретные рекомендации
        3. Не более 5 пунктов
        """
        return rag_app.query(input_query=prompt) # type: ignore


# Создаем экземпляры инструментов
program_tool            = ProgramTool()
fetch_program_plan_tool = FetchProgramPlanTool()

def create_agent(
    role: str, goal: str, backstory: str,
    tools: list,
    max_tokens: int = 1000,
    allow_delegation: bool = False,
    num_ctx: int = 4000,
    temperature: float = 0.3
) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=LLM(
            model=f"ollama/{settings.LLM_MODEL}",
            base_url=settings.OLLAMA_LLM_URL,
            max_completion_tokens=max_tokens,
            presence_penalty=-2,
            temperature=temperature,
            num_ctx=num_ctx,
            top_p=0.95,
            top_k=20,
            seed=42
        ),
        tools=tools,
        verbose=True,
        allow_delegation=allow_delegation
    )


# Агент 1: Менеджер программы (старший агент, управляет остальными)
program_manager = create_agent(
    role="Менеджер образовательных программ в университете ИТМО",
    goal=(
        "Проконсультировать абитуриента о различных аспектах образовательных программ, "
        "университете ИТМО, в целом, и поддерживать с ним диалог, координируя действия команды"
    ),
    backstory=(
        f"Как ответственный менеджер магистрских программ {', '.join(PROGRAM_NAMES)}, "
        "ты предпочитаешь вести диалог с абитуриентом, исключительно об университете ИТМО "
        "и аспектах связанных с перечисленными программами, так как только они находятся "
        "в зоне твоей ответственности."
    ),
    tools=[],
    max_tokens=4000,
    allow_delegation=True,
    num_ctx=6_000,
)
# Агент 2: Интервьюер
interviewer = create_agent(
    role="Интервьюер абитуриентов",
    goal="Создавать и наполнять профиль абитуриента, включающую информацию о бэкграунде, целях, навыках и амбициях",
    backstory=(
        "Как дружелюбный консультант по образованию в университете ИТМО ты умеешь"
        "ненавязчиво и уместно интересоваться пользователем и его жизнью"
    ),
    tools=[],
    max_tokens=400
)
# Агент 3: Аналитик программ
program_analyst = create_agent(
    role="Аналитик образовательных программ",
    goal=(
        "Предоставить полную запрашиваемую информацию об одной или нескольких магистрских "
        "программ в ИТМО или провести анализ между несколькими програмами."
    ),
    backstory=(
        f"Как опытный аналитик магистрских программ {', '.join(PROGRAM_NAMES)}, "
        "ты можешь активно и неоднакартно пользоваться инструментов `ask_about_program`, "
        "который предназначен для уточнения деталей об одной из несскольких программ."
    ),
    tools=[program_tool],
    max_tokens=1000
)
# Агент 4: Советник по курсам
course_advisor = create_agent(
    role="Индивидуальный советник по курсам",
    goal=(
        "В зависимости от задачи:\n"
        "    - либо составить подробный индивидуальный план обучения абитуриента, на основе его профиля\n"
        "    - либо предложить наиболее подходящие дисциплины, учитывая вводные условия и критерии"
    ),
    backstory=(
        "Как специалист по академическому планированию в университете ИТМО, ты умеешь выделять интересующие "
        "абтуриента направления, соспоставлять их с его текущими знаниями, а затем предлагать "
        "оптимальный индивидуально подобранный план дисциплин"
    ),
    tools=[fetch_program_plan_tool],
    max_tokens=600,
    num_ctx=12_000
)

manager_task = Task(
    description="""Проанализируй сообщение пользователя: "{user_input}"
Задумайся, какой ответ желает получить пользователь,
а затем ответь ему самостоятельно или же кординируй свою
команду ассистентов для получения оптимального ответа.

В твоем распоряжении два помощника:
- Аналитик образовательных программ: Фокусируется на общей информации о программах
- Индивидуальный советник по курсам: Составляет индивидуальный план обучения по программе или же рекомендует конкретный дисциплины

Если запрос пользователя очень далек от темы образования
и связанных процессов в университете ИТМО, корректно дай
понять ему о своем предназначении.

ОБЯЗАТЕЛЬНО пользуйся личной информацией об абитуриенте, его бэкграундом, целями, навыками
    """,
    expected_output=(
        "Профессиональный, доброжелательный и информативный ответ абитуриенту. "
        "Ответ может варьроваться по размеру в зависимости от задачи, "
        "однако если ответ велик, то стоит его четко и красиво стркутурировать."
    ),
    agent=program_manager,
    async_execution=False,
    #context=[analysis_task, advising_task]  # Gets research output as context
)

# Создаем экипаж
academic_crew = Crew(
    agents=[program_analyst, course_advisor, program_manager], # interviewer
    tasks=[manager_task],
    manager_agent=program_manager,
    process=Process.sequential,
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
) # type: ignore

async def run_crew(user_input: str):
    res = await academic_crew.kickoff_async(inputs={'user_input': user_input})
    return res.raw