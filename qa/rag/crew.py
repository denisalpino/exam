from typing import Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool, tool
from crewai import Agent, Task, Crew, Process, LLM
import agentops

from common.settings import settings
from rag_app import app as rag_app

agentops.init(api_key=settings.AGENTOPS_TOKEN, default_tags=["crewai"])

# Инструмент 1: Запрос к базе знаний
class KnowledgeBaseInput(BaseModel):
    query: str = Field(..., description="Фактологический запрос о программах, дисциплинах и учебных планах")

class KnowledgeBaseTool(BaseTool):
    name: str = "knowledge_base_query"
    description: str = "Используется для фактологических вопросов о программах, дисциплинах и учебных планах"
    args_schema: Type[BaseModel] = KnowledgeBaseInput

    def _run(self, query: str) -> str:
        return rag_app.query(input_query=query) # type: ignore

# Инструмент 2: Рекомендация курсов
class CoursesRecommendationInput(BaseModel):
    context: str = Field(..., description="Контекст с предпочтениями абитуриента")

class CoursesRecommendationTool(BaseTool):
    name: str = "courses_recommendation"
    description: str = "Используется для генерации персонализированных рекомендаций по выборным дисциплинам"
    args_schema: Type[BaseModel] = CoursesRecommendationInput

    def _run(self, context: str) -> str:
        prompt = f"""
        На основе контекста:
        {context}

        Сгенерируй рекомендации по выборным дисциплинам. Учти:
        - Предыдущее образование
        - Профессиональные цели
        - Технические навыки
        - Предпочтения по нагрузке

        Формат ответа:
        1. [Название дисциплины] - [Краткое обоснование]
        2. ...
        """
        return rag_app.query(prompt) # type: ignore


# Создаем экземпляры инструментов
knowledge_tool = KnowledgeBaseTool()
courses_tool = CoursesRecommendationTool()

# Агент 1: Интервьюер
interviewer = Agent(
    role="Интервьюер",
    goal="Собрать информацию о бэкграунде и целях абитуриента",
    backstory="Ты дружелюбный консультант по образованию",
    llm=LLM(
        model=f"ollama/{settings.LLM_MODEL}",
        base_url=settings.OLLAMA_LLM_URL,
        temperature=0.2,
        timeout=600,
        top_p=0.95,
        top_k=20,
        seed=42,
        presence_penalty=-2,
        max_completion_tokens=6000,
        num_ctx=8000
    ),
    tools=[],
    verbose=True,
    allow_delegation=False
)

# Агент 2: Аналитик программ
program_analyst = Agent(
    role="Аналитик магистерских программ",
    goal="Анализировать учебные планы и программы",
    backstory="Ты эксперт по академическим программам",
    llm=LLM(
        model=f"ollama/{settings.LLM_MODEL}",
        base_url=settings.OLLAMA_LLM_URL,
        temperature=0.2,
        timeout=600,
        top_p=0.95,
        top_k=20,
        seed=42,
        presence_penalty=-2,
        max_completion_tokens=6000,
        num_ctx=8000
    ),

    tools=[knowledge_tool],
    verbose=True,
    allow_delegation=True
)

# Агент 3: Советник по курсам
course_advisor = Agent(
    role="Советник по выбору курсов",
    goal="Формировать персонализированные рекомендации",
    backstory="Ты специалист по академическому планированию",
    llm=LLM(
        model=f"ollama/{settings.LLM_MODEL}",
        base_url=settings.OLLAMA_LLM_URL,
        temperature=0.2,
        timeout=600,
        top_p=0.95,
        top_k=20,
        seed=42,
        presence_penalty=-2,
        max_completion_tokens=6000,
        num_ctx=8000
    ),
    tools=[courses_tool],
    verbose=True,
    allow_delegation=True
)

# Задачи для агентов
interview_task = Task(
    description="Собери информацию об абитуриенте: образование, цели, навыки, предпочтения\n\n{user_input}",
    agent=interviewer,
    expected_output="Структурированный профиль абитуриента"
)

analysis_task = Task(
    description="Анализ программ на основе профиля абитуриента",
    agent=program_analyst,
    expected_output="Сравнение программ и ключевых дисциплин"
)

advising_task = Task(
    description="Генерация рекомендаций по выборным дисциплинам",
    agent=course_advisor,
    expected_output="Персонализированный план с обоснованием выбора"
)

# Создаем экипаж
academic_crew = Crew(
    agents=[interviewer, program_analyst, course_advisor], # type: ignore
    tasks=[interview_task, analysis_task, advising_task],
    process=Process.sequential,
    verbose=True
)

def run_crew(user_input: str):
    return academic_crew.kickoff(inputs={'user_input': user_input})