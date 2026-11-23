import json
from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from coach import stats
from coach.data import TRAPS, load_problems
from coach.search import search
from coach.simulations import SIMULATIONS

PROBLEMS = {p["id"]: p for p in load_problems()}


def _user(config):
    return config.get("configurable", {}).get("user_id", "default")


def format_problem(p):
    sim = f"{p['simulation']} {json.dumps(p['sim_params'], ensure_ascii=False)}" if p["simulation"] else "нет"
    return (
        f"id: {p['id']} (ловушка: {p['trap']})\n"
        f"Условие: {p['text']}\n"
        f"Правильный ответ (не показывать, пока пользователь не ответит): {p['answer']}\n"
        f"Типичный интуитивный ответ: {p['intuitive_answer']}\n"
        f"Разбор: {p['explanation']}\n"
        f"Симуляция: {sim}"
    )


@tool
def knowledge_search(
    query: str,
    runtime: ToolRuntime,
    kind: Literal["theory", "problem"] = "theory",
    trap: Literal[tuple(TRAPS)] | None = None,
) -> str:
    """Поиск в базе знаний.

    kind="theory" — статьи о вероятностных парадоксах и когнитивных искажениях, чтобы объяснить ошибку со ссылкой на источник.
    kind="problem" — банк задач; trap сужает поиск до одного типа ловушки, уже решённые пользователем задачи не возвращаются.
    """
    if kind == "theory":
        question = next((m.content for m in reversed(runtime.state["messages"]) if m.type == "human"), None)
        docs = search(query, 4, {"kind": "theory"}, question=question)
    else:
        flt = {"kind": "problem", "problem_id": {"$nin": stats.solved(_user(runtime.config))}}
        if trap:
            flt["trap"] = trap
        docs = search(query, 2, flt)

    if not docs:
        return "Ничего не найдено: возможно, все подходящие задачи уже решены."
    if kind == "theory":
        return "\n\n".join(f"[Источник: Википедия, «{d.metadata['title']}»]\n{d.page_content}" for d in docs)
    return "\n\n".join(format_problem(PROBLEMS[d.metadata["problem_id"]]) for d in docs)


@tool
def simulate(name: Literal[tuple(SIMULATIONS)], params: dict) -> str:
    """Запускает симуляцию или раскладывает задачу на натуральные частоты и сохраняет график, где видно, откуда берётся ответ.

    Параметры: monty_hall(n_doors, host_knows); base_rate(prevalence, sensitivity, false_positive_rate, population, labels);
    birthday(n_people, mode="any_pair" или "my_birthday"); gamblers_fallacy(streak_length, p);
    small_samples(sizes, more_than, p); regression_to_mean(reliability, share);
    two_children(condition="any_boy" или "older_boy"); simpson(groups={вариант: {подгруппа: [успехов, всего]}}).
    Для задачи из банка берите параметры из её описания.
    """
    try:
        result = SIMULATIONS[name](**params)
    except (TypeError, KeyError, ValueError) as e:
        return f"Неверные параметры для {name}: {e}"
    return json.dumps(result, ensure_ascii=False)


@tool
def save_attempt(problem_id: str, is_correct: bool, confidence: int, config: RunnableConfig) -> str:
    """Сохраняет ответ пользователя на задачу: верно ли он решил и насколько был уверен (0–100).

    Вызывай, только когда пользователь сам назвал уверенность, или если он сдался (тогда confidence=0).
    """
    if problem_id not in PROBLEMS:
        return f"Задачи {problem_id} нет в банке."
    stats.save(_user(config), problem_id, PROBLEMS[problem_id]["trap"], is_correct, min(max(confidence, 0), 100))
    if is_correct:
        return "Сохранено."
    # правила в промпте не хватало: после просьбы о симуляции модель часто пропускала поиск теории
    # и называла источник, которого не видела
    return ("Сохранено. Ответ неверный: если в этом ходе ещё не было knowledge_search с kind=\"theory\", "
            "вызови его и объясняй ошибку по найденным фрагментам.")


@tool
def my_stats(config: RunnableConfig) -> str:
    """Статистика пользователя: точность по типам ловушек (слабые первыми), средняя уверенность
    против доли верных ответов, калибровка по корзинам уверенности (0–20%, …, 80–100%)
    и Brier score (0 — идеальная калибровка, 0.25 — как всегда отвечать «50 на 50»)."""
    return json.dumps(stats.summary(_user(config)), ensure_ascii=False)


TOOLS = [knowledge_search, simulate, save_attempt, my_stats]
