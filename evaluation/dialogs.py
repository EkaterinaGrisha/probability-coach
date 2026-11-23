import argparse
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from langgraph.checkpoint.memory import InMemorySaver

from coach import stats
from coach.agent import build_agent
from coach.search import search


def theory_searched(calls):
    return any(c["name"] == "knowledge_search" and c["args"].get("kind", "theory") == "theory" for c in calls)


def cites_source(answer):
    return any(word in answer for word in ("Источник", "Википеди", "статья", "статье"))


# реплики пользователя и проверки последнего хода агента: вызванные инструменты и текст ответа
DIALOGS = [
    (["Дай задачу про двух детей", "1/2, уверена на 80%. Покажите симуляцию"], {
        "после неверного ответа и просьбы о симуляции агент ищет теорию":
            lambda calls, answer: theory_searched(calls),
        "без поиска теории агент не называет источник":
            lambda calls, answer: theory_searched(calls) or not cites_source(answer),
    }),
    (["Дай задачу про двери и ведущего", "Думаю, менять не надо"], {
        "ответ без уверенности не сохраняется": lambda calls, answer: not any(c["name"] == "save_attempt" for c in calls),
    }),
    (["Какова вероятность, что у двоих из 23 человек совпадут дни рождения?"], {
        "вопрос, похожий на задачу, ищется в теории":
            lambda calls, answer: not any(c["args"].get("kind") == "problem" for c in calls),
    }),
]


def main():
    parser = argparse.ArgumentParser(description="Проверка поведения агента на целых диалогах")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    # модели поиска грузятся при первом вызове; без прогрева параллельные потоки загружают их одновременно
    search("прогрев", 1, {"kind": "theory"})
    agent = build_agent(InMemorySaver())

    def run(dialog):
        turns, checks = dialog
        user = f"check-{uuid.uuid4().hex[:8]}"
        run_config = {"configurable": {"thread_id": user, "user_id": user}}
        for text in turns:
            before = len(agent.get_state(run_config).values.get("messages", []))
            messages = agent.invoke({"messages": [{"role": "user", "content": text}]}, run_config)["messages"][before:]
        with stats.connect() as conn:
            conn.execute("DELETE FROM attempts WHERE user_id = %s", (user,))
        calls = [c for m in messages if m.type == "ai" for c in m.tool_calls]
        return [(name, check(calls, messages[-1].content)) for name, check in checks.items()]

    with ThreadPoolExecutor(3) as pool:
        results = [r for rs in pool.map(run, [d for d in DIALOGS for _ in range(args.repeats)]) for r in rs]

    passed = Counter(name for name, ok in results if ok)
    for _, checks in DIALOGS:
        for name in checks:
            print(f"{passed[name]:2}/{args.repeats}  {name}")


if __name__ == "__main__":
    main()
