from collections import Counter

import numpy as np
from langchain_deepseek import ChatDeepSeek
from langsmith import evaluate
from pydantic import BaseModel, Field

from coach import config
from coach.agent import build_agent
from coach.search import search

DATASET = "probability-coach-retrieval-v2"

JUDGE_PROMPT = """Ты проверяешь ответ ассистента. Ниже вопрос пользователя, фрагменты статей, которые ассистент нашёл в базе знаний, эталонная фраза с правильным ответом и сам ответ ассистента.

Оцени:
- faithfulness — доля утверждений ответа, которые подтверждаются фрагментами (1 — подтверждено всё, 0 — ничего);
- correctness — передаёт ли ответ то же, что эталонная фраза (1 — да, 0.5 — частично, 0 — нет);
- unsupported — утверждения ответа, которых во фрагментах нет. Общие связки и пересказ вопроса утверждениями не считай.

Вопрос: {question}

Фрагменты:
{context}

Эталонная фраза: {anchor}

Ответ ассистента:
{answer}"""


class Verdict(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    correctness: float = Field(ge=0, le=1)
    unsupported: list[str]


def main():
    # модели поиска грузятся при первом вызове; без прогрева параллельные потоки загружают их одновременно
    search("прогрев", 1, {"kind": "theory"})
    agent = build_agent()
    judge = ChatDeepSeek(model=config.DEEPSEEK_MODEL, temperature=0).with_structured_output(Verdict)

    def answer(inputs):
        messages = agent.invoke({"messages": [{"role": "user", "content": inputs["question"]}]})["messages"]
        context = "\n\n".join(m.content for m in messages if m.type == "tool" and m.name == "knowledge_search")
        return {"answer": messages[-1].content, "context": context}

    def judge_answer(inputs, outputs, reference_outputs):
        if not outputs["context"]:
            return {"results": [{"key": "used_search", "score": 0}]}
        verdict = judge.invoke(JUDGE_PROMPT.format(question=inputs["question"], context=outputs["context"],
                                                   anchor=reference_outputs["anchor"], answer=outputs["answer"]))
        anchor = " ".join(reference_outputs["anchor"].lower().split())
        return {"results": [
            {"key": "used_search", "score": 1},
            {"key": "anchor_found", "score": anchor in " ".join(outputs["context"].lower().split())},
            {"key": "faithfulness", "score": verdict.faithfulness, "comment": "; ".join(verdict.unsupported)},
            {"key": "correctness", "score": verdict.correctness},
        ]}

    results = evaluate(answer, data=DATASET, evaluators=[judge_answer], experiment_prefix="answers",
                       metadata={"model": config.DEEPSEEK_MODEL, "judge": config.DEEPSEEK_MODEL}, max_concurrency=4)

    rows = []
    for row in results:
        question = row["example"].inputs["question"]
        if row["run"].error:
            print(f"ошибка агента: {question[:60]!r}: {row['run'].error[:200]}")
            continue
        scores = {res.key: float(res.score) for res in row["evaluation_results"]["results"]}
        comment = next((res.comment for res in row["evaluation_results"]["results"] if res.key == "faithfulness"), "")
        rows.append({**scores, "lang": row["example"].outputs["lang"], "question": question, "unsupported": comment})

    def report(title, subset):
        print(f"\n{title}, n={len(subset)}")
        for key in ("used_search", "anchor_found", "faithfulness", "correctness"):
            values = [r[key] for r in subset if key in r]
            if values:
                means = np.random.default_rng(0).choice(values, (2000, len(values))).mean(axis=1)
                low, high = np.percentile(means, [2.5, 97.5])
                print(f"  {key:13} {np.mean(values):.2f}  95% ДИ {low:.2f}–{high:.2f}")

    judged = [r for r in rows if "correctness" in r]
    report("все вопросы", rows)
    for lang in ("ru", "en"):
        report(f"статья на языке {lang}", [r for r in judged if r["lang"] == lang])
    report("нужный фрагмент среди найденных", [r for r in judged if r["anchor_found"]])
    report("нужного фрагмента нет среди найденных", [r for r in judged if not r["anchor_found"]])

    print("\ncorrectness:", Counter(r["correctness"] for r in judged))
    print("faithfulness:", Counter("1" if r["faithfulness"] == 1 else "0.8–1" if r["faithfulness"] >= 0.8 else "<0.8"
                                   for r in judged))
    for r in sorted(judged, key=lambda r: r["faithfulness"])[:8]:
        print(f"\n{r['faithfulness']:.2f} / {r['correctness']} {r['question']}\n     неподтверждено: {r['unsupported']}")


if __name__ == "__main__":
    main()
