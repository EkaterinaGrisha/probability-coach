import argparse
import json
from collections import defaultdict

import numpy as np
from langsmith import Client, evaluate

from coach import config
from coach.search import search
from coach.store import COLLECTION

DATASET = "probability-coach-retrieval-v2"
QUESTIONS = config.ROOT / "evaluation" / "retrieval_questions.json"
K = 5


def norm(text):
    return " ".join(text.lower().split())


def metrics(outputs, reference_outputs):
    # релевантным считается чанк, где есть опорная фраза: так разметка не зависит от нарезки на чанки
    anchor = norm(reference_outputs["anchor"])
    rank = next((i for i, chunk in enumerate(outputs["chunks"], 1) if anchor in norm(chunk)), None)
    # один оценщик вместо четырёх: LangSmith считает каждый вызов оценщика отдельной трассой
    return {"results": [
        {"key": "hit_at_1", "score": rank == 1},
        {"key": "hit_at_5", "score": rank is not None},
        {"key": "mrr", "score": 1 / rank if rank else 0.0},
        {"key": "article_hit_at_5", "score": reference_outputs["title"] in outputs["titles"]},
    ]}


def ci(values):
    # бутстреп по вопросам: разброс метрики, если бы вопросы были другой выборкой того же размера
    means = np.random.default_rng(0).choice(values, (2000, len(values))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def upload_dataset(client):
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    dataset = client.create_dataset(DATASET, description="Вопросы на русском к чанкам статей Википедии с опорной фразой")
    client.create_examples(dataset_id=dataset.id, examples=[
        {"inputs": {"question": q["question"]}, "outputs": {k: q[k] for k in ("anchor", "title", "lang", "chunk_id")}}
        for q in questions
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    client = Client()
    if not client.has_dataset(dataset_name=DATASET):
        upload_dataset(client)

    def retrieve(inputs):
        docs = search(inputs["question"], K, {"kind": "theory"}, args.collection, args.rerank)
        return {"chunks": [d.page_content for d in docs], "titles": [d.metadata["title"] for d in docs]}

    name = f"{config.EMBEDDING_MODEL.split('/')[-1]} · {args.collection}" + (" · rerank" if args.rerank else "")
    results = evaluate(
        retrieve,
        data=DATASET,
        evaluators=[metrics],
        experiment_prefix=name,
        metadata={"collection": args.collection, "k": K, "embedding": config.EMBEDDING_MODEL,
                  "reranker": config.RERANKER_MODEL if args.rerank else None},
    )

    scores = defaultdict(list)
    for row in results:
        lang = row["example"].outputs["lang"]
        for res in row["evaluation_results"]["results"]:
            scores[res.key, "все"].append(res.score)
            scores[res.key, lang].append(res.score)

    groups = ("все", "ru", "en")
    print(f"\n{name}: " + ", ".join(f"{g} — {len(scores['mrr', g])}" for g in groups))
    print(f"{'':18}" + "".join(f"{g:>8}" for g in groups))
    for key in ("hit_at_1", "hit_at_5", "mrr", "article_hit_at_5"):
        low, high = ci(scores[key, "все"])
        print(f"{key:18}" + "".join(f"{np.mean(scores[key, g]):8.2f}" for g in groups) + f"   95% ДИ {low:.2f}–{high:.2f}")


if __name__ == "__main__":
    main()
