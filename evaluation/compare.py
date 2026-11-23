import argparse
from collections import defaultdict

import numpy as np
from langsmith import Client

from coach import config  # noqa: F401  ключи LangSmith загружаются из .env при импорте


def scores(client, experiment):
    runs = list(client.list_runs(project_name=experiment, is_root=True))
    ids = [r.id for r in runs]
    by_run = defaultdict(dict)
    for i in range(0, len(ids), 50):
        for fb in client.list_feedback(run_ids=ids[i:i + 50]):
            by_run[fb.run_id][fb.key] = fb.score
    return {r.reference_example_id: by_run[r.id] for r in runs}


def main():
    parser = argparse.ArgumentParser(description="Парный бутстреп: на сколько второй эксперимент лучше первого на тех же примерах")
    parser.add_argument("base")
    parser.add_argument("other")
    args = parser.parse_args()

    client = Client()
    a, b = scores(client, args.base), scores(client, args.other)
    common = [i for i in a if i in b]
    rng = np.random.default_rng(0)
    print(f"общих вопросов: {len(common)}")
    for key in sorted(a[common[0]]):
        diff = np.array([b[i][key] - a[i][key] for i in common], dtype=float)
        low, high = np.percentile(rng.choice(diff, (5000, len(diff))).mean(axis=1), [2.5, 97.5])
        print(f"{key:17} {diff.mean():+.3f}  95% ДИ {low:+.3f}…{high:+.3f}  лучше: {sum(diff > 0)}, хуже: {sum(diff < 0)}")


if __name__ == "__main__":
    main()
