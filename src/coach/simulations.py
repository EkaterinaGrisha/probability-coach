import math
import time
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coach.config import ROOT

OUTPUT_DIR = ROOT / "outputs"
rng = np.random.default_rng()


def _save(fig, name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _num(x):
    return f"{round(x):,}".replace(",", " ")


def _pct(x):
    if x == 0 or x >= 0.001:
        return f"{x:.1%}".replace(".", ",")
    return np.format_float_positional(x * 100, precision=1, fractional=False).replace(".", ",") + "%"


def monty_hall(n_doors=3, host_knows=True, trials=10_000):
    car = rng.integers(n_doors, size=trials)
    pick = rng.integers(n_doors, size=trials)
    random_other = (pick + rng.integers(1, n_doors, size=trials)) % n_doors
    other = np.where(car == pick, random_other, car) if host_knows else random_other
    # ведущий, который не знает, где приз, иногда открывает автомобиль; такие игры не идут в счёт
    revealed = (car != pick) & (car != other)
    stay = (pick == car)[~revealed]
    switch = (other == car)[~revealed]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 4.6), height_ratios=[1, 1.3])
    parts = [
        (f"первый выбор — автомобиль (игр: {_num(stay.sum())}), здесь выигрывает «не менять»", stay.sum(), "tab:blue"),
        (f"первый выбор — коза (игр: {_num(switch.sum())}), здесь выигрывает «сменить»", switch.sum(), "tab:orange"),
    ]
    if not host_knows:
        parts.append((f"ведущий открыл автомобиль (игр: {_num(revealed.sum())}), не в счёт", revealed.sum(), "lightgray"))
    left = 0
    for label, n, color in parts:
        top.barh(0, n, left=left, color=color, label=label)
        left += n
    top.set(xlim=(0, trials), yticks=[], xticks=[],
            title=f"Игр: {_num(trials)}, дверей: {n_doors}, ведущий {'знает' if host_knows else 'не знает'}, где автомобиль")
    top.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=9)
    top.spines[:].set_visible(False)

    bars = bottom.barh(["сменить", "не менять"], [switch.mean(), stay.mean()], color=["tab:orange", "tab:blue"])
    texts = bottom.bar_label(bars, labels=[f"{_num(w.sum())} из {_num(len(w))} ({_pct(w.mean())})" for w in (switch, stay)],
                             padding=4)
    for text in texts:
        text.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 1})
    bottom.axvline(0.5, color="gray", ls="--", label="интуиция: 50 на 50")
    bottom.set(xlim=(0, 1.15), xlabel="доля выигрышей")
    bottom.legend(loc="lower right", frameon=False)
    fig.tight_layout()

    return {
        "games": len(stay),
        "discarded": int(revealed.sum()),
        "first_pick_car": int(stay.sum()),
        "stay_win_rate": round(stay.mean(), 3),
        "switch_win_rate": round(switch.mean(), 3),
        "plot": _save(fig, "monty_hall"),
    }


def base_rate(prevalence, sensitivity, false_positive_rate, population=None,
              labels=("больны", "здоровы", "тест положительный", "тест отрицательный")):
    if population is None:
        # население берём таким, чтобы больных было не меньше сотни: так натуральные частоты получаются наглядными
        population = 10 ** int(np.ceil(np.log10(100 / prevalence)))
    yes, no, positive, negative = labels
    sick = population * prevalence
    healthy = population - sick
    true_pos = sick * sensitivity
    false_pos = healthy * false_positive_rate
    ppv = true_pos / (true_pos + false_pos)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set(xlim=(0, 4), ylim=(-0.3, 3))
    ax.axis("off")
    nodes = {
        "all": (2, 2.6, f"всего: {_num(population)}", "white"),
        "yes": (1, 1.6, f"{yes}: {_num(sick)}", "white"),
        "no": (3, 1.6, f"{no}: {_num(healthy)}", "white"),
        "tp": (0.5, 0.6, f"{positive}:\n{_num(true_pos)}", "#f4b6b6"),
        "fn": (1.5, 0.6, f"{negative}:\n{_num(sick - true_pos)}", "white"),
        "fp": (2.5, 0.6, f"{positive}:\n{_num(false_pos)}", "#f9d9a6"),
        "tn": (3.5, 0.6, f"{negative}:\n{_num(healthy - false_pos)}", "white"),
    }
    edges = [("all", "yes", prevalence), ("all", "no", 1 - prevalence), ("yes", "tp", sensitivity),
             ("yes", "fn", 1 - sensitivity), ("no", "fp", false_positive_rate), ("no", "tn", 1 - false_positive_rate)]
    for a, b, share in edges:
        (x1, y1, *_), (x2, y2, *_) = nodes[a], nodes[b]
        ax.plot([x1, x2], [y1 - 0.15, y2 + 0.2], color="gray", lw=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, _pct(share), ha="center", fontsize=8, color="gray",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    for x, y, text, color in nodes.values():
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round", "facecolor": color, "edgecolor": "gray"})
    ax.text(2, -0.2, f"{positive}: {_num(true_pos)} + {_num(false_pos)} = {_num(true_pos + false_pos)}. "
                     f"Из них {yes}: {_num(true_pos)}, это {_pct(ppv)}", ha="center", fontsize=10, weight="bold")

    return {
        "population": population,
        "sick": round(sick),
        "true_positives": round(true_pos),
        "false_positives": round(false_pos),
        "p_sick_if_positive": round(ppv, 4),
        "plot": _save(fig, "base_rate"),
    }


def birthday(n_people, mode="any_pair", trials=20_000):
    days = rng.integers(365, size=(trials, n_people))
    if mode == "any_pair":
        hits = np.any(np.diff(np.sort(days, axis=1), axis=1) == 0, axis=1)
    else:
        hits = np.any(days[:, 1:] == days[:, :1], axis=1)
    sizes = np.arange(1, max(80, n_people) + 1)
    any_pair = 1 - np.cumprod((366 - sizes) / 365)
    my_day = 1 - (364 / 365) ** (sizes - 1)
    pairs = n_people * (n_people - 1) // 2

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sizes, any_pair, lw=2.5 if mode == "any_pair" else 1,
            label=f"совпадение у любой пары (у {n_people} человек пар: {_num(pairs)})")
    ax.plot(sizes, my_day, lw=2.5 if mode == "my_birthday" else 1,
            label=f"кто-то родился в ваш день (сравнений: {n_people - 1})")
    ax.scatter([n_people], [hits.mean()], color="tab:red", zorder=3,
               label=f"симуляция для {n_people} человек: {_pct(hits.mean())}")
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set(xlabel="человек в группе", ylabel="вероятность", ylim=(0, 1),
           title="Пар растёт как квадрат числа людей,\nпоэтому совпадение у любой пары наступает быстро")
    ax.legend(loc="best", fontsize=9)

    return {
        "n_people": n_people,
        "pairs": pairs,
        "simulated": round(hits.mean(), 3),
        "any_pair": round(any_pair[n_people - 1], 4),
        "my_birthday": round(my_day[n_people - 1], 4),
        "plot": _save(fig, "birthday"),
    }


def gamblers_fallacy(streak_length, p=0.5, flips=1_000_000):
    x = (rng.random(flips) < p).astype(int)
    lengths = np.arange(1, streak_length + 1)
    rates, counts = [], []
    for k in lengths:
        streak_ends = np.convolve(x, np.ones(k), "valid")[:-1] == k
        following = x[k:][streak_ends]
        rates.append(following.mean())
        counts.append(len(following))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(lengths, rates)
    ax.bar_label(bars, labels=[f"{_pct(r)}\nсерий: {_num(c)}" for r, c in zip(rates, counts)], fontsize=8)
    ax.axhline(p, color="tab:red", ls="--", label=f"вероятность без всякой серии: {_pct(p)}")
    ax.set(xlabel="сколько раз подряд выпал один исход", ylabel="тот же исход в следующем броске", ylim=(0, 1),
           xticks=lengths, title=f"Бросков: {_num(flips)}. После любой серии доля та же")
    ax.legend(loc="upper right")

    return {
        "p": p,
        "after_streak": round(rates[-1], 3),
        "streaks_found": counts[-1],
        "plot": _save(fig, "gamblers_fallacy"),
    }


def small_samples(sizes, more_than, p=0.5, trials=10_000):
    fig, axes = plt.subplots(len(sizes), 1, figsize=(8, 1.3 + 1.7 * len(sizes)), sharex=True, squeeze=False)
    samples = []
    for ax, n in zip(axes[:, 0], sizes):
        # round убирает погрешность вида 0.69 * 100 = 68.99999999999999
        k_min = math.floor(round(more_than * n, 9)) + 1
        exact = sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(k_min, n + 1))
        counts = rng.binomial(n, p, size=trials)
        values, freq = np.unique(counts, return_counts=True)
        ax.bar(values / n, freq / trials, width=0.8 / n, color=np.where(values >= k_min, "tab:red", "tab:gray"))
        ax.set_title(f"выборка из {n}: {k_min} из {n} и больше в {_pct(exact)} случаев", fontsize=10)
        samples.append({"size": n, "min_count": k_min, "exact": round(exact, 5),
                        "simulated": round(float(np.mean(counts >= k_min)), 4)})
    axes[-1, 0].set_xlabel("доля в выборке")
    fig.supylabel("доля выборок")
    fig.suptitle(f"Случайных выборок каждого размера: {_num(trials)}. Маленькие чаще дают крайние значения")
    fig.tight_layout()

    return {"more_than": more_than, "samples": samples, "plot": _save(fig, "small_samples")}


def regression_to_mean(reliability=0.5, share=0.1, people=10_000):
    skill = rng.standard_normal(people)
    luck = np.sqrt(1 - reliability) * rng.standard_normal((2, people))
    first, second = 60 + 10 * (np.sqrt(reliability) * skill + luck)
    top = first >= np.quantile(first, 1 - share)
    bottom = first <= np.quantile(first, share)
    groups = [(f"лучшие {share:.0%} в первой попытке", top, "tab:red"),
              ("все", np.ones(people, dtype=bool), "gray"),
              (f"худшие {share:.0%} в первой попытке", bottom, "tab:blue")]

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, mask, color in groups:
        a, b = first[mask].mean(), second[mask].mean()
        ax.plot([0, 1], [a, b], marker="o", color=color, label=label)
        ax.annotate(f"{a:.1f}", (0, a), xytext=(-30, 0), textcoords="offset points", va="center")
        ax.annotate(f"{b:.1f}", (1, b), xytext=(8, 0), textcoords="offset points", va="center")
    ax.set(xticks=[0, 1], xticklabels=["первая попытка", "вторая попытка"], xlim=(-0.3, 1.3), ylabel="средний результат",
           title=f"Между попытками ничего не меняли.\nРезультат на {reliability:.0%} зависит от умения, остальное — случайность")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    return {
        "reliability": reliability,
        "share": share,
        "top_first": round(first[top].mean(), 1),
        "top_second": round(second[top].mean(), 1),
        "bottom_first": round(first[bottom].mean(), 1),
        "bottom_second": round(second[bottom].mean(), 1),
        "plot": _save(fig, "regression_to_mean"),
    }


def two_children(condition="any_boy", families=100_000):
    older, younger = rng.random((2, families)) < 0.5
    known = older | younger if condition == "any_boy" else older
    both = older & younger
    kinds = {"ММ": older & younger, "МД": older & ~younger, "ДМ": ~older & younger, "ДД": ~older & ~younger}
    condition_text = "хотя бы один мальчик" if condition == "any_boy" else "старший — мальчик"

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, mask in kinds.items():
        fits = bool(np.all(known[mask]))
        ax.bar(name, mask.sum(), color="tab:orange" if name == "ММ" else "tab:blue" if fits else "lightgray",
               hatch="" if fits else "//", edgecolor="white")
        ax.text(name, mask.sum(), _num(mask.sum()), ha="center", va="bottom")
    ax.margins(y=0.12)
    ax.set(ylabel="семей", xlabel="старший и младший ребёнок (М — мальчик, Д — девочка);\n"
                                 "заштрихованы семьи, которые условие исключает",
           title=f"Условие «{condition_text}». Подходящих семей: {_num(known.sum())},\n"
                 f"из них с двумя мальчиками: {_num(both.sum())}, это {_pct(both.sum() / known.sum())}")

    return {
        "condition": condition,
        "families": int(known.sum()),
        "both_boys": int(both.sum()),
        "p_both_boys": round(both.sum() / known.sum(), 3),
        "plot": _save(fig, "two_children"),
    }


def simpson(groups):
    """Доля успехов по подгруппам и в сумме; groups: {вариант: {подгруппа: [успехов, всего]}}."""
    options = list(groups)
    subgroups = list(groups[options[0]])
    rows = {}
    for option in options:
        cells = [groups[option][s] for s in subgroups]
        rows[option] = cells + [[sum(c[0] for c in cells), sum(c[1] for c in cells)]]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(subgroups) + 1)
    width = 0.8 / len(options)
    for i, option in enumerate(options):
        bars = ax.bar(x + i * width, [s / n for s, n in rows[option]], width, label=option)
        ax.bar_label(bars, labels=[f"{_pct(s / n)}\n{s} из {n}" for s, n in rows[option]], fontsize=8)
    ax.set(xticks=x + width * (len(options) - 1) / 2, xticklabels=subgroups + ["всего"], ylim=(0, 1.2),
           ylabel="доля успехов", title="Доля успехов в каждой подгруппе и в сумме")
    ax.legend(loc="upper right")

    return {
        "rates": {o: {name: round(s / n, 3) for name, (s, n) in zip(subgroups + ["всего"], rows[o])} for o in options},
        "plot": _save(fig, "simpson"),
    }


SIMULATIONS = {
    "monty_hall": monty_hall,
    "base_rate": base_rate,
    "birthday": birthday,
    "gamblers_fallacy": gamblers_fallacy,
    "small_samples": small_samples,
    "regression_to_mean": regression_to_mean,
    "two_children": two_children,
    "simpson": simpson,
}
