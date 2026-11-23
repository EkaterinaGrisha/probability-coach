from collections import Counter

from coach.data import TRAPS, load_problems, load_wiki_titles
from coach.simulations import SIMULATIONS

FIELDS = {"id", "trap", "text", "answer", "intuitive_answer", "explanation", "simulation", "sim_params"}


def test_problems():
    problems = load_problems()
    assert len({p["id"] for p in problems}) == len(problems)
    for p in problems:
        assert set(p) == FIELDS, p["id"]
        assert p["trap"] in TRAPS, p["id"]
        assert (p["simulation"] is None) == (p["sim_params"] is None), p["id"]
        assert p["simulation"] is None or p["simulation"] in SIMULATIONS, p["id"]


def test_each_trap_has_problems():
    counts = Counter(p["trap"] for p in load_problems())
    assert all(counts[trap] >= 2 for trap in TRAPS)


def test_wiki_titles():
    articles = load_wiki_titles()
    assert len(articles) == len(set(articles))
    assert {lang for lang, _ in articles} == {"en", "ru"}
