import json

from coach.config import DATA_DIR

TRAPS = {
    "monty_hall": "задачи типа Монти Холла: подсказка ведущего, который знает ответ",
    "base_rate": "игнорирование базовой частоты, ошибка прокурора",
    "conjunction": "ошибка конъюнкции, эвристика репрезентативности",
    "gamblers_fallacy": "ошибка игрока: вера в память у случайности",
    "birthday": "недооценка совпадений, парадокс дней рождения",
    "crt": "интуитивный ответ без проверки, тест когнитивной рефлексии",
    "small_samples": "нечувствительность к размеру выборки",
    "simpson": "парадокс Симпсона",
    "regression_to_mean": "регрессия к среднему",
    "conditional": "неверное условие в условной вероятности",
}


def load_problems():
    return json.loads((DATA_DIR / "problems.json").read_text(encoding="utf-8"))


def load_wiki_titles():
    articles = []
    for line in (DATA_DIR / "wiki_titles.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lang, title = line.split("|", 1)
            articles.append((lang.strip(), title.strip()))
    return articles
