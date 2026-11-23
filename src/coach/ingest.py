import argparse
import re
import time

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from coach.config import DATA_DIR
from coach.data import TRAPS, load_problems, load_wiki_titles
from coach.store import COLLECTION, get_store

RAW_DIR = DATA_DIR / "raw"
HEADERS = {"User-Agent": "probability-coach/0.1 (student project)"}
HEADING = re.compile(r"^(==+)\s*(.*?)\s*\1\s*$", re.M)
SKIP_SECTIONS = {
    "See also", "References", "Notes", "Further reading", "External links", "Sources", "Bibliography",
    "См. также", "Примечания", "Литература", "Ссылки",
}


def fetch(lang, title):
    path = RAW_DIR / lang / f"{title}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")

    print(f"скачиваю {lang}: {title}")
    params = {"action": "query", "titles": title, "prop": "extracts", "explaintext": 1,
              "redirects": 1, "format": "json", "formatversion": 2}
    for attempt in range(5):
        resp = requests.get(f"https://{lang}.wikipedia.org/w/api.php", params=params, headers=HEADERS, timeout=30)
        if resp.status_code != 429:
            break
        time.sleep(5 * (attempt + 1))
    resp.raise_for_status()

    text = resp.json()["query"]["pages"][0]["extract"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    time.sleep(1.5)
    return text


def clean(text):
    parts = HEADING.split(text)
    kept, skipping = [parts[0]], False
    for i in range(1, len(parts), 3):
        level, heading, body = len(parts[i]), parts[i + 1], parts[i + 2]
        if level == 2:
            skipping = heading in SKIP_SECTIONS
        if not skipping:
            kept.append(f"{heading}\n{body}")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def theory_docs(chunk_size):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.15))
    docs = []
    for lang, title in load_wiki_titles():
        for i, chunk in enumerate(splitter.split_text(clean(fetch(lang, title)))):
            meta = {"kind": "theory", "lang": lang, "title": title, "chunk_id": f"{lang}:{title}:{i}"}
            docs.append(Document(chunk, metadata=meta))
    return docs


def problem_docs():
    # описание ловушки в тексте помогает находить задачи по запросам вида «дай задачу на ошибку игрока»
    return [
        Document(f"{p['text']}\n{TRAPS[p['trap']]}",
                 metadata={"kind": "problem", "problem_id": p["id"], "trap": p["trap"], "chunk_id": p["id"]})
        for p in load_problems()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--collection", default=COLLECTION)
    args = parser.parse_args()

    docs = theory_docs(args.chunk_size) + problem_docs()
    store = get_store(args.collection, pre_delete_collection=True)
    # таблица эмбеддингов общая для всех коллекций, поэтому id должен включать имя коллекции
    store.add_documents(docs, ids=[f"{args.collection}/{d.metadata['chunk_id']}" for d in docs])
    print(f"{args.collection}: {len(docs)} документов")


if __name__ == "__main__":
    main()
