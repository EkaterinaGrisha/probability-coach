import json
import random
from collections import defaultdict

from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

from coach import config
from coach.ingest import theory_docs

OUT = config.ROOT / "evaluation" / "retrieval_questions.json"
PER_ARTICLE = 2

PROMPT = """Ниже фрагмент статьи Википедии. Придумай один вопрос на русском языке, который мог бы задать человек, разбирающийся с вероятностными ловушками, и ответ на который есть в этом фрагменте.

Требования:
- вопрос не повторяет формулировки фрагмента дословно и не упоминает «фрагмент», «текст» или «статью»;
- anchor — точная цитата из фрагмента на его языке, 4–10 слов подряд, в которой содержится ответ;
- если во фрагменте нет содержательной информации (перечень ссылок, одни заголовки), верни пустые строки.

Фрагмент:
{text}"""


class QA(BaseModel):
    question: str
    anchor: str = Field(description="дословная цитата из фрагмента, 4–10 слов")


def norm(text):
    return " ".join(text.lower().split())


def main():
    by_article = defaultdict(list)
    for d in theory_docs(1000):
        if len(d.page_content) >= 400:
            by_article[d.metadata["lang"], d.metadata["title"]].append(d)
    rnd = random.Random(42)
    sample = [d for chunks in by_article.values() for d in rnd.sample(chunks, min(PER_ARTICLE, len(chunks)))]

    llm = ChatDeepSeek(model=config.DEEPSEEK_MODEL, temperature=0.2).with_structured_output(QA)
    answers = llm.batch([PROMPT.format(text=d.page_content) for d in sample], config={"max_concurrency": 8})

    questions = [
        {"question": a.question, "anchor": a.anchor, **{k: d.metadata[k] for k in ("title", "lang", "chunk_id")}}
        for d, a in zip(sample, answers)
        if a.question and len(a.anchor.split()) >= 4 and norm(a.anchor) in norm(d.page_content)
    ]
    OUT.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(questions)} из {len(sample)} вопросов прошли проверку -> {OUT.name}")


if __name__ == "__main__":
    main()
