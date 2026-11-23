import json
import logging
import re
import uuid

import psycopg
import streamlit as st
from langgraph.checkpoint.postgres import PostgresSaver
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from psycopg.rows import dict_row

from coach import config, stats
from coach.agent import build_agent
from coach.data import TRAPS
from coach.search import search

st.set_page_config(page_title="Тренер вероятностной интуиции", layout="wide")


@st.cache_resource(show_spinner="Загружаю модели и подключаюсь к базе…")
def get_agent():
    (config.ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(format="%(asctime)s [web] %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(config.ROOT / "logs" / "agent.log", encoding="utf-8")])
    logging.getLogger("coach").setLevel(logging.INFO)

    conn = psycopg.connect(config.DATABASE_URL, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    search("прогрев", 1, {"kind": "theory"})
    return build_agent(checkpointer)


def show_steps(steps):
    results = {m.tool_call_id: m.content for m in steps if m.type == "tool"}
    calls = [call for m in steps if m.type == "ai" for call in m.tool_calls]
    chunks, log = [], []
    for call in calls:
        result = results.get(call["id"], "")
        line = f"`{call['name']}` {json.dumps(call['args'], ensure_ascii=False)}"
        if call["name"] == "simulate" and result.startswith("{"):
            st.image(json.loads(result)["plot"], width=640)
        elif call["name"] == "knowledge_search" and call["args"].get("kind") == "problem":
            line += " → " + ", ".join(re.findall(r"^id: (\S+)", result, re.M))
        elif call["name"] == "knowledge_search":
            chunks += [c for c in re.split(r"\n\n(?=\[Источник: )", result) if c.startswith("[Источник: ")]
        log.append(line)

    if chunks:
        with st.expander(f"Источники, фрагментов: {len(chunks)}"):
            for chunk in chunks:
                source, _, text = chunk.partition("\n")
                st.caption(f"{source} {text[:300]}…")
    st.caption("Инструменты: " + "; ".join(log))


def show_dialog(messages):
    steps = []
    for m in messages:
        if m.type == "human":
            st.chat_message("user").markdown(m.content)
            steps = []
        elif m.type == "tool" or m.tool_calls:
            steps.append(m)
        else:
            with st.chat_message("assistant"):
                st.markdown(m.content)
                if steps:
                    show_steps(steps)
            steps = []


def show_calibration(bins):
    fig = Figure(figsize=(3.2, 3.1))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="уверенность = точность")
    x, y = [b["mean_confidence"] for b in bins], [b["accuracy"] for b in bins]
    ax.plot(x, y, color="tab:blue", lw=1)
    ax.scatter(x, y, s=[25 + 15 * b["attempts"] for b in bins], color="tab:blue", zorder=3, label="ваши ответы")
    for b in bins:
        ax.annotate(str(b["attempts"]), (b["mean_confidence"], b["accuracy"]), xytext=(6, -10),
                    textcoords="offset points", fontsize=7)
    ax.set(xlim=(0, 1.05), ylim=(-0.05, 1.05), xlabel="уверенность", ylabel="доля верных")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Ответы сгруппированы по уверенности (0–20%, …, 80–100%), цифра у точки — сколько их. "
               "Точки ниже диагонали — сверхуверенность.")


def show_stats(user):
    summary = stats.summary(user)
    if not summary["attempts"]:
        st.caption("Пока нет решённых задач.")
        return
    left, right = st.columns(2)
    left.metric("Попыток", summary["attempts"])
    right.metric("Верных", f"{summary['accuracy']:.0%}")
    left.metric("Уверенность", f"{summary['mean_confidence']:.0%}")
    right.metric("Brier", summary["brier"])
    show_calibration(summary["calibration"])
    st.dataframe(
        [{"ловушка": TRAPS[t["trap"]], "попыток": t["attempts"], "верных": f"{t['accuracy']:.0%}"} for t in summary["by_trap"]],
        hide_index=True,
    )


agent = get_agent()

with st.sidebar:
    st.header("Тренер вероятностной интуиции")
    user = st.text_input("Пользователь", value="default")
    new_dialog = st.button("Новый диалог")
    if new_dialog or "thread" not in st.query_params:
        st.query_params["thread"] = uuid.uuid4().hex[:8]
    st.caption(f"Диалог `{st.query_params['thread']}`")
    st.subheader("Статистика")
    show_stats(user)

run_config = {"configurable": {"thread_id": st.query_params["thread"], "user_id": user}}
show_dialog(agent.get_state(run_config).values.get("messages", []))

if prompt := st.chat_input("Попросите задачу, ответьте или задайте вопрос"):
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"), st.spinner("Думаю…"):
        agent.invoke({"messages": [{"role": "user", "content": prompt}]}, run_config)
    st.rerun()
