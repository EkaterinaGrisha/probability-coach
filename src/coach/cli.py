import argparse
import logging
import uuid

from langgraph.checkpoint.postgres import PostgresSaver

from coach import config
from coach.agent import build_agent


def main():
    parser = argparse.ArgumentParser(description="Тренер вероятностной интуиции")
    parser.add_argument("--thread", help="id диалога, чтобы продолжить прошлый разговор")
    parser.add_argument("--user", default="default", help="чью статистику вести")
    args = parser.parse_args()
    thread = args.thread or uuid.uuid4().hex[:8]

    log_dir = config.ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        format=f"%(asctime)s [{thread}] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_dir / "agent.log", encoding="utf-8"), logging.StreamHandler()],
    )
    logging.getLogger("coach").setLevel(logging.INFO)

    run_config = {"configurable": {"thread_id": thread, "user_id": args.user}}
    with PostgresSaver.from_conn_string(config.DATABASE_URL) as checkpointer:
        checkpointer.setup()
        agent = build_agent(checkpointer)
        print(f"Диалог {thread}. Продолжить позже: --thread {thread}. Выход: Ctrl+D\n")
        while True:
            try:
                text = input("вы> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text:
                result = agent.invoke({"messages": [{"role": "user", "content": text}]}, run_config)
                print(f"\nтренер> {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
