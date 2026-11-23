import json
from collections import defaultdict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langsmith import Client, evaluate

from coach import config
from coach.agent import SYSTEM_PROMPT, chat_model
from coach.tools import PROBLEMS, TOOLS, format_problem

DATASET = "probability-coach-routing-v2"
REPEATS = 3
CASES = config.ROOT / "evaluation" / "routing_cases.json"


def shown_problem(problem_id):
    if not problem_id:
        return []
    problem = PROBLEMS[problem_id]
    call = {"name": "knowledge_search", "args": {"query": "задача", "kind": "problem"}, "id": "call_0"}
    return [
        HumanMessage("Дай задачу"),
        AIMessage("", tool_calls=[call]),
        ToolMessage(format_problem(problem), tool_call_id="call_0"),
        AIMessage(f"{problem['text']}\n\nВаш ответ и уверенность в процентах?"),
    ]


def routing(outputs, reference_outputs):
    expected = reference_outputs["tool"]
    calls = [c for c in outputs["calls"] if c["name"] == expected]
    results = [{"key": "tool_correct", "score": not outputs["calls"] if expected is None else bool(calls)}]
    if reference_outputs["args"]:
        args_ok = any(all(c["args"].get(k) == v for k, v in reference_outputs["args"].items()) for c in calls)
        results.append({"key": "args_correct", "score": args_ok})
    return {"results": results}


def main():
    client = Client()
    if not client.has_dataset(dataset_name=DATASET):
        cases = json.loads(CASES.read_text(encoding="utf-8"))
        dataset = client.create_dataset(DATASET, description="Реплики пользователя и инструмент, который должен выбрать агент")
        client.create_examples(dataset_id=dataset.id, examples=[
            {"inputs": {"message": c["message"], "problem_id": c["problem_id"]},
             "outputs": {"tool": c["tool"], "args": c["args"], "hard": c.get("hard", False)}}
            for c in cases
        ])

    model = chat_model().bind_tools(TOOLS)

    def first_step(inputs):
        messages = [SystemMessage(SYSTEM_PROMPT), *shown_problem(inputs["problem_id"]), HumanMessage(inputs["message"])]
        calls = model.invoke(messages).tool_calls
        return {"calls": [{"name": c["name"], "args": c["args"]} for c in calls]}

    # модель отвечает с температурой 0.3, поэтому каждую реплику прогоняем трижды
    results = evaluate(first_step, data=DATASET, evaluators=[routing], experiment_prefix="routing",
                       metadata={"model": config.DEEPSEEK_MODEL}, max_concurrency=4, num_repetitions=REPEATS)

    scores = defaultdict(list)
    for row in results:
        example = row["example"].outputs
        groups = ["все", "трудные" if example["hard"] else "однозначные", example["tool"] or "без инструмента"]
        for res in row["evaluation_results"]["results"]:
            for group in groups:
                scores[res.key, group].append(res.score)
            if not res.score:
                print(f"{res.key}: {row['example'].inputs['message']!r} -> {row['run'].outputs['calls']}")

    for (key, group), values in sorted(scores.items()):
        print(f"{key:13} {group:17} {sum(values)}/{len(values)}")


if __name__ == "__main__":
    main()
