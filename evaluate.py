"""
Evaluation module using RAGAS framework.
Measures retrieval quality and answer faithfulness.
"""

import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from router import ask

load_dotenv()

# ── Test questions for evaluation ──────────────────────────────
# Customize these for your specific repo
EVAL_QUESTIONS = [
    {
        "question": "How does Click handle command groups?",
        "ground_truth": "Click handles command groups through the Group class which extends MultiCommand. Groups maintain a dictionary of commands and can be nested.",
    },
    {
        "question": "What is the purpose of the Context class in Click?",
        "ground_truth": "The Context class holds state for the entire CLI invocation, including parameters, parent context, and configuration. It manages the command execution lifecycle.",
    },
    {
        "question": "How does Click parse command line arguments?",
        "ground_truth": "Click uses parameter decorators (option, argument) that define how CLI arguments are parsed. The parsing is handled by the BaseCommand.parse_args method which processes the argument list.",
    },
    {
        "question": "What decorators does Click provide for defining CLI commands?",
        "ground_truth": "Click provides decorators like @click.command(), @click.group(), @click.option(), @click.argument(), and @click.pass_context for defining CLI commands and their parameters.",
    },
    {
        "question": "How does Click handle help text generation?",
        "ground_truth": "Click automatically generates help text from docstrings and parameter definitions. The format_help method in BaseCommand handles formatting, and --help is automatically added.",
    },
]


def run_evaluation(
    collection_name: str = "repo_click",
    top_k_search: int = 20,
    top_k_rerank: int = 5,
) -> dict:
    """
    Run RAGAS evaluation against test questions.
    Returns evaluation metrics.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"🔄 Running evaluation on {len(EVAL_QUESTIONS)} questions...")

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        q = item["question"]
        gt = item["ground_truth"]

        print(f"\n  [{i}/{len(EVAL_QUESTIONS)}] {q}")

        result = ask(
            question=q,
            collection_name=collection_name,
            top_k_search=top_k_search,
            top_k_rerank=top_k_rerank,
        )

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([src["code"] for src in result["sources"]])
        ground_truths.append(gt)

        print(f"  ✅ Got answer ({len(result['sources'])} sources)")

    # Build RAGAS dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print(f"\n🔄 Computing RAGAS metrics...")

    # Run evaluation
    results = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    print(f"\n{'='*50}")
    print(f"📊 RAGAS Evaluation Results")
    print(f"{'='*50}")
    for metric, score in results.items():
        if isinstance(score, (int, float)):
            bar = "█" * int(score * 20)
            print(f"  {metric:25s}: {score:.4f} {bar}")
    print(f"{'='*50}")

    return dict(results)


if __name__ == "__main__":
    run_evaluation()
