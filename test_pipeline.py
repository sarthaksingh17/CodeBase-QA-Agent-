"""Quick end-to-end pipeline test."""
from router import ask

question = "How does Click handle command groups?"
print(f"Question: {question}\n")

result = ask(question, "repo_click")

print("=" * 60)
print("ANSWER:")
print("=" * 60)
print(result["answer"])

print("\n" + "=" * 60)
print("SOURCES:")
print("=" * 60)
for s in result["sources"]:
    print(f"  [{s['score']:.3f}] {s['name']} ({s['type']}) -> {s['file']} L{s['lines']}")
