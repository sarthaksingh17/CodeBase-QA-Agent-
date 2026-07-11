"""
Generator: Context building + LLM generation via Groq (Llama 3.3 70B).
Stages 10 and 11 of the pipeline.
"""

import os
from groq import Groq
from config import GROQ_API_KEY

groq_client = Groq(api_key=GROQ_API_KEY)

# ── System prompt for code-aware QA ────────────────────────────
SYSTEM_PROMPT = """You are a senior software engineer analyzing a codebase.

Rules:
1. Answer ONLY based on the code context provided below.
2. Always reference specific files, functions, and line numbers in your answer.
3. Explain the code logic clearly — don't just paste code.
4. If the context is insufficient to fully answer the question, say so clearly.
5. Use markdown formatting for readability (headers, code blocks, bullet points).
6. When mentioning code elements, use inline code formatting like `function_name()`.
"""


def build_context(ranked_chunks: list[tuple]) -> str:
    """
    Stage 10: Build structured context from reranked chunks.

    Takes list of (ScoredPoint, rerank_score) tuples.
    Returns formatted context string for LLM consumption.
    """
    if not ranked_chunks:
        return "No relevant code context found."

    context_parts = []

    for i, (chunk, score) in enumerate(ranked_chunks, 1):
        payload = chunk.payload
        lang = payload.get("language", "")
        file_path = payload.get("file", "unknown")
        name = payload.get("name", "unknown")
        chunk_type = payload.get("type", "unknown")
        start_line = payload.get("start_line", "?")
        end_line = payload.get("end_line", "?")
        code = payload.get("code", "")

        context_parts.append(
            f"### Source {i} (relevance: {score:.2f})\n"
            f"**File**: `{file_path}`\n"
            f"**Symbol**: `{name}` ({chunk_type})\n"
            f"**Lines**: {start_line}–{end_line}\n\n"
            f"```{lang}\n{code}\n```"
        )

    return "\n\n---\n\n".join(context_parts)


def generate(
    question: str,
    ranked_chunks: list[tuple],
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> dict:
    """
    Stage 11: Send context + question to Groq LLM and get answer.

    Returns dict with: answer, sources, model.
    """
    context = build_context(ranked_chunks)

    user_message = f"""## Code Context

{context}

## Question

{question}"""

    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    answer = response.choices[0].message.content

    # Build source references for the UI
    sources = []
    for chunk, score in ranked_chunks:
        p = chunk.payload
        sources.append({
            "file":     p.get("file", "unknown"),
            "name":     p.get("name", "unknown"),
            "type":     p.get("type", "unknown"),
            "language": p.get("language", ""),
            "lines":    f"{p.get('start_line', '?')}–{p.get('end_line', '?')}",
            "score":    round(float(score), 3),
            "code":     p.get("code", ""),
        })

    return {
        "answer":  answer,
        "sources": sources,
        "model":   model,
    }
