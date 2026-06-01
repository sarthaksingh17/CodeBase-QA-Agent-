import os
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import tree_sitter_java as tsjava
import tree_sitter_go as tsgo

# ── Language registry ──────────────────────────────────────────
LANGUAGE_MAP = {
    ".py": {
        "language": Language(tspython.language()),
        "nodes": ["function_definition", "class_definition", "decorated_definition"],
    },
    ".js": {
        "language": Language(tsjavascript.language()),
        "nodes": ["function_declaration", "class_declaration", "arrow_function"],
    },
    ".ts": {
        "language": Language(tstypescript.language_typescript()),
        "nodes": ["function_declaration", "class_declaration", "method_definition"],
    },
    ".java": {
        "language": Language(tsjava.language()),
        "nodes": ["method_declaration", "class_declaration"],
    },
    ".go": {
        "language": Language(tsgo.language()),
        "nodes": ["function_declaration", "method_declaration"],
    },
}

# ── Skip these folders entirely ────────────────────────────────
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__",
    ".venv", "venv", "dist", "build", ".next"
}

# ── Skip these file patterns ───────────────────────────────────
SKIP_PATTERNS = [
    "test_", "_test", ".min.js",
    "migrations", "generated", "dist"
]


def get_node_name(node, source: bytes) -> str:
    """Extract function or class name from AST node."""
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte:child.end_byte].decode("utf-8")
    return "unknown"


def should_skip_file(file_path: str) -> bool:
    """Check if file should be skipped."""
    return any(pattern in file_path for pattern in SKIP_PATTERNS)


def chunk_file(file_path: str, repo_name: str) -> list[dict]:
    """
    Parse a single file and extract function/class level chunks.
    Returns list of chunk dicts with code + metadata.
    """
    ext = os.path.splitext(file_path)[1]

    # Skip unsupported extensions
    if ext not in LANGUAGE_MAP:
        return []

    # Skip test/generated files
    if should_skip_file(file_path):
        return []

    try:
        with open(file_path, "rb") as f:
            source = f.read()
    except Exception:
        return []

    # Skip empty files
    if not source.strip():
        return []

    lang_config = LANGUAGE_MAP[ext]
    parser = Parser(lang_config["language"])
    target_nodes = lang_config["nodes"]

    tree = parser.parse(source)
    chunks = []

    def traverse(node):
        if node.type in target_nodes:
            code = source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            name = get_node_name(node, source)

            # Skip very small chunks (likely stubs)
            if len(code.strip()) < 30:
                return

            chunks.append({
                "code": code,
                "name": name,
                "type": node.type,
                "file": file_path,
                "repo": repo_name,
                "language": ext.lstrip("."),
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                # Enriched text for embedding
                "enriched": f"""
Repo: {repo_name}
File: {file_path}
Function: {name}
Language: {ext.lstrip('.')}
Code:
{code}
""".strip()
            })
            # Don't traverse deeper to avoid double chunking
            return

        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return chunks


def chunk_repo(repo_path: str, repo_name: str) -> list[dict]:
    """
    Walk entire repo directory and chunk all supported files.
    Returns all chunks across all files.
    """
    all_chunks = []

    for root, dirs, files in os.walk(repo_path):
        # Skip noise directories in place
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for file in files:
            file_path = os.path.join(root, file)
            chunks = chunk_file(file_path, repo_name)
            all_chunks.extend(chunks)

    print(f"✅ {repo_name} → {len(all_chunks)} chunks extracted")
    return all_chunks